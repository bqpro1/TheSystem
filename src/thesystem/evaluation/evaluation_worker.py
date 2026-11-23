"""Evaluate the fixed release checkpoints. No training actions are available."""
import argparse
import importlib.metadata
from pathlib import Path

from .evaluation_data import ROOT, load_suites, tokenize
from .evaluation_io import (bind_run, digest, evaluate_stage, exclusive_run, heartbeat,
                           outside_package, read_json, read_stage, sha, write_json)
from .evaluation_models import load_backend
from .writing_checks import check_writing
from thesystem.reproduction.common import implementation_fingerprint


def decision_batch(backend, rows):
    values = backend.decisions(rows)
    return [dict(value, id=row["id"], group_id=row["group_id"], dataset=row["family"],
                 kind=row["kind"], target=row["target_index"], option_keys=row["option_keys"],
                 input_sha256=row["input_sha256"])
            for row, value in zip(rows, values)]


def routing_batch(backend, rows):
    probabilities = backend.routing(rows)
    return [dict(id=row["id"], group_id=row["group_id"], target=row["mode_target"],
                 probability=p, route=int(p >= .5), input_sha256=row["input_sha256"])
            for row, p in zip(rows, probabilities)]


def run(args):
    from thesystem.artifacts import resolve_release_resources
    resolve_release_resources(ROOT)
    from .evaluation_data import audit_sources
    audit_sources()
    folder, cache = outside_package(args.output), outside_package(args.cache)
    identity = dict(package_sha256=sha(ROOT / "RELEASE-MANIFEST.json"), arm=args.arm,
                    batch_size=args.batch_size, generation_caps=[512, 1024], seed=42, smoke=args.smoke,
                    source_sha256=implementation_fingerprint(),
                    packages={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "numpy", "safetensors")})
    bind_run(folder, identity)
    with exclusive_run(folder):
        run_locked(args, folder, cache, identity)


def run_locked(args, folder, cache, identity):
    if (folder / "COMPLETE.json").exists():
        if read_json(folder / "COMPLETE.json") != dict(complete=True, identity=digest(identity)):
            raise ValueError("Completed run identity changed")
        suites = load_suites(smoke=args.smoke)
        expected = {**{"decisions/" + k: v for k,v in suites["decisions"].items()},
                    **{k: suites[k] for k in ("paired_writing", "routing_historical", "writing")}}
        for name, rows in expected.items():
            if [r['id'] for r in read_stage(folder / name)] != [r['id'] for r in rows]:
                raise ValueError("Completed run lost or changed examples")
        print(f"Verified completed {args.arm}; no backbone load or inference needed.", flush=True)
        return
    state = dict(stage="Setup", completed=0, planned=1)
    try:
        with heartbeat(folder, state):
            backend, environment = load_backend(args.arm, cache, state)
            environment_path = folder / "environment.json"
            if environment_path.exists():
                if read_json(environment_path) != environment:
                    raise ValueError("GPU environment changed; use a new output folder")
            write_json(environment_path, environment)
            suites = load_suites(smoke=args.smoke)
            write_json(folder / "token-audit.json", tokenize(suites, backend.tokenizer))
            sampled = [row for rows in suites["decisions"].values() for row in rows[:3]]
            write_json(folder / "dispatch-checks.json", backend.dispatch_checks(sampled))
            for name, rows in suites["decisions"].items():
                evaluate_stage(rows, folder / "decisions" / name, identity,
                               lambda batch: decision_batch(backend, batch), state, args.batch_size)
            for name in ("paired_writing", "routing_historical"):
                evaluate_stage(suites[name], folder / name, identity,
                               lambda batch: routing_batch(backend, batch), state, args.batch_size)
            evaluate_stage(suites["writing"], folder / "writing", identity,
                           lambda batch: [check_writing(backend, batch[0])], state,
                           batch_size=1, shard_size=1)
            state.update(stage="Complete")
            write_json(folder / "COMPLETE.json", dict(complete=True, identity=digest(identity)))
    except BaseException as error:
        import torch
        detail = str(error)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            detail = (
                "GPU memory ran out during evaluation. This run did not complete. "
                "Stop other GPU workloads or select a GPU with more memory, then restart "
                "and retry. The worker uses one model and a batch size of one by default. "
                "Keep benchmark prompts, precision and generation limits unchanged. "
                "Saved completed examples are preserved; an environment change requires a new run folder."
            )
            print(detail, flush=True)
        state.update(stage="Interrupted or failed", error=type(error).__name__, detail=detail)
        write_json(folder / "STATUS.json", state)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise RuntimeError(detail) from None
        raise
    write_json(folder / "STATUS.json", state)
    print(f"Completed {args.arm}: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Tiny mechanics check, not benchmark accuracy")
    parser.add_argument("--arm", choices=("probe", "lora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
