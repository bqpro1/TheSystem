"""Opt-in GPU save/resume acceptance against the installed, unchanged release wheel.

This is not collected by pytest. --self-test uses CPU fixtures only.
--run uses real Qwen and production smoke recipes; it never changes release assets.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

PAUSED = 77
STAGES = {
    "probe": "probe/run/training/seed-42/mlp256",
    "lora": "lora/original/stage1-lora",
    "original-fast": "lora/original/stage2-fast",
    "diverse-fast": "lora/diverse-gate/training",
}
EXPORTS = (
    "probe/run/export/heads.safetensors",
    "lora/original/exports/lora/weights.safetensors",
    "lora/original/exports/original-fast/weights.safetensors",
    "lora/diverse-gate/export/weights.safetensors",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def worker(args):
    """Only pause controls are injected; production data/settings stay untouched."""
    os.environ["THESYSTEM_RELEASE_ROOT"] = str(args.root)
    argv = ["gpu-acceptance", "train", "--output", str(args.output),
            "--cache", str(args.cache), "--smoke"]
    if args.worker == "probe":
        from thesystem.reproduction import _probe_runner as runner
        if args.stop == "probe":
            runner.train_probe = partial(runner.train_probe, stop_after_updates=1)
    else:
        from thesystem.reproduction import _lora_runner as runner
        if args.stop == "lora":
            runner.train_lora = partial(runner.train_lora, stop_after_updates=1)
        elif args.stop == "original-fast":
            runner.train_fast = partial(runner.train_fast, stop_after_updates=1)
        elif args.stop == "diverse-fast":
            from thesystem._vendor.fast import gate_training
            gate_training.fit_gate = partial(gate_training.fit_gate, stop_after=1)
        if args.worker == "diverse":
            argv[1] = "train-fast"
            argv += ["--initial-gate", str(args.output.parent / "original/exports/original-fast/weights.safetensors")]
        elif args.worker == "evaluate":
            argv[1] = "evaluate"
            argv += ["--lora-weights", str(args.output.parent / "original/exports/lora/weights.safetensors"),
                     "--gate-weights", str(args.output.parent / "diverse-gate/export/weights.safetensors")]
    sys.argv = argv
    try:
        runner.main()
    except (KeyboardInterrupt, InterruptedError):
        if not args.stop:
            raise
        traceback.print_exc()
        return PAUSED
    if args.stop:
        raise AssertionError("Requested pause was ignored; completed training is not resume evidence")
    return 0


def run_child(command, log):
    started, stop = time.monotonic(), threading.Event()
    print("RUN", " ".join(command), flush=True)
    with Path(log).open("w", buffering=1) as handle:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True)
        def heartbeat():
            while not stop.wait(30):
                message = f"ACCEPTANCE heartbeat: PID {child.pid}, {time.monotonic()-started:.0f}s, log {log}\n"
                print(message, end="", flush=True)
                handle.write(message)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            return child.wait()
        except BaseException:
            # Never leave a GPU child running when the controller is interrupted.
            for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                if child.poll() is not None:
                    break
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
            child.wait()
            raise
        finally:
            stop.set()
            thread.join()


def checkpoint(folder, stage):
    """Read only locally generated test checkpoints, using production hash checks."""
    if stage == "probe":
        from thesystem._vendor.probe.probe_training import _read_checked
        files = sorted(folder.glob("checkpoint-*.ptc"))
        if not files:
            raise AssertionError(f"No probe checkpoint: {folder}")
        payload = _read_checked(files[-1])
        return payload, [folder / "identity.json", files[-1]]
    if stage == "diverse-fast":
        from thesystem._vendor.fast.gate_reference_training import Checkpoints
    else:
        from thesystem._vendor.lora.training import Checkpoints
    commits = sorted(folder.glob("checkpoint-*.commit.json"))
    if not commits:
        raise AssertionError(f"No committed checkpoint: {folder}")
    identity = folder / "identity.json"
    payload = Checkpoints(folder, read(identity))._valid(commits[-1])
    return payload, [identity, commits[-1], folder / read(commits[-1])["file"]]


def require_paused(payload):
    state = payload["state"]
    if state.get("completed") is not False or state.get("updates") != 1 or state.get("examples_seen", 0) <= 0:
        raise AssertionError(f"Expected incomplete checkpoint after exactly one update: {state}")
    return {key: state.get(key) for key in ("completed", "epoch", "updates", "cursor", "examples_seen")}


def tensor_leaves(value, path="root"):
    import torch
    if isinstance(value, torch.Tensor):
        return {path: value}
    result = {}
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda pair: repr(pair[0]))
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return result
    for key, item in items:
        result.update(tensor_leaves(item, f"{path}/{key!r}"))
    return result


def compare_tensors(left, right):
    import torch
    left, right = tensor_leaves(left), tensor_leaves(right)
    if not left or left.keys() != right.keys():
        raise AssertionError("Missing or different checkpoint tensor structure")
    mismatches = [key for key in left if left[key].dtype != right[key].dtype
                  or left[key].shape != right[key].shape or not torch.equal(left[key], right[key])]
    if mismatches:
        raise AssertionError(f"Resume differs from uninterrupted run: {mismatches[:10]}")
    return dict(exact=True, tensors=len(left), elements=sum(t.numel() for t in left.values()))


def acceptance(args):
    import torch
    from thesystem.artifacts import resolve_release_resources
    from thesystem._vendor.lora.setup_runtime import verify_runtime
    verify_runtime()
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 40 * 2**30 or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("Acceptance requires native BF16 and at least 40 GiB total GPU memory")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; existing acceptance evidence is never overwritten")
    root = Path(resolve_release_resources(args.root)).resolve()
    cache = args.cache.expanduser().resolve()
    for path in (output, cache):
        if path == root or root in path.parents or path in root.parents:
            raise ValueError("Outputs/cache must be outside immutable release resources")
    output.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    evidence = dict(status="running", complete=False, smoke=True, gpu=gpu.name,
                    memory_gib=gpu.total_memory / 2**30, driver_sha256=sha(__file__),
                    resource_manifest_sha256=sha(root / "RELEASE-MANIFEST.json"),
                    python=sys.version, phases=[], pauses={}, comparisons={},
                    scope="Deterministic saved-boundary pause/restart using existing trainer stop hooks",
                    limitations=["Not full training or benchmark reproduction",
                                 "Not a sudden process-loss, OS signal-forwarding, or Colab shutdown test",
                                 "Probe/gate smoke resumes validation after its only update; LoRA resumes another update"])
    record = output / "ACCEPTANCE.json"
    write(record, evidence)

    def phase(branch, name, stop=None, resume_stage=None):
        relative = {"probe": "probe/run", "original": "lora/original",
                    "diverse": "lora/diverse-gate", "evaluate": "lora/evaluation"}[name]
        label = f"{len(evidence['phases'])+1:02d}-{branch}-{name}-{stop or 'complete'}"
        log = output / f"{label}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name,
                   "--root", str(root), "--output", str(output / branch / relative), "--cache", str(cache)]
        if stop:
            command += ["--stop", stop]
        entry = dict(name=label, command=command, log=log.name, expected_exit=PAUSED if stop else 0)
        evidence["phases"].append(entry)
        write(record, evidence)
        entry["exit_code"] = run_child(command, log)
        write(record, evidence)
        if entry["exit_code"] != entry["expected_exit"]:
            raise RuntimeError(f"{label} failed: inspect {log}")
        if resume_stage:
            prefix = {"probe": "RESUME mlp256 seed 42:", "lora": "RESUME stage1-lora:",
                      "original-fast": "RESUME stage2-fast:", "diverse-fast": "RESUME FAST:"}[resume_stage]
            lines = [line for line in log.read_text().splitlines() if line.startswith(prefix)]
            if not lines or not any(marker in lines[0] for marker in ("updates 1", "updates=1,", "update 1,")):
                raise AssertionError(f"Missing one-update restoration evidence: {resume_stage}")
            entry["resume_evidence"] = lines[0]
            write(record, evidence)
        if stop:
            payload, files = checkpoint(output / branch / STAGES[stop], stop)
            observed = require_paused(payload)
            saved = output / "pause-evidence" / stop
            saved.mkdir(parents=True)
            observed["files"] = {}
            for path in files:
                shutil.copy2(path, saved / path.name)
                observed["files"][str((saved / path.name).relative_to(output))] = sha(saved / path.name)
            evidence["pauses"][stop] = observed
            write(record, evidence)

    try:
        for name in ("probe", "original", "diverse", "evaluate"):
            phase("uninterrupted", name)
        phase("resumed", "probe", "probe")
        phase("resumed", "probe", resume_stage="probe")
        phase("resumed", "original", "lora")
        phase("resumed", "original", "original-fast", resume_stage="lora")
        phase("resumed", "original", resume_stage="original-fast")
        phase("resumed", "diverse", "diverse-fast")
        phase("resumed", "diverse", resume_stage="diverse-fast")
        phase("resumed", "evaluate")
        for stage, relative in STAGES.items():
            folders = [output / branch / relative for branch in ("uninterrupted", "resumed")]
            left, right = [checkpoint(folder, stage)[0] for folder in folders]
            if read(folders[0] / "identity.json") != read(folders[1] / "identity.json"):
                raise AssertionError(f"Different run configuration: {stage}")
            for key in ("completed", "epoch", "updates", "cursor", "examples_seen", "best_epoch"):
                if left["state"].get(key) != right["state"].get(key):
                    raise AssertionError(f"Different final progress: {stage}/{key}")
            if not left["state"]["completed"] or not right["state"]["completed"]:
                raise AssertionError(f"Unfinished stage: {stage}")
            evidence["comparisons"][stage] = compare_tensors(left, right)
        from safetensors.torch import load_file
        for relative in EXPORTS:
            files = [output / branch / relative for branch in ("uninterrupted", "resumed")]
            result = compare_tensors(*(load_file(str(path)) for path in files))
            result["file_sha256"] = {branch: sha(path) for branch, path in zip(("uninterrupted", "resumed"), files)}
            evidence["comparisons"][relative] = result
        for branch in ("uninterrupted", "resumed"):
            for relative in ("probe/run/SUMMARY.json", "lora/evaluation/SUMMARY.json"):
                if read(output / branch / relative).get("complete") is not True:
                    raise AssertionError(f"Missing completed smoke evaluation: {branch}/{relative}")
        evidence.update(status="passed", complete=True)
    except BaseException as error:
        evidence.update(status="failed", error=type(error).__name__, detail=str(error))
        raise
    finally:
        write(record, evidence)
        print("Acceptance evidence:", record, flush=True)


def self_test():
    """Small CPU tests of false-positive protection; no models or downloads."""
    import unittest
    import torch
    class Checks(unittest.TestCase):
        def test_completed_and_zero_update_are_rejected(self):
            for state in (dict(completed=True, updates=1, examples_seen=32),
                          dict(completed=False, updates=0, examples_seen=0)):
                with self.assertRaises(AssertionError):
                    require_paused(dict(state=state))
            self.assertEqual(require_paused(dict(state=dict(completed=False, updates=1, examples_seen=32)))["updates"], 1)

        def test_tensor_comparison_includes_optimizer_and_rng(self):
            left = {"weights": torch.ones(2), "optimizer": {0: {"moment": torch.zeros(2)}}, "rng": [torch.tensor([1], dtype=torch.uint8)]}
            right = {"weights": torch.ones(2), "optimizer": {0: {"moment": torch.zeros(2)}}, "rng": [torch.tensor([1], dtype=torch.uint8)]}
            self.assertEqual(compare_tensors(left, right)["tensors"], 3)
            right["optimizer"][0]["moment"][0] = 1
            with self.assertRaises(AssertionError):
                compare_tensors(left, right)
            with self.assertRaises(AssertionError):
                compare_tensors({}, {})
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly start real GPU acceptance")
    parser.add_argument("--self-test", action="store_true", help="CPU fixtures only; no downloads")
    parser.add_argument("--root", type=Path, help="Verified resources; default downloads pinned private snapshots")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--worker", choices=("probe", "original", "diverse", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--stop", choices=tuple(STAGES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.run or args.worker) or not args.output or not args.cache:
        parser.error("Specify --run --output NEW_DIRECTORY --cache CACHE, or --self-test")
    if args.worker:
        if args.root is None:
            parser.error("Internal worker requires --root")
        return worker(args)
    acceptance(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
