"""Run the frozen LoRA recipe, diverse FAST continuation, or release evaluation."""
import argparse
from dataclasses import asdict
from .data import load_original, load_probe, smoke_rows, tokenize_shared
from pathlib import Path
import sys
import os

from .common import ROOT, bind_run, heartbeat, load_data, output_path, read_json, sha, write_json

import torch
from safetensors.torch import load_file, save_file
from thesystem._vendor.lora.data import experiment_identity, load_datasets, tokenize_datasets
from thesystem._vendor.lora.runtime import check_gpu, enable_fast_kernels, load_model, preflight
from thesystem._vendor.lora.setup_runtime import verify_runtime
from thesystem._vendor.lora.training import Settings, cache_features, load_selected, seed_everything, train_fast, train_lora, weights_sha


def save_weights(folder, weights, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "weights.safetensors"
    if path.exists() and weights_sha(load_file(str(path))) != weights_sha(weights):
        raise ValueError(f"Refusing to overwrite different selected weights: {path}")
    if not path.exists():
        save_file({k: v.detach().cpu().contiguous() for k, v in weights.items()}, str(path))
    write_json(folder / "config.json", dict(metadata, tensor_sha256=weights_sha(weights),
                                          file_sha256=sha(path)))


def train_original(model, tokenizer, environment, run, smoke=False):
    settings = Settings(micro_batch_size=2, accumulation=8, lora_epochs=1 if smoke else 3, fast_epochs=1 if smoke else 10)
    datasets, manifest = load_original(ROOT)
    if smoke:
        datasets = {name: smoke_rows(rows, 32 if name.endswith("train") else 16) for name, rows in datasets.items()}
    identity = experiment_identity(model, manifest, Path(sys.modules["thesystem._vendor.lora.training"].__file__).parent, environment)
    identity = bind_run(run, dict(identity, task="train-original", smoke=smoke, settings=asdict(settings)))
    with heartbeat(run, "LoRA and original FAST training"):
        tokenize_datasets(datasets, tokenizer, manifest, run / "tokens")
        write_json(run / "preflight.json", preflight(model, datasets["decision_train"], settings))
        state = train_lora(model, datasets["decision_train"], datasets["decision_validation"],
                           run, settings, identity)
        if not state["completed"]:
            raise InterruptedError("LoRA paused and saved. Rerun with the same output folder.")
        model.stage("gate")
        features = {split: cache_features(model, datasets[f"fast_{split}"], run / "features" / split,
                                         identity, batch_size=4) for split in ("train", "validation")}
        state = train_fast(model, features["train"], features["validation"], run, settings, identity)
        if not state["completed"]:
            raise InterruptedError("FAST paused and saved. Rerun with the same output folder.")
        selected = load_selected(model, run, identity, settings, features["train"], features["validation"])
        common = dict(model=identity["model"], revision=identity["revision"], seed=42,
                      gate_after=16, threshold=.5, status="new reproduction; does not replace release")
        save_weights(run / "exports/lora", model.adapters.state_dict(),
                     dict(common, selected_epoch=selected["lora"]["selected_epoch"], rank=16, alpha=32))
        save_weights(run / "exports/original-fast", model.gate.state_dict(),
                     dict(common, selected_epoch=selected["fast"]["selected_epoch"]))
        write_json(run / "COMPLETE.json", dict(complete=True, selected=selected, settings=asdict(settings)))


def train_diverse(model, tokenizer, environment, run, initial_path, smoke=False):
    from thesystem._vendor.fast.gate_training import GateSettings, fit_gate
    datasets, manifest = load_data("fast")
    datasets = {name: datasets[name] for name in ("diverse_train", "old_validation", "new_validation")}
    if smoke:
        datasets = {name: smoke_rows(rows, 32) for name, rows in datasets.items()}
    unique = {}
    for rows in datasets.values():
        for row in rows:
            if row["id"] in unique and unique[row["id"]] != row:
                raise ValueError("Conflicting duplicate row")
            unique[row["id"]] = row
    rows = [unique[k] for k in sorted(unique)]
    initial = load_file(str(initial_path))
    config = read_json(initial_path.with_name("config.json"))
    if sha(initial_path) != config["file_sha256"] or weights_sha(initial) != config["tensor_sha256"]:
        raise ValueError("Initial gate identity mismatch")
    settings = GateSettings(seed=42, epochs=1 if smoke else 10)
    identity = bind_run(run, dict(task="train-diverse-fast", smoke=smoke, environment=environment,
                                 settings=asdict(settings), initial_gate=weights_sha(initial)))
    with heartbeat(run, "Diverse FAST features and training"):
        tokenize_datasets({"unique": rows}, tokenizer, manifest, run / "tokens")
        for part in datasets.values():
            for row in part:
                row["input_ids"] = unique[row["id"]]["input_ids"]
        model.stage("evaluation")
        features = cache_features(model, rows, run / "features", identity, batch_size=1, chunk_size=256)
        lookup = {row_id: i for i, row_id in enumerate(features["ids"])}
        indices = {key: [lookup[r["id"]] for r in values] for key, values in datasets.items()}
        result = fit_gate(features, indices["diverse_train"], datasets, indices, initial,
                          run / "training", settings, identity)
        if result is None:
            raise InterruptedError("FAST paused and saved. Rerun with the same output folder.")
        save_weights(run / "export", result["weights"], result["metadata"])
        write_json(run / "COMPLETE.json", dict(complete=True, summary=result["summary"]))


def evaluate(model, tokenizer, environment, run, lora_path, gate_path, smoke=False):
    from thesystem._vendor.lora.evaluation import (decision_summary, evaluate_decisions, evaluate_generation,
                            evaluate_routing, generation_summary, routing_summary)
    datasets, manifest = load_probe(ROOT)
    # The shared tokenizer contract is identical to the original answer-data wrapper.
    datasets = {k: v for k, v in datasets.items() if k not in ("decision_train", "decision_validation")}
    if smoke:
        datasets = {name: smoke_rows(rows, 2) for name, rows in datasets.items()}
    identity = bind_run(run, dict(task="evaluate-lora", smoke=smoke, environment=environment,
                                 lora_file=sha(lora_path), gate_file=sha(gate_path)))
    lora = load_file(str(lora_path))
    config = read_json(lora_path.with_name("config.json"))
    if sha(lora_path) != config["file_sha256"] or weights_sha(lora) != config["tensor_sha256"]:
        raise ValueError("LoRA identity mismatch")
    gate = load_file(str(gate_path))
    gate_config_path = gate_path.with_name("fast.json" if gate_path.name == "fast.safetensors" else "config.json")
    gate_config = read_json(gate_config_path)
    expected_file = gate_config.get("file_sha256", gate_config.get("safetensors_sha256"))
    expected_tensor = gate_config.get("tensor_sha256", gate_config.get("selected_gate_sha256"))
    if sha(gate_path) != expected_file or weights_sha(gate) != expected_tensor:
        raise ValueError("Gate identity mismatch")
    model.adapters.load_state_dict(lora, strict=True)
    model.gate.load_state_dict(gate, strict=True)
    model.stage("evaluation")
    checkpoints = {"lora_sha256": weights_sha(lora), "fast_sha256": weights_sha(gate)}
    with heartbeat(run, "LoRA release evaluation"):
        tokenize_shared(datasets, tokenizer, manifest, run / "tokens", tokenize_datasets)
        summary = {"model": "lora", "identity": identity, "checkpoints": checkpoints, "decisions": {}}
        for name in ("decision_test_historical", "decision_test_fresh"):
            records = evaluate_decisions(model, datasets[name], run / "results" / name, identity, checkpoints)
            _, report = decision_summary(records)
            selected = {arm: [r for r in records if r["arm"] == arm] for arm in ("lora", "fast")}
            n = len(datasets[name])
            summary["decisions"][name] = dict(rows=n,
                forced_correct=sum(r["end_to_end_correct"] for r in selected["lora"]),
                automatic_correct=sum(r["end_to_end_correct"] for r in selected["fast"]), report=report)
            write_json(run / "progress-summary.json", summary)
        summary["routing"] = {}
        for name in ("routing_test_old", "routing_test_updated"):
            records = evaluate_routing(model, datasets[name], run / "results" / name, identity, checkpoints)
            _, report = routing_summary(records)
            summary["routing"][name] = report
        for cap in (512, 1024):
            records = evaluate_generation(model, datasets["writing"], run / f"results/writing-{cap}",
                                          identity, checkpoints, max_new_tokens=cap, always_on_count=0)
            _, summary["writing"] = generation_summary(records)
            if not any(r["hit_token_limit"] for r in records):
                break
        summary["complete"] = True
        write_json(run / "SUMMARY.json", summary)
        print("LoRA evaluation saved:", run / "SUMMARY.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "train-fast", "evaluate"))
    parser.add_argument("--smoke", action="store_true", help="Tiny mechanics check, not a reproduction result")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lora-weights", type=Path, default=ROOT / "checkpoints/lora/weights.safetensors")
    parser.add_argument("--gate-weights", type=Path, default=ROOT / "checkpoints/fast/fast.safetensors")
    parser.add_argument("--initial-gate", type=Path, default=ROOT / "checkpoints/original-fast/weights.safetensors")
    args = parser.parse_args()
    from thesystem.artifacts import resolve_release_resources
    resolve_release_resources(ROOT)
    run = output_path(args.output)
    output_path(args.cache)
    verify_runtime()
    environment = check_gpu()
    enable_fast_kernels()
    seed_everything(42)
    model, tokenizer = load_model(token=os.environ.get("HF_TOKEN") or False, cache_dir=str(args.cache))
    if args.action == "train":
        train_original(model, tokenizer, environment, run, args.smoke)
    elif args.action == "train-fast":
        train_diverse(model, tokenizer, environment, run, args.initial_gate, args.smoke)
    else:
        evaluate(model, tokenizer, environment, run, args.lora_weights, args.gate_weights, args.smoke)


if __name__ == "__main__":
    main()
