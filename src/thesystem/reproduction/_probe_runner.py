"""Train or evaluate the selected one-hidden-layer probe; never install LoRA."""
import argparse
from dataclasses import asdict
from .data import load_original, load_probe, smoke_rows
from pathlib import Path
import sys
import os

from .common import ROOT, bind_run, heartbeat, output_path, sha, write_json

import torch
from thesystem._vendor.probe.probe_model import ProbeModel
from thesystem._vendor.probe.probe_training import ProbeSettings, _cpu_state, _state_digest, extract_features, train_probe
from thesystem._vendor.probe.release_artifacts import export_head, gate_digest, load_export, load_gate, GATE_TENSOR_SHA
from thesystem._vendor.probe.release_data import load_inputs, tokenize_rows
from thesystem._vendor.probe.release_evaluation import (decision_records, decision_summary, routing_records,
                                routing_summary, writing_records)
from thesystem._vendor.probe.runtime import check_gpu, enable_fast_kernels, load_model
from thesystem._vendor.probe.setup_runtime import verify_runtime


def evaluate(model, datasets, run, identity, training):
    trained = {"mlp256-seed42": {"summary": {"arm": "mlp256"},
                               "weights": _cpu_state(model.heads["mlp256"])}}
    evaluation_identity = dict(run=identity, answer_digest=_state_digest(model.heads["mlp256"]))
    summary = dict(model="probe-256", identity=identity, training=training, decisions={}, routing={})
    for name in ("decision_test_historical", "decision_test_fresh"):
        records = decision_records(model, trained, datasets[name], run / "results" / name, evaluation_identity)
        report = decision_summary(records)
        selected = next(r for r in report if r["model"] == "mlp256-seed42")
        n = len(records)
        summary["decisions"][name] = dict(rows=n, forced_correct=round(selected["forced_accuracy"] * n),
                                         automatic_correct=round(selected["automatic_accuracy"] * n), report=report)
        write_json(run / "progress-summary.json", summary)
    for name in ("routing_test_old", "routing_test_updated"):
        records = routing_records(model, datasets[name], run / "results" / name, evaluation_identity)
        summary["routing"][name] = routing_summary(records)
    writing = writing_records(model, datasets["writing"], run / "results/writing", evaluation_identity)
    summary["writing"] = dict(rows=len(writing), passed=sum(r["passed"] for r in writing),
                               remaining_truncation=sum(r["remaining_truncation"] for r in writing))
    model.reference.assert_unchanged()
    if gate_digest(model.reference.gate.state_dict()) != GATE_TENSOR_SHA:
        raise AssertionError("Shared FAST gate changed")
    summary["complete"] = True
    write_json(run / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--smoke", action="store_true", help="Tiny mechanics check, not a reproduction result")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "checkpoints/probe-256")
    args = parser.parse_args()
    from thesystem.artifacts import resolve_release_resources
    resolve_release_resources(ROOT)
    run = output_path(args.output)
    output_path(args.cache)
    verify_runtime()
    environment = check_gpu()
    enable_fast_kernels()
    torch.manual_seed(42)
    reference, tokenizer = load_model(token=os.environ.get("HF_TOKEN") or False, cache_dir=str(args.cache))
    load_gate(ROOT / "checkpoints/fast", reference)
    model = ProbeModel(reference, max_choices=18)
    datasets, manifest = load_probe(ROOT)
    if args.smoke:
        datasets = {name: smoke_rows(rows, 32 if name.endswith("train") else 2) for name, rows in datasets.items()}
    settings = ProbeSettings(seed=42, max_epochs=1 if args.smoke else 30)
    identity = dict(task=args.action + "-probe", smoke=args.smoke, environment=environment, data=manifest,
                    backbone_adapters=0, gate=GATE_TENSOR_SHA, settings=asdict(settings),
                    feature_extraction_batch_size=1)
    if args.action == "evaluate":
        config = load_export(model, args.probe_dir)
        identity["probe_file"] = sha(args.probe_dir / "heads.safetensors")
        training = {"selected_epoch": config["selected_epoch"], "seed": config["seed"]}
    identity = bind_run(run, identity)
    with heartbeat(run, args.action + " probe-256"):
        needed = datasets if args.action == "train" else {
            k: v for k, v in datasets.items() if k not in ("decision_train", "decision_validation")}
        for name, rows in needed.items():
            print("Tokenize:", name, len(rows), flush=True)
            tokenize_rows(rows, tokenizer)
        if args.action == "train":
            # Same frozen features and row order; recomputed without private Drive files.
            features = {split: extract_features(model, datasets[f"decision_{split}"], run / "features" / split,
                         dict(identity, split=split), batch_size=1, shard_size=64)
                        for split in ("train", "validation")}
            training = train_probe(model, "mlp256", features["train"], features["validation"],
                                   run / "training/seed-42/mlp256", identity, settings)
            if not training.get("completed"):
                raise InterruptedError("Probe paused and saved. Rerun the same output folder.")
            # Save selected weights before lengthy test generation.
            export_head(model, "mlp256", run / "export", identity, training,
                        {"evaluation_complete": False, "status": "new reproduction"})
        summary = evaluate(model, datasets, run, identity, training)
        if args.action == "train":
            export_head(model, "mlp256", run / "export", identity, training,
                        {"evaluation_complete": True, "writing": summary["writing"],
                         "routing": summary["routing"], "status": "new reproduction; not the frozen release"})
        print("Probe evaluation saved:", run / "SUMMARY.json", flush=True)


if __name__ == "__main__":
    main()
