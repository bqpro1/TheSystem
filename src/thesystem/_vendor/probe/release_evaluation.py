"""Resumable evaluation; wrong routes and capped writing remain visible."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .probe_model import HEAD_ARMS
from .probe_training import _digest, _open_folder, _read_checked, _write_checked, _write_json
from .progress import tqdm


def records_for(rows, folder, identity, evaluate, batch_size=4, shard_size=64):
    folder = Path(folder)
    key = _open_folder(folder, {"caller": identity, "rows": _digest(rows),
                                "batch_size": batch_size, "shard_size": shard_size})
    expected = [f"responses-{i:08d}.ptc" for i in range(0, len(rows), shard_size)]
    existing = sorted(p.name for p in folder.glob("responses-*.ptc"))
    if existing != expected[:len(existing)]:
        raise ValueError("Evaluation shards have a gap or unexpected file")
    if (folder / "complete.json").exists() and existing != expected:
        raise ValueError("A completed evaluation lost a shard; refusing to recompute it")
    all_records = []
    for start in tqdm(range(0, len(rows), shard_size), desc=folder.name, unit="shard"):
        subset = rows[start:start + shard_size]
        path = folder / f"responses-{start:08d}.ptc"
        if path.exists():
            saved = _read_checked(path)
        else:
            records = []
            for offset in range(0, len(subset), batch_size):
                records.extend(evaluate(subset[offset:offset + batch_size]))
            saved = {"key": key, "records": records}
            _write_checked(path, saved)
        if saved["key"] != key or [r["id"] for r in saved["records"]] != [r["id"] for r in subset]:
            raise ValueError("Saved evaluation response identity changed")
        all_records.extend(saved["records"])
    _write_json(folder / "complete.json", {"key": key, "records": len(all_records), "complete": True})
    return all_records


def decision_records(model, trained, rows, folder, identity):
    heads = {}
    for key, item in trained.items():
        head = deepcopy(model.heads[item["summary"]["arm"]])
        head.load_state_dict(item["weights"], strict=True)
        heads[key] = head.requires_grad_(False).eval()

    @torch.no_grad()
    def evaluate(batch):
        values = model.inspect_rows(batch)
        scores = {"original": values["original_scores"].cpu().tolist()}
        for key, head in heads.items():
            output = head(values["features"])
            if not torch.isfinite(output).all():
                raise FloatingPointError("Nonfinite classifier scores")
            scores[key] = output.cpu().tolist()
        probabilities = values["probabilities"].cpu().tolist()
        return [{"id": row["id"], "group_id": row["group_id"],
                 "family": row.get("family") or row["source"],
                 "input_format": row.get("input_format", "legacy"), "target": row["target_index"],
                 "probability": probabilities[i], "route": int(probabilities[i] >= .5),
                 "scores": {name: output[i][:len(row["options"])] for name, output in scores.items()}}
                for i, row in enumerate(batch)]

    return records_for(rows, folder, identity, evaluate)


def routing_records(model, rows, folder, identity):
    def evaluate(batch):
        probabilities = model.reference.gate_probabilities(batch)
        return [{"id": r["id"], "group_id": r["group_id"], "target": r["mode_target"],
                 "input_format": r.get("input_format", "legacy"), "probability": p,
                 "route": int(p >= .5)} for r, p in zip(batch, probabilities)]
    return records_for(rows, folder, identity, evaluate)


def interval(rows, values, repetitions=2000):
    grouped = defaultdict(list)
    for row, value in zip(rows, values):
        grouped[row["group_id"]].append(float(value))
    totals = np.array([sum(v) for v in grouped.values()])
    sizes = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(42)
    samples = np.empty(repetitions)
    for i in range(repetitions):
        indices = rng.integers(len(sizes), size=len(sizes))
        samples[i] = totals[indices].sum() / sizes[indices].sum()
    return np.quantile(samples, [.025, .975]).tolist()


def decision_summary(records):
    results = []
    original = [int(max(range(len(r["scores"]["original"])), key=r["scores"]["original"].__getitem__) == r["target"])
                for r in records]
    for key in records[0]["scores"]:
        correct = [int(max(range(len(r["scores"][key])), key=r["scores"][key].__getitem__) == r["target"])
                   for r in records]
        automatic = [c * r["route"] for c, r in zip(correct, records)]
        families = sorted({r["family"] for r in records})
        results.append({"model": key, "rows": len(records), "forced_accuracy": float(np.mean(correct)),
                        "automatic_accuracy": float(np.mean(automatic)) if key != "original" else None,
                        "wrong_routes": sum(1-r["route"] for r in records) if key != "original" else None,
                        "forced_accuracy_ci95": interval(records, correct),
                        "paired_gain_vs_original_ci95": interval(records, [a-b for a, b in zip(correct, original)]),
                        "by_family": {f: float(np.mean([c for c, r in zip(correct, records) if r["family"] == f]))
                                      for f in families}})
    return results


def routing_summary(records):
    decisions = [r for r in records if r["target"] == 1]
    writing = [r for r in records if r["target"] == 0]
    return {"rows": len(records), "decision_recall": float(np.mean([r["route"] for r in decisions])),
            "writing_error_rate": float(np.mean([r["route"] for r in writing])),
            "writing_error_ci95": interval(writing, [r["route"] for r in writing]),
            "by_format_decision_recall": {fmt: float(np.mean([r["route"] for r in decisions if r["input_format"] == fmt]))
                                          for fmt in sorted({r["input_format"] for r in decisions})}}


def writing_records(model, rows, folder, identity):
    def evaluate(batch):
        row = batch[0]
        attempts = []
        for cap in (512, 1024):
            native = model.respond(row, "original", mode="text", max_new_tokens=cap)
            forced = model.respond(row, "mlp256", mode="text", max_new_tokens=cap)
            automatic = {arm: model.respond(row, arm, max_new_tokens=cap) for arm in HEAD_ARMS}
            attempts.append({"cap": cap, "native": native, "forced": forced, "automatic": automatic})
            if not any(r["hit_limit"] for r in [native, forced, *automatic.values()]):
                break
        forced_match = forced["generated_ids"] == native["generated_ids"]
        matches = {arm: r["kind"] == "text" and r["generated_ids"] == native["generated_ids"]
                   for arm, r in automatic.items()}
        capped = any(r["hit_limit"] for r in [native, forced, *automatic.values()])
        no_answer_head_calls = all(
            all(r["trace"]["head_calls"][arm] == 0 for arm in HEAD_ARMS)
            for r in [forced, *automatic.values()] if r["kind"] == "text")
        return [{"id": row["id"], "group_id": row["group_id"], "attempts": attempts,
                 "forced_matches_native": forced_match, "automatic_matches_native": matches,
                 "remaining_truncation": capped, "writing_bypasses_answer_heads": no_answer_head_calls,
                 "passed": forced_match and all(matches.values()) and not capped and no_answer_head_calls}]
    return records_for(rows, folder, identity, evaluate, batch_size=1, shard_size=1)
