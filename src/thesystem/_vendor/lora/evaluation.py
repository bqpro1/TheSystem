"""Resumable, paired tests. A raw FAST mistake is never repaired by a fallback.

The model protocol is small: score_requests, gate_probabilities, and
generate_request. No model weights or training data are changed here.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

DECISION_ARMS = ("original", "lora", "fast")
NAMES = {"original": "Original Qwen", "lora": "LoRA always on", "fast": "LoRA + FAST",
         "native": "Original Qwen", "off": "LoRA forced off", "on": "LoRA always on"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _progress(items, **kwargs):
    from tqdm.auto import tqdm
    return tqdm(items, **kwargs)


class Journal:
    """Append and flush each completed response; reject a different experiment.

    A disconnected runtime can leave only the last JSON line incomplete. Preserve
    that tail for inspection, recover the valid prefix, and redo that response.
    Earlier corruption or a changed checkpoint is an error, not a silent restart.
    """
    def __init__(self, output_dir, name, plan, identity, checkpoint_identity, settings):
        for key in ("lora_sha256", "fast_sha256"):
            digest = checkpoint_identity.get(key, "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Evaluation needs the loaded selected checkpoint's {key}")
        self.plan = plan
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{name}.jsonl"
        metadata_path = root / f"{name}.identity.json"
        identity_payload = dict(schema=1, run=identity, selected=checkpoint_identity,
                                settings=settings, plan=[(r["id"], a, _digest(r)) for r, a in plan],
                                evaluation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.key = _digest(identity_payload)
        if metadata_path.exists():
            if json.loads(metadata_path.read_text())["key"] != self.key:
                raise ValueError("Evaluation identity changed. Use a new evaluation folder; saved results are preserved.")
        elif self.path.exists():
            raise ValueError("Results exist without an identity file; inspect them before continuing.")
        else:
            _atomic_json(metadata_path, {"key": self.key, **identity_payload})
        self.records = []
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.splitlines(keepends=True)
            valid_bytes = 0
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if i != len(lines) - 1:
                        raise ValueError("An earlier evaluation record is corrupt; inspect the saved file.") from None
                    tail = root / f"{name}.interrupted-tail-{hashlib.sha256(line).hexdigest()[:12]}.txt"
                    tail.write_bytes(line)
                    with self.path.open("r+b") as handle:
                        handle.truncate(valid_bytes)
                    print(f"Recovered {len(self.records)} saved responses; incomplete last response will run again.", flush=True)
                    break
                if len(self.records) >= len(plan):
                    raise ValueError("Evaluation has more responses than its plan")
                row, arm = plan[len(self.records)]
                if (record.get("id"), record.get("arm"), record.get("request_sha256"),
                    record.get("identity"), record.get("plan_size")) != (
                        row["id"], arm, _digest(row), self.key, len(plan)):
                    raise ValueError("Saved results do not match the exact evaluation prefix")
                self.records.append(record)
                valid_bytes += len(line)
            if self.path.stat().st_size and not self.path.read_bytes().endswith(b"\n"):
                with self.path.open("ab") as handle:
                    handle.write(b"\n")
        print(f"{name}: {len(self.records)}/{len(plan)} responses already saved.", flush=True)

    def append(self, row, arm, output):
        expected_row, expected_arm = self.plan[len(self.records)]
        if expected_row != row or expected_arm != arm:
            raise ValueError("Evaluation attempted to save responses out of order")
        record = dict(id=row["id"], arm=arm, request_sha256=_digest(row), identity=self.key,
                      plan_size=len(self.plan), source=row.get("source", "authored"),
                      group_id=row.get("group_id", row["id"]),
                      input_format=row.get("input_format", "unknown"), **output)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)


def _validate_rows(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Evaluation needs nonempty rows with unique IDs")


def _complete(records):
    if not records or any(r["plan_size"] != len(records) for r in records):
        raise ValueError("Evaluation is incomplete. Rerun its cell to finish before summarizing.")


def _scores(output, option_count):
    scores = [float(v) for v in output["candidate_scores"]]
    if len(scores) != option_count or not all(math.isfinite(v) for v in scores):
        raise ValueError("Candidate scores are missing, non-finite, or the wrong size")
    return scores


def _probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid FAST probability")
    return value


def evaluate_decisions(model, rows, output_dir, identity, checkpoint_identity, stop_after=None):
    """2,000 held-out decisions; wrong FAST route counts as an end-to-end failure."""
    _validate_rows(rows)
    plan = [(row, arm) for row in rows for arm in DECISION_ARMS]
    journal = Journal(output_dir, "decisions", plan, identity, checkpoint_identity,
                      {"arms": DECISION_ARMS, "fallback": False})
    start = len(journal.records)
    for row, arm in _progress(plan[start:], initial=start, total=len(plan), desc="Decision comparison", unit="response"):
        mode = {"original": "off", "lora": "on", "fast": "auto"}[arm]
        options, target = row["options"], int(row["target_index"])
        if not 2 <= len(options) <= 26 or not 0 <= target < len(options):
            raise ValueError("Decision requires 2–26 options and one valid target")
        output = model.score_requests([row], mode=mode)[0]
        scores = _scores(output, len(options))
        chosen = max(range(len(scores)), key=scores.__getitem__)
        route = int(output["route"])
        if route not in (0, 1):
            raise ValueError("Invalid FAST route")
        answered = arm != "fast" or route == 1
        rating_error = abs(int(options[chosen]) - int(options[target])) if row["source"] == "helpsteer2" else None
        journal.append(row, arm, dict(target_index=target, chosen_index=chosen,
            candidate_scores=scores, choice_correct=chosen == target,
            answered_as_selection=answered, end_to_end_correct=answered and chosen == target,
            gate_probability=_probability(output["gate_probability"]), raw_fast=route,
            option_count=len(options), rating_absolute_error=rating_error,
            prompt=row["prompt"], options=options))
        if stop_after is not None and len(journal.records) - start >= stop_after:
            break
    return journal.records


def _paired_groups(records, arm, exclude_rating=False):
    indexed = {(r["id"], r["arm"]): r for r in records}
    pairs = []
    for baseline in records:
        if baseline["arm"] != "original" or (exclude_rating and baseline["source"] == "helpsteer2"):
            continue
        other = indexed[(baseline["id"], arm)]
        pairs.append((baseline["source"], baseline["group_id"],
                      int(other["end_to_end_correct"]) - int(baseline["end_to_end_correct"])))
    return pairs


def paired_interval(records, arm, repetitions=2000, seed=42, exclude_rating=False):
    """Paired cluster bootstrap. Shared content groups are resampled together.

    Each replicate computes accuracy within each source, then gives each source
    equal weight. Cross-source groups keep one shared resampling multiplicity.
    """
    import numpy as np
    pairs = _paired_groups(records, arm, exclude_rating)
    if not pairs or repetitions < 1:
        raise ValueError("Bootstrap requires paired observations and repetitions > 0")
    sources = sorted({r[0] for r in pairs})
    groups = sorted({r[1] for r in pairs})
    source_index, group_index = {s: i for i, s in enumerate(sources)}, {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(groups), len(sources)))
    changes = np.zeros_like(counts)
    for source, group, change in pairs:
        i, j = group_index[group], source_index[source]
        counts[i, j] += 1
        changes[i, j] += change
    rng = np.random.default_rng(seed)
    draws, attempts = [], 0
    while len(draws) < repetitions and attempts < repetitions * 20:
        weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        denominator = weights @ counts
        attempts += 1
        if (denominator == 0).any():
            continue
        draws.append(float(((weights @ changes) / denominator).mean() * 100))
    if len(draws) != repetitions:
        raise ValueError("Too few independent groups for a stable source-macro interval")
    return dict(gain_percentage_points=float((changes.sum(0) / counts.sum(0)).mean() * 100),
                interval_95_percentage_points=[float(x) for x in np.quantile(draws, [0.025, 0.975])],
                groups=len(groups), repetitions=repetitions, seed=seed,
                method="paired content-group bootstrap; equal weight per source",
                limitation="Conditional on this test set and one trained checkpoint; not training-seed uncertainty.")


def decision_summary(records, repetitions=2000):
    import pandas as pd
    _complete(records)
    by_arm = {a: [r for r in records if r["arm"] == a] for a in DECISION_ARMS}
    expected = {r["id"] for r in by_arm["original"]}
    if any(len(rows) != len(expected) or {r["id"] for r in rows} != expected for rows in by_arm.values()):
        raise ValueError("Decision comparison must contain exactly the same cases in every arm")
    table, per_source = [], []
    for arm, rows in by_arm.items():
        sources = sorted({r["source"] for r in rows})
        source_accuracy = {}
        for source in sources:
            subset = [r for r in rows if r["source"] == source]
            accuracy = sum(r["end_to_end_correct"] for r in subset) / len(subset)
            source_accuracy[source] = accuracy
            per_source.append({"Model": NAMES[arm], "Source": source, "Cases": len(subset),
                               "Answer accuracy %": round(100 * accuracy, 2)})
        nonrating = [v for k, v in source_accuracy.items() if k != "helpsteer2"]
        rating = [r for r in rows if r["source"] == "helpsteer2" and r["answered_as_selection"]]
        all_rating = [r for r in rows if r["source"] == "helpsteer2"]
        table.append({"Model": NAMES[arm], "Cases": len(rows),
                      "Answer accuracy %": round(100 * sum(r["end_to_end_correct"] for r in rows) / len(rows), 2),
                      "Equal-source accuracy %": round(100 * sum(source_accuracy.values()) / len(sources), 2),
                      "Excluding ratings %": round(100 * sum(nonrating) / len(nonrating), 2) if nonrating else None,
                      "Wrong route": sum(not r["answered_as_selection"] for r in rows),
                      "Rating MAE (answered only)": round(sum(r["rating_absolute_error"] for r in rating) / len(rating), 3) if rating else None,
                      "Rating coverage": f"{len(rating)}/{len(all_rating)}"})
    contrasts = {arm: paired_interval(records, arm, repetitions) for arm in ("lora", "fast")}
    nonrating_contrasts = ({arm: paired_interval(records, arm, repetitions, exclude_rating=True)
                           for arm in ("lora", "fast")} if any(r["source"] != "helpsteer2" for r in records) else {})
    report = dict(table=table, per_source=per_source, gains_vs_original=contrasts,
                  gains_excluding_ratings=nonrating_contrasts,
                  random_choice_accuracy=sum(1 / r["option_count"] for r in by_arm["original"]) / len(expected),
                  interpretation="Always-on isolates answer learning. FAST accuracy also penalizes routing misses. "
                                 "HelpSteer2 measures agreement with subjective ratings; conditional MAE must be read with coverage.")
    return pd.DataFrame(table), report


def evaluate_routing(model, rows, output_dir, identity, checkpoint_identity, suite="test", stop_after=None):
    _validate_rows(rows)
    if suite not in ("test", "fresh", "regressions"):
        raise ValueError("Keep test, fresh challenge, and known regressions separate")
    plan = [(row, "fast") for row in rows]
    journal = Journal(output_dir, f"routing-{suite}", plan, identity, checkpoint_identity,
                      {"suite": suite, "fallback": False, "threshold": model.threshold})
    start = len(journal.records)
    for row, arm in _progress(plan[start:], initial=start, total=len(plan), desc=f"FAST routing: {suite}", unit="request"):
        probability = _probability(model.gate_probabilities([row])[0])
        prediction = int(probability >= model.threshold)
        target = row["mode_target"]
        if target not in (0, 1):
            raise ValueError("Routing target must be 0 or 1")
        journal.append(row, arm, dict(suite=suite, target=target, predicted_fast=prediction,
            gate_probability=probability, correct=prediction == target, prompt=row["prompt"],
            options=row.get("options", []), category=row.get("category", "unknown"),
            fallback_applied=False))
        if stop_after is not None and len(journal.records) - start >= stop_after:
            break
    return journal.records


def routing_summary(records):
    import pandas as pd
    _complete(records)
    if len({r["suite"] for r in records}) != 1:
        raise ValueError("Do not pool known regressions with fresh test cases")
    counts = Counter((r["target"], r["predicted_fast"]) for r in records)
    n0, n1 = sum(r["target"] == 0 for r in records), sum(r["target"] == 1 for r in records)
    recalls = [counts[0, 0] / n0 if n0 else None, counts[1, 1] / n1 if n1 else None]
    table = [{"Suite": records[0]["suite"], "Requests": len(records),
              "Routing accuracy %": round(100 * sum(r["correct"] for r in records) / len(records), 2),
              "Balanced accuracy %": round(50 * sum(recalls), 2) if None not in recalls else None,
              "Text incorrectly sent to FAST": counts[0, 1],
              "Decisions incorrectly sent to text": counts[1, 0]}]
    slices = []
    for field in ("source", "input_format", "category"):
        for value in sorted({r[field] for r in records}):
            part = [r for r in records if r[field] == value]
            slices.append(dict(field=field, value=value, cases=len(part),
                               accuracy=sum(r["correct"] for r in part) / len(part)))
    errors = [dict(id=r["id"], expected="FAST" if r["target"] else "Text",
                   predicted="FAST" if r["predicted_fast"] else "Text", probability=r["gate_probability"],
                   source=r["source"], prompt=r["prompt"]) for r in records if not r["correct"]]
    return pd.DataFrame(table), dict(table=table, confusion={"text_to_text": counts[0, 0],
        "text_to_fast": counts[0, 1], "fast_to_text": counts[1, 0], "fast_to_fast": counts[1, 1]},
        slices=slices, errors=errors, fallback_applied=False,
        limitation="Fresh challenge has repeated authored rule families; known regressions were seen before. Report separately.")


def generation_sample(rows, count=12, seed=42):
    """Choose BEFORE inference, round-robin by request type; no output cherry-picking."""
    rng = random.Random(seed)
    categories = defaultdict(list)
    for row in rows:
        if row["mode_target"] == 0:
            categories[row.get("category", "unknown")].append(row)
    for subset in categories.values():
        rng.shuffle(subset)
    selected = []
    while len(selected) < count and any(categories.values()):
        for category in sorted(categories):
            if categories[category] and len(selected) < count:
                selected.append(categories[category].pop())
    if len(selected) < count:
        raise ValueError("Not enough generation examples for this sample")
    return selected


def evaluate_generation(model, rows, output_dir, identity, checkpoint_identity,
                        max_new_tokens=128, always_on_count=4, seed=42, stop_after=None):
    _validate_rows(rows)
    if any(row["mode_target"] != 0 for row in rows):
        raise ValueError("Writing tests require generation requests")
    if max_new_tokens < 1 or not 0 <= always_on_count <= len(rows):
        raise ValueError("Invalid generation test limits")
    plan = [(row, arm) for i, row in enumerate(rows)
            for arm in (("native", "off", "fast", "on") if i < always_on_count else ("native", "off", "fast"))]
    journal = Journal(output_dir, "generation", plan, identity, checkpoint_identity,
                      dict(max_new_tokens=max_new_tokens, always_on_count=always_on_count,
                           seed=seed, do_sample=False, fresh_cache_per_response=True, fallback=False))
    start = len(journal.records)
    for row, arm in _progress(plan[start:], initial=start, total=len(plan), desc="Text preservation", unit="response"):
        if arm == "fast":
            routing = model.score_requests([row], mode="auto")[0]
            probability, route = _probability(routing["gate_probability"]), int(routing["route"])
            if route not in (0, 1):
                raise ValueError("Invalid raw FAST route")
            if route:
                options = row.get("options", [])
                scores = _scores(routing, len(options))
                chosen = max(range(len(scores)), key=scores.__getitem__) if options else None
                output = dict(text=options[chosen] if options else "", generated_ids=[], trace={},
                              outcome="choice_instead_of_text" if options else "fast_without_options")
            else:
                output = model.generate_request(row, mode="off", max_new_tokens=max_new_tokens, seed=seed)
                output["outcome"] = "text"
        else:
            output = model.generate_request(row, mode=arm, max_new_tokens=max_new_tokens, seed=seed)
            probability, route = output.get("gate_probability"), output.get("route", 0)
            output["outcome"] = "text"
        token_ids = [int(t) for t in output["generated_ids"]]
        journal.append(row, arm, dict(text=output["text"], generated_ids=token_ids,
            outcome=output["outcome"], raw_fast=route, gate_probability=probability,
            hit_token_limit=len(token_ids) >= max_new_tokens,
            trace=output.get("trace", {}), prompt=row["prompt"],
            category=row.get("category", "unknown"), fallback_applied=False))
        if stop_after is not None and len(journal.records) - start >= stop_after:
            break
    return journal.records


def generation_summary(records):
    import pandas as pd
    _complete(records)
    indexed = {(r["id"], r["arm"]): r for r in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicated generation outputs")
    table = []
    for arm in ("native", "off", "fast", "on"):
        rows = [r for r in records if r["arm"] == arm]
        if not rows:
            continue
        same = sum(r["outcome"] == "text" and r["generated_ids"] == indexed[r["id"], "native"]["generated_ids"] for r in rows)
        table.append({"Model": NAMES[arm], "Requests": len(rows),
                      "Exactly matches original tokens": same,
                      "Wrong route": sum(r["outcome"] != "text" for r in rows),
                      "Hit output limit": sum(r["hit_token_limit"] for r in rows)})
    forced_off = [r for r in records if r["arm"] == "off"]
    mismatches = [r["id"] for r in forced_off if r["generated_ids"] != indexed[r["id"], "native"]["generated_ids"]]
    errors = [r for r in records if r["arm"] == "fast" and r["outcome"] != "text"]
    trace_failures = []
    for row in records:
        if row["outcome"] != "text":
            continue
        trace = row["trace"]
        expected_prefill = trace.get("adapter_count", 0) if row["arm"] == "on" else 0
        expected_decode = expected_prefill * max(len(row["generated_ids"]) - 1, 0)
        valid = (trace.get("prefill_projection_calls") == expected_prefill and
                 trace.get("decode_projection_calls") == expected_decode and
                 (row["arm"] != "on" or expected_prefill > 0))
        if not valid:
            trace_failures.append({"id": row["id"], "arm": row["arm"], "observed": trace,
                                   "expected_prefill_calls": expected_prefill,
                                   "expected_decode_calls": expected_decode})
    return pd.DataFrame(table), dict(table=table, forced_off_native_mismatches=mismatches,
        raw_fast_routing_errors=errors, adapter_trace_failures=trace_failures,
        interpretation="Matching tokens checks that the original path is preserved on these prompts. "
        "Different text does not establish better or worse writing. Inspect saved responses; "
        "this small sample is not a broad writing-quality benchmark. Always-on is a diagnostic ablation.",
        numerical_note="Every arm uses fresh native generation. Cached-versus-full logits are not assumed bit-identical in BF16.")
