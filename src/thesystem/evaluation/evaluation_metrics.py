"""Task metrics and intervals that keep related contexts together."""
from collections import defaultdict
import math


def probabilities(scores):
    if not scores or any(not math.isfinite(value) for value in scores):
        raise ValueError("Missing or nonfinite candidate scores")
    values = [math.exp(value - max(scores)) for value in scores]
    total = sum(values)
    return [value / total for value in values]


def group_interval(rows, values, repetitions=2000):
    import numpy as np
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["group_id"]].append(value)
    if not groups:
        return None
    totals = np.array([sum(group) for group in groups.values()])
    sizes = np.array([len(group) for group in groups.values()])
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(repetitions):
        chosen = rng.integers(len(groups), size=len(groups))
        estimates.append(float(totals[chosen].sum() / sizes[chosen].sum()))
    return [float(x) for x in np.quantile(estimates, [.025, .975])]


def decision_metrics(rows, arm, automatic=False):
    correct, absolute_errors = [], []
    for row in rows:
        key = "automatic_scores" if automatic else "scores"
        scores = row[key][arm]
        probs = probabilities(scores)
        if len(scores) != len(row["option_keys"]):
            raise ValueError("Scores do not cover exactly the offered candidates")
        predicted = max(range(len(scores)), key=scores.__getitem__)
        correct.append(int(predicted == row["target"] and (not automatic or row["route"] == 1)))
        if row["kind"] == "score" and not automatic:
            expected = sum(i * p for i, p in enumerate(probs))
            absolute_errors.append(abs(expected - row["target"]))
    return dict(rows=len(rows), correct=sum(correct), accuracy=sum(correct)/len(rows),
                ci95=group_interval(rows, correct),
                wrong_routes=sum(1-r["route"] for r in rows) if automatic else None,
                expected_score_mae=sum(absolute_errors)/len(absolute_errors) if absolute_errors else None)


def routing_metrics(rows):
    result = dict(rows=len(rows), matrix=[[sum(r["target"] == target and r["route"] == predicted for r in rows)
                                          for predicted in (1, 0)] for target in (1, 0)],
                  matrix_order=["decision", "writing"])
    for target, name in ((1, "decision"), (0, "writing")):
        selected = [row for row in rows if row["target"] == target]
        if not selected:
            continue
        errors = [int(row["route"] != target) for row in selected]
        result[name] = dict(rows=len(selected), errors=sum(errors), error_rate=sum(errors)/len(selected),
                            ci95=group_interval(selected, errors))
    return result


def writing_metrics(rows):
    result = {}
    for suite in sorted({r["suite"] for r in rows}):
        selected = [r for r in rows if r["suite"] == suite]
        result[suite] = dict(rows=len(selected),
            forced_matches=sum(r["forced_matches_native"] for r in selected),
            automatic_matches=sum(r["automatic_matches_native"] for r in selected),
            wrong_routes=sum(not r["automatic_writing_route"] for r in selected),
            capped=sum(r["remaining_truncation"] for r in selected),
            decision_modules_bypassed=sum(r["decision_modules_bypassed"] for r in selected),
            passed=sum(r["passed"] for r in selected))
    return result
