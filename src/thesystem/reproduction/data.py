"""Read the exact release rows once; both answer models share canonical files."""
import json
from pathlib import Path
from .common import sha


def load_collection(root, name, keys=None):
    folder = Path(root) / "data" / name
    manifest = json.loads((folder / ("manifest.json" if name == "original" else "data-manifest.json")).read_text())
    datasets = {}
    for relative, expected in manifest["files"].items():
        if isinstance(expected, dict):
            relative, expected = expected.get("path", relative), expected["sha256"]
        key = Path(relative).stem.replace("-", "_")
        if keys is not None and key not in keys:
            continue
        path = (folder / relative).resolve()
        if folder.resolve() not in path.parents:
            raise ValueError("Dataset path escapes its collection")
        if sha(path) != expected:
            raise ValueError(f"Frozen dataset changed: {relative}")
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs: {relative}")
        datasets[key] = rows
    if keys is not None and set(datasets) != set(keys):
        raise ValueError(f"Missing required datasets: {set(keys) - set(datasets)}")
    return datasets, manifest


def load_original(root):
    from thesystem._vendor.lora.data import load_datasets
    return load_datasets(Path(root) / "data/original")


def load_probe(root):
    from thesystem._vendor.probe.release_data import audit_splits
    original, original_manifest = load_collection(root, "original", ("decision_train", "decision_validation", "decision_test"))
    fast, fast_manifest = load_collection(root, "fast", ("fresh_decisions", "old_test", "new_test", "writing"))
    datasets = {"decision_train": original["decision_train"],
                "decision_validation": original["decision_validation"],
                "decision_test_historical": original["decision_test"],
                "decision_test_fresh": fast["fresh_decisions"],
                "routing_test_old": fast["old_test"], "routing_test_updated": fast["new_test"],
                "writing": fast["writing"]}
    audit_splits(datasets)
    manifest = {"name": "selected-probe-256", "files": {
        **{"original/" + k: v for k, v in original_manifest["files"].items()},
        **{"fast/" + k: v for k, v in fast_manifest["files"].items()}}}
    return datasets, manifest


def smoke_rows(rows, limit=16):
    """Small deterministic mechanics check, retaining both routing labels."""
    if {row.get("mode_target") for row in rows} == {0, 1}:
        chosen = set()
        for mode in (0, 1):
            chosen.update(row["id"] for row in [r for r in rows if r["mode_target"] == mode][:max(1, limit // 2)])
        return [row for row in rows if row["id"] in chosen]
    return rows[:limit]


def tokenize_shared(datasets, tokenizer, manifest, cache, tokenize):
    """Tokenize repeated answer/routing rows once without dropping evaluations."""
    unique = {}
    for rows in datasets.values():
        for row in rows:
            previous = unique.get(row["id"])
            if previous is not None:
                clean = lambda value: {k: v for k, v in value.items() if k != "input_ids"}
                if clean(previous) != clean(row):
                    raise ValueError(f"Conflicting repeated input: {row['id']}")
            else:
                unique[row["id"]] = row
    tokenize({"unique": list(unique.values())}, tokenizer, manifest, cache)
    for rows in datasets.values():
        for row in rows:
            row["input_ids"] = unique[row["id"]]["input_ids"]
