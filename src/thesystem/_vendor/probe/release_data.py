"""Validate the frozen release-training inputs and exact tokenizer contract."""
from collections import Counter
import hashlib
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SYSTEM_PROMPT = "You are a helpful assistant."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eligible(row):
    return (row.get("mode_target") == 1 and row.get("answer_scoring_valid") is True
            and row.get("output_supported") is True and row.get("has_answer_codes") is True
            and row.get("request_scope") == "single")


def audit_splits(datasets):
    groups, prompts = {}, {}
    for split in ("train", "validation", "test"):
        rows = [r for name, values in datasets.items() if name.startswith("decision_")
                for r in values if r["split"] == split]
        groups[split] = {r["group_id"] for r in rows}
        prompts[split] = {" ".join(r["prompt"].split()).casefold() for r in rows}
        for row in rows:
            count, target = len(row["options"]), row["target_index"]
            if not 2 <= count <= 18 or isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < count:
                raise ValueError(f"Invalid answer target: {row['id']}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right] or prompts[left] & prompts[right]:
            raise ValueError(f"Cross-split overlap: {left}/{right}")
    for name, rows in datasets.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        if name.startswith("routing_") or name == "writing":
            if groups["train"] & {r["group_id"] for r in rows}:
                raise ValueError(f"Training group overlaps {name}")
            if prompts["train"] & {" ".join(r["prompt"].split()).casefold() for r in rows}:
                raise ValueError(f"Training prompt overlaps {name}")
    return {"group_overlap": False, "normalized_prompt_overlap": False,
            "limits": "Source grouping and exact normalized text; not exhaustive semantic deduplication."}


def load_inputs(root):
    root = Path(root)
    manifest = json.loads((root / "data-manifest.json").read_text())
    datasets = {}
    for name, record in manifest["files"].items():
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        rows = read_rows(path)
        if len(rows) != record["rows"]:
            raise ValueError(f"Dataset row-count mismatch: {name}")
        datasets[name] = rows
    audit_splits(datasets)
    return datasets, manifest


def tokenize_rows(rows, tokenizer):
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not 0 < len(ids) <= 2048:
            raise ValueError(f"{row['id']}: {len(ids)} tokens; no truncation")
        if row.get("token_count") is not None and row["token_count"] != len(ids):
            raise ValueError(f"Tokenizer/template drift for {row['id']}")
        row["input_ids"] = ids


def overview(datasets):
    return [{"split": name, "rows": len(rows), "groups": len({r["group_id"] for r in rows}),
             "sources": dict(Counter(r.get("source", "authored") for r in rows))}
            for name, rows in datasets.items()]
