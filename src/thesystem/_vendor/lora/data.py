"""Read the frozen review bundle; never rebuild or silently shorten examples."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SYSTEM_PROMPT = "You are a helpful assistant."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_datasets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["tokenizer"]["revision"] != MODEL_REVISION:
        raise ValueError("Dataset tokenizer revision differs from this experiment.")
    datasets = {}
    for name, expected in tqdm(manifest["files"].items(), desc="Verify and read datasets", unit="file"):
        path = directory / name
        if sha256(path) != expected:
            raise ValueError(f"Dataset checksum failed: {name}. Restore the supplied bundle.")
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        for row in rows:
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Missing prompt: {row['id']}")
            if path.stem.startswith("decision-"):
                if not 2 <= len(row["options"]) <= 26 or not 0 <= row["target_index"] < len(row["options"]):
                    raise ValueError(f"Invalid answer choices: {row['id']}")
            elif row.get("mode_target") not in (0, 1):
                raise ValueError(f"Invalid FAST label: {row['id']}")
        datasets[path.stem.replace("-", "_")] = rows
    for split, sizes in manifest["counts"].items():
        for stage, expected in sizes.items():
            if len(datasets[f"{stage}_{split}"]) != expected:
                raise ValueError(f"Wrong number of {stage} {split} examples")
    print(f"Verified all {len(datasets)} dataset files. No training data was regenerated.")
    return datasets, manifest


def overview(datasets):
    counts = Counter((row["source"], len(row["options"])) for row in datasets["decision_train"])
    return pd.DataFrame([{"Source": source, "Choices": choices, "Training examples": n}
                         for (source, choices), n in sorted(counts.items())])


def tokenize_datasets(datasets, tokenizer, manifest, cache_dir):
    """Cache only token IDs; prompts/gold labels stay in the reviewed data."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = json_hash({"files": manifest["files"], "revision": MODEL_REVISION,
                          "template": tokenizer.chat_template, "system": SYSTEM_PROMPT,
                          "thinking": False, "limit": 2048})
    path, commit = cache_dir / "token-ids.jsonl", cache_dir / "token-ids.commit.json"
    if path.exists() and commit.exists():
        saved = json.loads(commit.read_text())
        if saved.get("identity") == identity and saved.get("sha256") == sha256(path):
            with path.open() as handle:
                cached = {r["id"]: r["input_ids"] for r in map(json.loads, handle)}
            expected_ids = {r["id"] for rows in datasets.values() for r in rows}
            if set(cached) != expected_ids:
                raise ValueError("Token cache is incomplete; remove only this token cache and rerun.")
            for rows in datasets.values():
                for row in rows:
                    row["input_ids"] = cached[row["id"]]
            print(f"Reused token IDs for {len(cached):,} requests.")
            return
    temporary = path.with_suffix(".tmp")
    total = sum(len(rows) for rows in datasets.values())
    seen = set()
    with temporary.open("w") as handle, tqdm(total=total, desc="Prepare token IDs", unit="request") as bar:
        for rows in datasets.values():
            for row in rows:
                ids = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if not 0 < len(ids) <= 2048:
                    raise ValueError(f"Prompt {row['id']} has {len(ids)} tokens; no truncation is allowed.")
                if row.get("token_count") is not None and row["token_count"] != len(ids):
                    raise ValueError(f"Chat template drift for {row['id']}: audited {row['token_count']}, now {len(ids)}")
                row["input_ids"] = ids
                if row["id"] in seen:
                    raise ValueError(f"Duplicate ID across files: {row['id']}")
                seen.add(row["id"])
                handle.write(json.dumps({"id": row["id"], "input_ids": ids}) + "\n")
                bar.update(1)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    write_json(commit, {"identity": identity, "sha256": sha256(path), "rows": total})


def generation_examples(rows, count=12):
    """Fixed held-out examples, spread across writing categories; no cherry-picking."""
    groups = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["mode_target"] == 0:
            groups.setdefault(row.get("category", "other"), []).append(row)
    result, offset = [], 0
    while len(result) < count:
        batch = [rows[offset] for _, rows in sorted(groups.items()) if len(rows) > offset]
        if not batch:
            break
        result.extend(batch)
        offset += 1
    return result[:count]


def experiment_identity(model, manifest, source_dir, environment):
    """Hash small experiment inputs, never copy 18 GB of model weights to the CPU."""
    files = ("model.py", "training.py", "data.py", "runtime.py", "requirements.txt")
    return {"schema": 1, "model": MODEL_ID, "revision": MODEL_REVISION,
            "architecture": model.description(), "data_version": manifest["version"],
            "data_files": manifest["files"], "system_prompt": SYSTEM_PROMPT,
            "enable_thinking": False, "max_length": 2048,
            "source": {name: sha256(Path(source_dir) / name) for name in files},
            "runtime": environment}
