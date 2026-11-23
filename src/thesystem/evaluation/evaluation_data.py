"""One fixed prompt mapping; reference labels never enter model inputs."""
import hashlib
import json
from pathlib import Path
import string

ROOT = Path(__import__("os").environ.get("THESYSTEM_RELEASE_ROOT", ".")).expanduser().resolve()
SUBSETS = ("vitaminc-dev", "paws", "squad2", "aegis2", "pubmedqa")
COUNTS = (599, 250, 299, 250, 250)
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SYSTEM_PROMPT = "You are a helpful assistant."


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_rows():
    return {name: read_rows(ROOT / "data/public" / (name + ".jsonl")) for name in SUBSETS}


def question_options(question):
    kind, criteria = question["type"], question["criteria"]
    if kind == "noul":
        keys = ["false", "true"]
    elif kind == "choice":
        keys = list(criteria)
    elif kind == "score":
        keys = [str(i) for i in range(len(criteria))]
        criteria = dict(zip(keys, criteria))
    else:
        raise ValueError("Unsupported decision type")
    if not 2 <= len(keys) <= 18:
        raise ValueError("This release supports 2–18 options")
    return keys, [f"{key}: {criteria[key]}" for key in keys]


def render_input(model_input, writing=False):
    if set(model_input) != {"state", "questions"} or set(model_input["questions"]) != {"decision"}:
        raise ValueError("Expected a single benchmark decision")
    question = model_input["questions"]["decision"]
    keys, options = question_options(question)
    context = model_input["state"]
    if not isinstance(context, str):
        context = json.dumps(context, ensure_ascii=False, indent=2)
    candidates = "\n".join(f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, options))
    request = ("Explain your answer in two or three sentences, using the context."
               if writing else "Select the best option. Reply with only its letter.")
    prompt = f"{question['instructions']}\n\nContext:\n{context}\n\nOptions:\n{candidates}\n\n{request}"
    return prompt, keys, options


def convert(record, subset, writing=False):
    prompt, keys, options = render_input(record["input"], writing)
    row = dict(id=("write-" if writing else "") + record["id"], prompt=prompt, options=options,
               option_keys=keys, group_id=subset + ":" + record["family"], family=subset,
               source=subset, input_format="public-benchmark-v1", mode_target=int(not writing),
               kind=record["input"]["questions"]["decision"]["type"], split="test")
    if not writing:
        target = record["reference"]["target"]
        key = str(target).lower() if isinstance(target, bool) else str(target)
        row["target_index"] = keys.index(key)
    return row


def load_suites(smoke=False):
    public = source_rows()
    decisions = {name: [convert(r, name) for r in public[name]] for name in SUBSETS}
    paired = [convert(r, name, writing=True) for name in SUBSETS for r in public[name]]
    # Five distinct context families per task, chosen by ID only, before any predictions.
    selected = []
    for name in SUBSETS:
        seen = set()
        for row in sorted((r for r in paired if r["family"] == name), key=lambda r: r["id"]):
            if row["group_id"] not in seen:
                seen.add(row["group_id"])
                selected.append(row)
            if len(seen) == 5:
                break
    historical = read_rows(ROOT / "data/fast/dataset/writing.jsonl")
    for row in historical:
        row["writing_suite"] = "historical"
    for row in selected:
        row["writing_suite"] = "external-paired"
    routing = read_rows(ROOT / "data/fast/dataset/new_test.jsonl")
    if len(historical) != 20 or len(selected) != 25 or len({r["id"] for r in historical + selected}) != 45:
        raise ValueError("Expected exactly 45 unique writing checks")
    if len(routing) != 4000 or sum(row["mode_target"] for row in routing) != 2000:
        raise ValueError("Expected 2,000 decisions and 2,000 writing requests in the internal routing set")
    writing = historical + selected
    if smoke:
        decisions = {name: rows[:2] for name, rows in decisions.items()}
        selected_ids = {"write-" + row["id"] for rows in decisions.values() for row in rows}
        paired = [row for row in paired if row["id"] in selected_ids]
        routing = [
            row for mode in (1, 0) for row in [r for r in routing if r["mode_target"] == mode][:2]]
        writing = [historical[0], selected[0]]
    return dict(decisions=decisions, paired_writing=paired, routing_historical=routing, writing=writing)


def tokenize(suites, tokenizer):
    rows = [r for values in suites["decisions"].values() for r in values]
    rows += suites["paired_writing"] + suites["routing_historical"] + suites["writing"]
    maximum, total = 0, 0
    for row in rows:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not 0 < len(ids) <= 2048:
            raise ValueError(f"{row['id']}: {len(ids)} tokens; no truncation or row removal allowed")
        if row.get("token_count") is not None and row["token_count"] != len(ids):
            raise ValueError(f"Tokenizer/template drift: {row['id']}")
        row["input_ids"] = ids
        row["input_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        maximum, total = max(maximum, len(ids)), total + 1
    return dict(rows=total, maximum_tokens=maximum, truncated=0)


def audit_sources():
    provenance = json.loads((ROOT / "provenance/datasets.json").read_text())
    result = []
    for name, count in zip(SUBSETS, COUNTS):
        path = ROOT / "data/public" / (name + ".jsonl")
        rows = read_rows(path)
        spec = provenance["subsets"][name]
        manifest = json.loads((ROOT / spec["publisher_manifest"]).read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["dataset_sha256"] or digest != spec["sha256"]:
            raise ValueError(f"Published dataset changed: {name}")
        if len(rows) != count or [r["id"] for r in rows] != manifest["ids"]:
            raise ValueError(f"Published selection changed: {name}")
        result.append(dict(dataset=name, rows=count, families=len({r['family'] for r in rows}), sha256=digest))
    return result
