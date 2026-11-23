"""Measure exact token preservation independently of automatic routing success."""


def check_writing(backend, row, caps=(512, 1024)):
    attempts = []
    for cap in caps:
        outputs = {mode: backend.generate(row, mode, cap) for mode in ("native", "text", "auto")}
        attempts.append(dict(cap=cap, outputs=outputs))
        if not any(value["hit_limit"] for value in outputs.values()):
            break
    native, forced, automatic = (outputs[mode] for mode in ("native", "text", "auto"))
    forced_match = forced["generated_ids"] == native["generated_ids"]
    automatic_match = (automatic["kind"] == "text"
                       and automatic["generated_ids"] == native["generated_ids"])
    capped = any(value["hit_limit"] for value in outputs.values())
    trace = forced.get("trace", {})
    if backend.arm == "probe":
        calls = trace.get("head_calls", {})
        bypassed = calls.get("mlp256") == 0
        if automatic["kind"] == "text":
            bypassed = bypassed and automatic.get("trace", {}).get("head_calls", {}).get("mlp256") == 0
    else:
        bypassed = (trace.get("prefill_projection_calls") == 0 and trace.get("decode_projection_calls") == 0)
    return dict(id=row["id"], group_id=row["group_id"], suite=row["writing_suite"],
                prompt=row["prompt"], input_sha256=row["input_sha256"], attempts=attempts,
                forced_matches_native=forced_match, automatic_matches_native=automatic_match,
                automatic_writing_route=automatic["kind"] == "text", remaining_truncation=capped,
                decision_modules_bypassed=bypassed,
                passed=forced_match and automatic_match and not capped and bypassed)
