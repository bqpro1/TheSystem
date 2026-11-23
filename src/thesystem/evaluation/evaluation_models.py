"""Inference adapters around unchanged release implementations, one process per arm."""
import sys
import os
from pathlib import Path

from .evaluation_io import read_json, sha
from .runtime import check_gpu

ROOT = Path(__import__("os").environ.get("THESYSTEM_RELEASE_ROOT", ".")).expanduser().resolve()


def load_backend(arm, cache, state):
    runtime_root = ROOT
    from importlib import import_module
    verify_runtime = import_module(f"thesystem._vendor.{arm}.setup_runtime").verify_runtime
    runtime = import_module(f"thesystem._vendor.{arm}.runtime")
    enable_fast_kernels, load_model = runtime.enable_fast_kernels, runtime.load_model
    import torch
    from safetensors.torch import load_file
    verify_runtime()
    environment = check_gpu()
    state.update(stage="Check GPU kernels", completed=0, planned=1)
    enable_fast_kernels()
    torch.manual_seed(42)
    state.update(stage="Load pinned 9B backbone", completed=0, planned=1)
    reference, tokenizer = load_model(token=os.environ.get("HF_TOKEN") or False, cache_dir=str(cache))
    if arm == "lora":
        folder = runtime_root / "checkpoints/lora"
        config = read_json(folder / "config.json")
        if sha(folder / "weights.safetensors") != config["file_sha256"]:
            raise ValueError("Selected LoRA changed")
        reference.adapters.load_state_dict(load_file(str(folder / "weights.safetensors")), strict=True)
        gate_folder = runtime_root / "checkpoints/fast"
        gate_config = read_json(gate_folder / "fast.json")
        if sha(gate_folder / "fast.safetensors") != gate_config["safetensors_sha256"]:
            raise ValueError("Selected FAST changed")
        reference.gate.load_state_dict(load_file(str(gate_folder / "fast.safetensors")), strict=True)
        reference.stage("evaluation")
        model = reference
    else:
        from thesystem._vendor.probe.probe_model import ProbeModel
        from thesystem._vendor.probe.release_artifacts import load_export
        model = ProbeModel(reference, max_choices=18)
        config = load_export(model, runtime_root / "checkpoints/probe-256")
        reference.assert_unchanged()
        if config["selected_epoch"] != 6 or config["seed"] != 42:
            raise ValueError("Wrong probe selection")
    model.requires_grad_(False).eval()
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("Evaluation requires frozen parameters")
    print(f"Loaded selected {arm}; all parameters frozen", flush=True)
    return Backend(arm, model, tokenizer), environment


class Backend:
    def __init__(self, arm, model, tokenizer):
        self.arm, self.model, self.tokenizer = arm, model, tokenizer

    def decisions(self, rows):
        import torch
        with torch.inference_mode():
            if self.arm == "lora":
                forced = self.model.score_requests(rows, mode="on")
                automatic = self.model.score_requests(rows, mode="auto")
                return [dict(scores={"lora": a["candidate_scores"]},
                             automatic_scores={"lora": b["candidate_scores"]},
                             probability=b["gate_probability"], route=b["route"])
                        for a, b in zip(forced, automatic)]
            values = self.model.inspect_rows(rows)
            counts = [len(r["options"]) for r in rows]
            head = self.model.masked_logits(values["features"], "mlp256", counts)
            native = values["original_scores"].cpu().tolist()
            head = head.cpu().tolist()
            probabilities = values["probabilities"].cpu().tolist()
            return [dict(scores={"qwen": a[:n], "probe256": b[:n]},
                         automatic_scores={"probe256": b[:n]}, probability=p, route=int(p >= .5))
                    for a, b, p, n in zip(native, head, probabilities, counts)]

    def routing(self, rows):
        reference = self.model if self.arm == "lora" else self.model.reference
        return reference.gate_probabilities(rows)

    def generate(self, row, mode, cap):
        if self.arm == "probe":
            arm = "original" if mode == "native" else "mlp256"
            result = self.model.respond(row, arm, mode="auto" if mode == "auto" else "text",
                                        max_new_tokens=cap, seed=42)
            return result
        if mode == "auto":
            result = self.model.route_request(row, max_new_tokens=cap)
            result["kind"] = "text" if result["route"] == 0 else "decision"
            result["text"] = result.get("answer", "") if result["kind"] == "text" else ""
        else:
            result = self.model.generate_request(row, mode="native" if mode == "native" else "off",
                                                 max_new_tokens=cap, seed=42)
            result["kind"] = "text"
        tokens = result.get("generated_ids", [])
        eos = self.model.base.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        result["hit_limit"] = bool(len(tokens) >= cap and (not tokens or tokens[-1] not in eos))
        return result

    def dispatch_checks(self, rows):
        """Check actual single-request automatic dispatch against the score path."""
        results = []
        for row in rows:
            evaluated = self.decisions([row])[0]
            if self.arm == "lora":
                live = self.model.score_requests([row], mode="auto")[0]
            else:
                live = self.model.decide(row, "mlp256", automatic=True)
            if live["route"] != evaluated["route"]:
                raise AssertionError("Automatic dispatch differs from recorded routing")
            if live["route"]:
                key = "lora" if self.arm == "lora" else "probe256"
                scores = evaluated["automatic_scores"][key]
                if max(range(len(scores)), key=scores.__getitem__) != max(
                        range(len(live["candidate_scores"])), key=live["candidate_scores"].__getitem__):
                    raise AssertionError("Automatic dispatch selects a different answer")
            results.append(dict(id=row["id"], route=live["route"], passed=True))
        return results
