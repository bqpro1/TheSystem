"""A small public interface around the selected, unchanged model implementations."""
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
import os
from pathlib import Path
import string
from threading import Lock
from typing import Literal

from .artifacts import Variant, resolve_model_resources

Mode = Literal["auto", "decision", "generation", "text"]


def _request(prompt: str, choices: Mapping[str, str] | None) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    if choices is None:
        return {"id": "request", "prompt": prompt, "options": []}
    if not isinstance(choices, Mapping) or not 2 <= len(choices) <= 18:
        raise ValueError("choices must map 2–18 codes, A through R, to answer texts")
    codes = string.ascii_uppercase[:len(choices)]
    if set(choices) != set(codes):
        raise ValueError("Choice codes must be consecutive letters starting at A")
    options = [choices[code] for code in codes]
    if any(not isinstance(text, str) or not text.strip() for text in options):
        raise ValueError("Every choice must be nonempty text")
    if len(set(options)) != len(options):
        raise ValueError("Each choice text must be different")
    return {"id": "request", "prompt": prompt, "options": options}


class TheSystem:
    """One loaded variant. Choices describe codes already present in the prompt."""

    def __init__(self, variant: Variant, model: object) -> None:
        if variant not in ("probe", "lora"):
            raise ValueError("variant must be 'probe' or 'lora'")
        self.variant = variant
        self._model = model
        self._request_lock = Lock()

    def respond(self, prompt: str, choices: Mapping[str, str] | None = None,
                mode: Mode = "auto", max_new_tokens: int = 128) -> dict:
        """Route automatically, or explicitly select decision/generation mode.

        The exact prompt is passed through the frozen chat template. Choice metadata
        is never appended to it. A decision route without choices remains an error.
        """
        if mode not in ("auto", "decision", "generation", "text"):
            raise ValueError("mode must be auto, decision or generation")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        row = _request(prompt, choices)
        if mode == "decision" and not row["options"]:
            raise ValueError("Explicit decision mode requires supplied choices")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent requests on one model are not supported")
        try:
            result = self._probe(row, mode, max_new_tokens) if self.variant == "probe" else self._lora(row, mode, max_new_tokens)
            return _response(result)
        finally:
            self._request_lock.release()

    def _probe(self, row: dict, mode: Mode, limit: int) -> dict:
        if mode != "decision":
            return self._model.respond(row, "mlp256", mode="auto" if mode == "auto" else "text", max_new_tokens=limit)
        result = self._model.decide(row, "mlp256", automatic=False)
        index = max(range(len(result["candidate_scores"])), key=result["candidate_scores"].__getitem__)
        return dict(result, selected_index=index, answer=row["options"][index], generated_ids=[], hit_limit=False)

    def _lora(self, row: dict, mode: Mode, limit: int) -> dict:
        if mode == "auto":
            result = self._model.route_request(row, max_new_tokens=limit)
        elif mode == "decision":
            result = self._model.score_requests([row], mode="on")[0]
            index = max(range(len(result["candidate_scores"])), key=result["candidate_scores"].__getitem__)
            result.update(selected_index=index, answer=row["options"][index], generated_ids=[])
        else:
            result = self._model.generate_request(row, mode="off", max_new_tokens=limit)
            result["answer"] = result["text"]
        tokens = result.get("generated_ids", [])
        eos = self._model.base.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        result["hit_limit"] = bool(len(tokens) >= limit and (not tokens or tokens[-1] not in eos))
        return result


def _response(result: dict) -> dict:
    output = dict(result)
    if result.get("routing_error") or result.get("kind") == "unsupported":
        output.update(mode="unsupported", answer=None, selected_code=None)
    elif result.get("route") == 1:
        output.update(mode="decision", selected_code=string.ascii_uppercase[result["selected_index"]])
    else:
        output.update(mode="generation", selected_code=None)
    return output


def load_model(variant: Variant = "probe", revision: str | None = None,
               root: str | Path | None = None, *,
               execution: Literal["reference", "inference"] = "reference") -> TheSystem:
    """Load the exact release on a supported BF16 CUDA GPU; never retrain/quantize."""
    if execution not in ("reference", "inference"):
        raise ValueError("execution must be reference or inference")
    resources = resolve_model_resources(variant, revision, root)
    import torch
    from safetensors.torch import load_file
    torch.manual_seed(42)
    if execution == "inference":
        from .demo_runtime import load_model as load_demo
        reference, _ = load_demo(variant, token=os.environ.get("HF_TOKEN") or False)
    else:
        runtime = import_module(f"thesystem._vendor.{variant}.runtime")
        setup = import_module(f"thesystem._vendor.{variant}.setup_runtime")
        setup.verify_runtime()
        runtime.check_gpu()
        runtime.enable_fast_kernels()
        reference, _ = runtime.load_model(token=os.environ.get("HF_TOKEN") or False)
    if variant == "probe":
        from ._vendor.probe.probe_model import ProbeModel
        from ._vendor.probe.release_artifacts import load_export
        model = ProbeModel(reference, max_choices=18)
        config = load_export(model, resources / "checkpoints/probe-256")
        if config["selected_epoch"] != 6 or config["seed"] != 42:
            raise ValueError("Expected the original matched-data probe, seed 42 / epoch 6")
        reference.assert_unchanged()
    else:
        reference.adapters.load_state_dict(load_file(str(resources / "checkpoints/lora/weights.safetensors")), strict=True)
        reference.gate.load_state_dict(load_file(str(resources / "checkpoints/fast/fast.safetensors")), strict=True)
        reference.stage("evaluation")
        model = reference
    model.requires_grad_(False).eval()
    return TheSystem(variant, model)
