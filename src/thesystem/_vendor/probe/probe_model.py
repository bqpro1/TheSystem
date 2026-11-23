"""Frozen-Qwen answer heads and a shared, explicitly traced request dispatcher."""
from contextlib import contextmanager
import hashlib
import json
import time

import torch
from torch import nn

from .model import normalized_features


HEAD_ARMS = ("mlp256",)
ARMS = ("original", *HEAD_ARMS)


class MLPHead(nn.Module):
    """A nonlinear readout of the frozen final representation."""

    def __init__(self, in_features, hidden_size, out_features, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(in_features, hidden_size, device=device, dtype=torch.float32)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_size, out_features, device=device, dtype=torch.float32)

    def reset_parameters(self):
        self.hidden.reset_parameters()
        self.output.reset_parameters()

    def forward(self, features):
        return self.output(self.activation(self.hidden(features)))


def head_spec(head):
    hidden = getattr(head, "hidden_size", None)
    return {"in_features": head.in_features, "out_features": head.out_features,
            "hidden_size": hidden, "class": type(head).__name__,
            "activation": type(head.activation).__name__ if hidden else None,
            "activation_approximate": getattr(getattr(head, "activation", None), "approximate", None),
            "bias": True, "dropout": 0., "dtype": "float32",
            "initialization": "torch-linear-default" if hidden else "zeros",
            "parameters": {name: list(value.shape) for name, value in head.named_parameters()}}


def initialize_head(head):
    if isinstance(head, nn.Linear):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    elif isinstance(head, MLPHead):
        head.reset_parameters()
    else:
        raise TypeError("Unsupported answer-head architecture")


class _PrefixComplete(Exception):
    """End this request before any later decoder layer executes."""


def _new_trace(layer_count):
    return {"layer_calls": {str(i + 1): 0 for i in range(layer_count)},
            "head_calls": {name: 0 for name in ("gate", *HEAD_ARMS, "original")}}


def _merge_traces(left, right):
    return {"layer_calls": {k: left["layer_calls"][k] + right["layer_calls"][k]
                            for k in left["layer_calls"]},
            "head_calls": {k: left["head_calls"][k] + right["head_calls"][k]
                           for k in left["head_calls"]}}


def _input_hash(encoded):
    values = encoded["input_ids"].detach().cpu().tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class ProbeModel(nn.Module):
    """Add answer readouts without changing the reference model or its FAST gate.

    All classifiers read the same final normalized prompt state. Concurrent
    calls are rejected; do not share its reference with another caller.
    """

    def __init__(self, reference, max_choices=18):
        super().__init__()
        if not isinstance(max_choices, int) or isinstance(max_choices, bool) or max_choices != 18:
            raise ValueError("max_choices must be 18")
        if reference._context is not None:
            raise RuntimeError("Cannot attach probes during a reference forward")
        self.reference, self.max_choices = reference, max_choices
        reference.stage("evaluation")
        reference.requires_grad_(False)
        for parameter in reference.parameters():
            parameter.grad = None
        self.heads = nn.ModuleDict({
            "mlp256": MLPHead(reference.hidden_size, 256, max_choices, reference.device)})
        for head in self.heads.values():
            initialize_head(head)
        self._busy = False

    def head_specs(self):
        return {arm: head_spec(head) for arm, head in self.heads.items()}

    @property
    def device(self):
        return self.reference.device

    def _trace(self):
        return _new_trace(len(self.reference.decoder.layers))

    @contextmanager
    def _timed(self, result):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            result["model_seconds"] = time.perf_counter() - started
            result["model_timing_scope"] = "Synchronized model/readout and tracing work; excludes input tokenization"

    @contextmanager
    def _session(self, trace):
        reference = self.reference
        if self._busy or reference._context is not None or reference._stage != "evaluation":
            raise RuntimeError("Run one request at a time with the reference in evaluation stage")
        self._busy = True
        handles = []

        def count(bucket, key):
            def before(module, inputs):
                bucket[key] += 1
            return before

        try:
            reference.eval()
            for index, layer in enumerate(reference.decoder.layers, 1):
                handles.append(layer.register_forward_pre_hook(count(trace["layer_calls"], str(index))))
            modules = {"gate": reference.gate, "original": reference.base.lm_head, **dict(self.heads.items())}
            for name, module in modules.items():
                handles.append(module.register_forward_pre_hook(count(trace["head_calls"], name)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._busy = False

    @torch.no_grad()
    def features(self, rows):
        """One frozen forward per batch; returned tensors can train the new heads."""
        encoded = self.reference.encode_rows(rows)
        batch = torch.arange(len(rows), device=self.device)
        last = encoded["attention_mask"].sum(1).long() - 1
        with self._session(self._trace()):
            hidden = self.reference.decoder(**encoded, use_cache=False).last_hidden_state
            features = hidden[batch, last].detach().float()
        if not torch.isfinite(features).all():
            raise FloatingPointError("Non-finite frozen features")
        return {"final": features}

    @torch.no_grad()
    def inspect_rows(self, rows):
        """One live forward supplies all classifier, routing and baseline scores."""
        reference = self.reference
        encoded = reference.encode_rows(rows)
        batch = torch.arange(len(rows), device=self.device)
        last = encoded["attention_mask"].sum(1).long() - 1
        captured = {}

        def read(module, inputs, hidden):
            captured["probabilities"] = reference.gate_logits(
                normalized_features(hidden[batch, last])).sigmoid()

        with self._session(self._trace()):
            handle = reference.decoder.layers[reference.gate_after - 1].register_forward_hook(read)
            try:
                hidden = reference.decoder(**encoded, use_cache=False).last_hidden_state[batch, last]
            finally:
                handle.remove()
            scores = reference.base.lm_head(hidden).float().index_select(-1, reference.choice_ids)
        if not torch.isfinite(hidden).all() or not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite live evaluation features or scores")
        return {"features": hidden.detach().float(), "probabilities": captured["probabilities"],
                "original_scores": scores}

    def masked_logits(self, features, arm, counts):
        """Differentiable head-only scoring; absent candidate positions get -inf."""
        if arm not in self.heads:
            raise ValueError(f"Head must be one of {HEAD_ARMS}")
        if features.ndim != 2 or features.shape[1] != self.reference.hidden_size:
            raise ValueError("Expected [batch, hidden_size] features")
        counts = torch.as_tensor(counts, device=self.device)
        if (counts.shape != (features.shape[0],) or counts.dtype == torch.bool
                or torch.any(counts != counts.long()) or torch.any(counts < 2)
                or torch.any(counts > self.max_choices)):
            raise ValueError(f"Each example needs 2 to {self.max_choices} choices")
        logits = self.heads[arm](features.detach().to(device=self.device, dtype=torch.float32))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite answer-head logits")
        valid = torch.arange(self.max_choices, device=self.device)[None, :] < counts[:, None]
        return logits.masked_fill(~valid, -torch.inf)

    def _validate_arm(self, arm):
        if arm != "original" and arm not in self.heads:
            raise ValueError(f"Unknown arm: {arm}")

    @torch.inference_mode()
    def decide(self, row, arm, automatic=False):
        """Score one request; automatic writing routes stop at the FAST layer.

        Original Qwen always scores the candidates here, without consulting FAST.
        ``respond(original, ...)`` provides its independent native text baseline.
        """
        self._validate_arm(arm)
        reference = self.reference
        encoded = reference.encode_rows([row])
        last = int(encoded["attention_mask"].sum().item()) - 1
        count = len(row.get("options", []))
        result = {"kind": "decision", "candidate_scores": [], "gate_probability": None,
                  "route": 1, "trace": self._trace(), "input_ids_hash": _input_hash(encoded)}

        def choose_route(module, inputs, hidden):
            early = normalized_features(hidden[:, last].detach())
            if arm != "original":
                probability = float(reference.gate_logits(early).sigmoid().item())
                if not 0 <= probability <= 1:
                    raise FloatingPointError("Invalid FAST probability")
                result["gate_probability"] = probability
                result["route"] = int(probability >= reference.threshold) if automatic else 1
            if not result["route"]:
                result["kind"] = "text_required"
                raise _PrefixComplete()
            limit = self.max_choices
            if not 2 <= count <= limit:
                result.update(kind="unsupported", reason=f"Decision requires 2 to {limit} supplied choices")
                raise _PrefixComplete()

        with self._timed(result), self._session(result["trace"]):
            handle = reference.decoder.layers[reference.gate_after - 1].register_forward_hook(choose_route)
            try:
                try:
                    hidden = reference.decoder(**encoded, use_cache=False).last_hidden_state[:, last]
                except _PrefixComplete:
                    return result
                if arm in self.heads:
                    scores = self.masked_logits(hidden, arm, [count])[0, :count]
                else:
                    logits = reference.base.lm_head(hidden).float()
                    scores = logits[0].index_select(0, reference.choice_ids[:count])
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Non-finite candidate scores")
                result["candidate_scores"] = scores.cpu().tolist()
            finally:
                handle.remove()
        return result

    @torch.inference_mode()
    def _native_text(self, row, max_new_tokens, seed):
        reference = self.reference
        encoded = reference.encode_rows([row])
        trace = self._trace()
        result = {"trace": trace, "input_ids_hash": _input_hash(encoded)}
        with self._timed(result), self._session(trace):
            devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                output = reference.base.generate(**encoded, do_sample=False,
                    max_new_tokens=max_new_tokens, use_cache=True, logits_to_keep=1,
                    pad_token_id=reference.tokenizer.pad_token_id)
        tokens = output[0, encoded["input_ids"].shape[1]:].cpu().tolist()
        eos = reference.base.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        stop = "eos" if tokens and tokens[-1] in eos else "length" if len(tokens) >= max_new_tokens else "other"
        return {**result, "kind": "text", "generated_ids": tokens,
                "text": reference.tokenizer.decode(tokens, skip_special_tokens=True),
                "hit_limit": stop == "length", "stop_reason": stop}

    @torch.inference_mode()
    def respond(self, row, arm, mode="auto", max_new_tokens=2048, seed=42):
        """Dispatch once, or force the unmodified generation path for preservation."""
        self._validate_arm(arm)
        if mode not in {"auto", "text"}:
            raise ValueError("Response mode must be auto or text")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        routed = ({"kind": "text_required", "candidate_scores": [], "gate_probability": None,
                   "route": 0, "trace": self._trace()} if mode == "text" or arm == "original"
                  else self.decide(row, arm, automatic=True))
        if routed["kind"] != "text_required":
            routed.update(generated_ids=[], text="", hit_limit=False, stop_reason="not_generated")
            if routed["kind"] == "decision":
                index = max(range(len(routed["candidate_scores"])), key=routed["candidate_scores"].__getitem__)
                routed.update(selected_index=index, answer=row["options"][index])
            return routed
        generated = self._native_text(row, max_new_tokens, seed)
        if "input_ids_hash" in routed and routed["input_ids_hash"] != generated["input_ids_hash"]:
            raise RuntimeError("Input tokens changed between routing and native generation")
        return {**routed, **generated, "routing_trace": routed["trace"],
                "generation_trace": generated["trace"],
                "trace": _merge_traces(routed["trace"], generated["trace"]),
                "model_seconds": routed.get("model_seconds", 0.) + generated["model_seconds"],
                "answer": generated["text"]}

    @torch.no_grad()
    def native_logits(self, row):
        """Full-prompt original logits for a fixed-token diagnostic; no gate call."""
        encoded = self.reference.encode_rows([row])
        with self._session(self._trace()):
            hidden = self.reference.decoder(**encoded, use_cache=False).last_hidden_state
            return self.reference.base.lm_head(hidden).detach().float().cpu()
