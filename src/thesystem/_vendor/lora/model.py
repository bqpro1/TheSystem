"""Frozen Qwen3.5 with LoRA after an internal FAST gate (text-only experiment)."""
from __future__ import annotations

from contextlib import contextmanager
import math
import string

import torch
from torch import nn

SYSTEM_PROMPT = "You are a helpful assistant."


def normalized_features(hidden):
    hidden = hidden.float()
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)


class LoRA(nn.Module):
    def __init__(self, input_width, output_width, rank, alpha, device):
        super().__init__()
        self.down = nn.Linear(input_width, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, output_width, bias=False, device=device, dtype=torch.float32)
        self.scale = alpha / rank
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(self.down(hidden.float())) * self.scale


class SwitchableLinear(nn.Module):
    """Retain original weights; route=0 executes the original map exactly."""
    def __init__(self, original, addition):
        super().__init__()
        self.original = original
        self.addition = addition
        self.route = 0
        self.trace = None

    @property
    def weight(self):
        return self.original.weight

    def forward(self, hidden):
        original = self.original(hidden)
        if isinstance(self.route, int) and self.route == 0:
            return original
        delta = self.addition(hidden).to(original.dtype)
        if self.trace is not None:
            key = "decode_projection_calls" if hidden.shape[1] == 1 else "prefill_projection_calls"
            self.trace[key] += 1
        if isinstance(self.route, int):
            return original + delta
        active = self.route.to(device=hidden.device, dtype=torch.bool)[:, None, None]
        return torch.where(active, original + delta, original)


class PrefixReady(Exception):
    """Stop a feature-only pass immediately after the frozen early layers."""


class InternalFAST(nn.Module):
    def __init__(self, base, tokenizer, gate_after=16, lora_rank=16, lora_alpha=32.0,
                 max_length=2048, threshold=0.5):
        super().__init__()
        if getattr(base, "_thesystem_attached", False):
            raise ValueError("This Qwen instance already has a FAST wrapper.")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("FAST threshold must be between zero and one.")
        if lora_rank < 1 or not math.isfinite(lora_alpha) or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        self.base, self.tokenizer = base, tokenizer
        self.gate_after, self.max_length, self.threshold = gate_after, max_length, threshold
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self.hidden_size = self.decoder.config.hidden_size
        if not 0 < gate_after < len(self.decoder.layers):
            raise ValueError("FAST must leave at least one later layer to adapt.")
        if self.decoder.config.model_type != "qwen3_5_text":
            raise ValueError("Expected a Qwen3.5 text decoder, including within its multimodal wrapper.")
        self.base.requires_grad_(False).eval()
        # Keep references to original parameters so the freeze check excludes new LoRA matrices.
        self._original_parameters = tuple(self.base.parameters())
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in string.ascii_uppercase]
        if any(len(tokens) != 1 for tokens in ids) or len({tokens[0] for tokens in ids}) != 26:
            raise ValueError("Candidate labels A-Z must be distinct single tokens.")
        self.register_buffer("choice_ids", torch.tensor([tokens[0] for tokens in ids], device=self.device))
        self.gate = nn.Linear(self.hidden_size, 1, device=self.device, dtype=torch.float32)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.adapters = nn.ModuleDict()
        self._projections = []
        self.adapter_targets = []
        for index, layer in enumerate(self.decoder.layers):
            if index < gate_after:
                continue
            paths = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            if self.decoder.config.layer_types[index] == "full_attention":
                paths += [f"self_attn.{name}_proj" for name in ("q", "k", "v", "o")]
            else:
                paths += [f"linear_attn.{name}" for name in
                          ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")]
            for path in paths:
                parent_path, attr = path.rsplit(".", 1)
                parent = layer.get_submodule(parent_path)
                linear = getattr(parent, attr)
                if not isinstance(linear, nn.Linear):
                    raise ValueError(f"Expected a linear projection: layer {index + 1}, {path}")
                key = f"layer_{index + 1}__{path.replace('.', '__')}"
                addition = LoRA(linear.in_features, linear.out_features, lora_rank, lora_alpha,
                                linear.weight.device)
                self.adapters[key] = addition
                wrapped = SwitchableLinear(linear, addition)
                setattr(parent, attr, wrapped)
                self._projections.append(wrapped)
                self.adapter_targets.append(f"layer_{index + 1}.{path}")
        self._context = None
        self._gate_handle = self.decoder.layers[gate_after - 1].register_forward_hook(self._read_gate)
        self.base._thesystem_attached = True
        self._stage = "evaluation"
        self.stage("evaluation")

    @property
    def decoder(self):
        model = self.base.model
        return model.language_model if hasattr(model, "language_model") else model

    @property
    def device(self):
        return self.base.get_input_embeddings().weight.device

    def train(self, mode=True):
        super().train(mode)
        # LoRA has no dropout. Training mode is needed only for activation checkpointing.
        self.base.train(mode and self._stage == "lora")
        return self

    def stage(self, name):
        if name not in {"lora", "gate", "evaluation"}:
            raise ValueError(f"Unknown stage: {name}")
        if self._context is not None:
            raise RuntimeError("Cannot change training stage during a forward pass.")
        self._stage = name
        self.base.requires_grad_(False)
        self.adapters.requires_grad_(name == "lora")
        self.gate.requires_grad_(name == "gate")
        if name == "lora":
            self.base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Early layers are frozen and produce no gradient graph; do not recompute them.
            for layer in self.decoder.layers[:self.gate_after]:
                layer.gradient_checkpointing = False
        else:
            self.base.gradient_checkpointing_disable()
        self.train(name != "evaluation")
        self._set_route(1 if name == "lora" else 0)
        if any(parameter.requires_grad for parameter in self._original_parameters):
            raise RuntimeError("An original Qwen parameter was unexpectedly unfrozen.")

    def _set_route(self, route):
        for projection in self._projections:
            projection.route = route

    def encode_rows(self, rows):
        if not rows:
            raise ValueError("Cannot encode an empty batch.")
        all_ids = []
        for row in rows:
            if "input_ids" in row:
                ids = list(row["input_ids"])
            else:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not 0 < len(ids) <= self.max_length:
                raise ValueError(f"Request {row.get('id', '?')} has {len(ids)} tokens; limit={self.max_length}. No truncation is allowed.")
            all_ids.append(ids)
        width = max(map(len, all_ids))
        ids = [tokens + [self.tokenizer.pad_token_id] * (width - len(tokens)) for tokens in all_ids]
        masks = [[1] * len(tokens) + [0] * (width - len(tokens)) for tokens in all_ids]
        return {"input_ids": torch.tensor(ids, device=self.device),
                "attention_mask": torch.tensor(masks, device=self.device)}

    def gate_logits(self, features):
        return self.gate(features.to(device=self.device, dtype=torch.float32)).squeeze(-1)

    def _read_gate(self, module, args, hidden):
        context = self._context
        if context is None:
            return hidden
        features = normalized_features(hidden[context["batch"], context["last"]].detach())
        context["features"] = features
        if context["prefix_only"]:
            raise PrefixReady()
        probability = self.gate_logits(features).sigmoid()
        context["probability"] = probability
        route = (probability >= self.threshold).to(torch.int64)
        if context["mode"] != "auto":
            route = torch.full_like(route, int(context["mode"] == "on"))
        context["route"] = route
        # Candidate availability never masks the gate. Routing errors remain visible.
        self._set_route(route if context["mode"] == "auto" else int(context["mode"] == "on"))
        return hidden

    def _forward_rows(self, rows, mode, prefix_only=False):
        if mode not in {"on", "off", "auto"}:
            raise ValueError("Mode must be on, off, or auto.")
        if self._context is not None:
            raise RuntimeError("Concurrent forwards on one FAST instance are not supported.")
        if self._stage == "lora" and torch.is_grad_enabled() and not prefix_only and mode != "on":
            raise ValueError("LoRA training requires forced on routing for stable checkpoint recomputation.")
        encoded = self.encode_rows(rows)
        context = {"batch": torch.arange(len(rows), device=self.device),
                   "last": encoded["attention_mask"].sum(1).long() - 1,
                   "prefix_only": prefix_only, "mode": mode}
        self._context = context
        self._set_route(0)
        try:
            try:
                hidden = self.decoder(**encoded, use_cache=False).last_hidden_state
            except PrefixReady:
                if not prefix_only:
                    raise
                return context["features"]
            last = hidden[context["batch"], context["last"]]
            # Only project the final prompt position: avoid a [batch, length, vocabulary] tensor.
            logits = self.base.lm_head(last).float()
            return logits, context["probability"], context["route"]
        finally:
            self._context = None
            # Stage 1's backward pass recomputes later layers with the same forced-on adapters.
            self._set_route(1 if self._stage == "lora" else 0)

    def decision_logits(self, rows, route="on"):
        counts = [len(row.get("options", [])) for row in rows]
        if any(not 2 <= count <= 26 for count in counts):
            raise ValueError("Decision training requires 2 to 26 supplied options per request.")
        logits, _, _ = self._forward_rows(rows, route)
        selected = logits.index_select(-1, self.choice_ids)
        valid = torch.arange(26, device=self.device)[None, :] < torch.tensor(counts, device=self.device)[:, None]
        return selected.masked_fill(~valid, -torch.inf)

    @torch.no_grad()
    def gate_features(self, rows):
        return self._forward_rows(rows, "off", prefix_only=True)

    @torch.inference_mode()
    def gate_probabilities(self, rows):
        return self.gate_logits(self.gate_features(rows)).sigmoid().cpu().tolist()

    @torch.inference_mode()
    def score_requests(self, rows, mode="auto"):
        logits, probabilities, routes = self._forward_rows(rows, mode)
        scores = logits.index_select(-1, self.choice_ids).cpu().tolist()
        results = []
        for row, score, probability, route in zip(rows, scores, probabilities.tolist(), routes.tolist()):
            count = len(row.get("options", []))
            if count != 0 and not 2 <= count <= 26:
                raise ValueError("A decision needs 2 to 26 choices; text requests may have no choices.")
            results.append({"candidate_scores": score[:count], "gate_probability": probability, "route": route})
        return results

    @contextmanager
    def _native_projections(self):
        """Temporarily remove adapter wrappers for an independent original-Qwen baseline."""
        restored = []
        try:
            for index, layer in enumerate(self.decoder.layers):
                if index < self.gate_after:
                    continue
                for parent in (layer.mlp, getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
                    if parent is None:
                        continue
                    for attr, module in list(parent.named_children()):
                        if isinstance(module, SwitchableLinear):
                            restored.append((parent, attr, module))
                            setattr(parent, attr, module.original)
            yield
        finally:
            for parent, attr, wrapped in restored:
                setattr(parent, attr, wrapped)

    @torch.inference_mode()
    def generate_request(self, row, mode="off", max_new_tokens=128, seed=42):
        if mode not in {"native", "off", "on", "auto"}:
            raise ValueError("Unknown generation mode.")
        if self._context is not None or self._stage == "lora":
            raise RuntimeError("Run generation in the evaluation stage, after LoRA backward is complete.")
        self.eval()
        probability = None if mode == "native" else self.gate_probabilities([row])[0]
        route = int(mode == "on" or (mode == "auto" and probability >= self.threshold))
        encoded = self.encode_rows([row])
        self._set_route(route)
        trace = {"adapter_count": len(self._projections), "prefill_projection_calls": 0,
                 "decode_projection_calls": 0}
        for projection in self._projections:
            projection.trace = trace
        try:
            devices = [self.device] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                kwargs = dict(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                              use_cache=True, logits_to_keep=1, pad_token_id=self.tokenizer.pad_token_id)
                if mode == "native":
                    with self._native_projections():
                        result = self.base.generate(**kwargs)
                else:
                    result = self.base.generate(**kwargs)
            tokens = result[0, encoded["input_ids"].shape[1]:].cpu().tolist()
            return {"generated_ids": tokens, "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                    "gate_probability": probability, "route": route, "trace": trace}
        finally:
            self._set_route(0)
            for projection in self._projections:
                projection.trace = None

    @torch.inference_mode()
    def route_request(self, row, max_new_tokens=128):
        result = self.score_requests([row], "auto")[0]
        if result["route"] == 1:
            if not result["candidate_scores"]:
                return dict(result, routing_error="FAST requested a decision but no options were supplied.")
            index = max(range(len(result["candidate_scores"])), key=result["candidate_scores"].__getitem__)
            return dict(result, selected_index=index, answer=row["options"][index], generated_ids=[])
        generated = self.generate_request(row, "off", max_new_tokens=max_new_tokens)
        return dict(result, answer=generated["text"], generated_ids=generated["generated_ids"])

    def trainable_state_dict(self):
        return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                for name, module in (("adapters", self.adapters), ("gate", self.gate))}

    def load_trainable_state_dict(self, state):
        if set(state) != {"adapters", "gate"}:
            raise ValueError("Expected only adapter and FAST gate weights.")
        self.adapters.load_state_dict(state["adapters"], strict=True)
        self.gate.load_state_dict(state["gate"], strict=True)

    def description(self):
        return {"gate_after": self.gate_after, "layer_count": len(self.decoder.layers),
                "hidden_size": self.hidden_size, "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
                "threshold": self.threshold, "max_length": self.max_length,
                "original_parameters": sum(p.numel() for p in self._original_parameters),
                "lora_parameters": sum(p.numel() for p in self.adapters.parameters()),
                "gate_parameters": sum(p.numel() for p in self.gate.parameters()),
                "adapter_count": len(self._projections), "adapter_targets": self.adapter_targets,
                "adapted_layers": list(range(self.gate_after + 1, len(self.decoder.layers) + 1)),
                "adapted_positions": "all prompt positions; also all decoded tokens when forced on",
                "gate_candidate_fallback": False}
