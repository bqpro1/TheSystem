"""Original Qwen plus a separate frozen FAST gate; no backbone modifications."""
import string

import torch
from torch import nn

SYSTEM_PROMPT = "You are a helpful assistant."


def normalized_features(hidden):
    hidden = hidden.float()
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)


class PrefixReady(Exception):
    """Stop a routing-only forward after the gate's input layer."""


class FrozenQwen(nn.Module):
    def __init__(self, base, tokenizer, gate_after=16, max_length=2048, threshold=0.5):
        super().__init__()
        self.base, self.tokenizer = base, tokenizer
        self.gate_after, self.max_length, self.threshold = gate_after, max_length, threshold
        if self.decoder.config.model_type != "qwen3_5_text":
            raise ValueError("Expected the pinned Qwen3.5 text decoder")
        if not 0 < gate_after < len(self.decoder.layers) or threshold != 0.5:
            raise ValueError("Invalid gate position or changed fixed threshold")
        self.hidden_size = self.decoder.config.hidden_size
        self.base.requires_grad_(False).eval()
        self.gate = nn.Linear(self.hidden_size, 1, device=self.device, dtype=torch.float32)
        self.gate.requires_grad_(False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in string.ascii_uppercase[:18]]
        if any(len(tokens) != 1 for tokens in ids) or len({t[0] for t in ids}) != 18:
            raise ValueError("A–R must be distinct single tokens")
        self.register_buffer("choice_ids", torch.tensor([t[0] for t in ids], device=self.device))
        self._context, self._stage = None, "evaluation"
        self._base_modules = tuple((n, id(m), type(m)) for n, m in base.named_modules())
        self._base_parameters = tuple((n, id(p), p._version) for n, p in base.named_parameters())

    @property
    def decoder(self):
        model = self.base.model
        return model.language_model if hasattr(model, "language_model") else model

    @property
    def device(self):
        return self.base.get_input_embeddings().weight.device

    def stage(self, name):
        if name != "evaluation":
            raise ValueError("The backbone and FAST gate never train in this experiment")
        self.requires_grad_(False).eval()

    def assert_unchanged(self):
        if self._base_modules != tuple((n, id(m), type(m)) for n, m in self.base.named_modules()):
            raise AssertionError("A backbone module was added, replaced or removed")
        if self._base_parameters != tuple((n, id(p), p._version) for n, p in self.base.named_parameters()):
            raise AssertionError("An original Qwen parameter changed")
        if any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise AssertionError("Qwen and FAST must stay frozen")

    def encode_rows(self, rows):
        if not rows:
            raise ValueError("Empty input batch")
        ids = []
        for row in rows:
            if "input_ids" in row:
                tokens = list(row["input_ids"])
            else:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if not 0 < len(tokens) <= self.max_length:
                raise ValueError(f"{row.get('id', '?')}: input exceeds {self.max_length}; no truncation")
            ids.append(tokens)
        width = max(map(len, ids))
        return {
            "input_ids": torch.tensor([x + [self.tokenizer.pad_token_id] * (width-len(x)) for x in ids], device=self.device),
            "attention_mask": torch.tensor([[1]*len(x) + [0]*(width-len(x)) for x in ids], device=self.device),
        }

    def gate_logits(self, features):
        return self.gate(features.to(device=self.device, dtype=torch.float32)).squeeze(-1)

    @torch.no_grad()
    def gate_probabilities(self, rows):
        if self._context is not None:
            raise RuntimeError("Concurrent model calls are unsupported")
        encoded = self.encode_rows(rows)
        last = encoded["attention_mask"].sum(1).long() - 1
        captured = {}

        def read(module, inputs, hidden):
            features = normalized_features(hidden[torch.arange(len(rows), device=self.device), last])
            captured["probability"] = self.gate_logits(features).sigmoid().cpu().tolist()
            raise PrefixReady()

        self._context = captured
        handle = self.decoder.layers[self.gate_after - 1].register_forward_hook(read)
        try:
            try:
                self.decoder(**encoded, use_cache=False)
            except PrefixReady:
                pass
            return captured["probability"]
        finally:
            handle.remove()
            self._context = None
