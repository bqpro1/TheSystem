"""G4 checks and model loading. No model weights are changed by the preflight."""
from collections import Counter
from contextlib import contextmanager
import gc
import importlib.metadata
import inspect
import math
import platform
import threading
import time

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .data import MODEL_ID, MODEL_REVISION
from .model import InternalFAST

KERNEL_CALLS = Counter()


@contextmanager
def live_status(label):
    """Show activity while downloads, CUDA compilation or loading has no progress callback."""
    start, stop = time.monotonic(), threading.Event()
    def report():
        while not stop.wait(30):
            print(f"{label}: still running ({(time.monotonic() - start)/60:.1f} min).", flush=True)
    thread = threading.Thread(target=report, daemon=True)
    print(label, flush=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def check_gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a G4 GPU runtime with native BF16, then reconnect.")
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory / 2**30 < 39:
        raise RuntimeError("This full-BF16 9B recipe needs at least 40 GB GPU memory. Select G4; no automatic quantization.")
    if torch.version.cuda != "13.0":
        raise RuntimeError("Use the setup cell's PyTorch CUDA 13.0 build, then restart.")
    print(f"GPU: {gpu.name} | {gpu.total_memory / 2**30:.1f} GiB | BF16 | CUDA {torch.version.cuda}")
    return {"gpu": gpu.name, "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "cuda": torch.version.cuda, "precision": "bfloat16",
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("transformers", "huggingface-hub", "accelerate", "safetensors", "numpy",
                          "flash-linear-attention", "fla-core", "causal-conv1d")}}


def _relative_error(actual, expected):
    error = ((actual.float() - expected).square().mean() / expected.square().mean().clamp_min(1e-12)).sqrt().item()
    if not math.isfinite(error) or error > 0.05:
        raise RuntimeError(f"Accelerated GPU operation failed its reference check ({error=}). Training has not started.")
    return error


def enable_fast_kernels():
    """Use the training-capable kernel path from the earlier successful G4 run."""
    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    with live_status("Compile and check accelerated GPU operations"), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(42)
        q, k, v = [torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        g = (-torch.rand(1, 65, 2, device="cuda")).requires_grad_()
        beta = torch.rand(1, 65, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        actual, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        def normalize(t):
            return t.float() * torch.rsqrt(t.float().square().sum(-1, keepdim=True) + 1e-6)
        expected, _ = naive_recurrent_gated_delta_rule(normalize(q), normalize(k), v.float(), beta.float(), g)
        delta_error = _relative_error(actual, expected)
        actual.float().square().mean().backward()
        if any(t.grad is None or not bool(torch.isfinite(t.grad).all()) for t in (q, k, v, g, beta)):
            raise RuntimeError("Attention backward check failed. Training has not started.")
        inputs = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                  for shape in ((1, 64, 65), (64, 4), (64,))]
        reference_inputs = [t.detach().float().requires_grad_() for t in inputs]
        observed = causal_conv1d_fn(*inputs, activation="silu")
        x, w, bias = reference_inputs
        reference = F.silu(F.conv1d(x, w[:, None], bias, padding=3, groups=64)[..., :65])
        upstream = torch.randn_like(observed)
        gradients = torch.autograd.grad(observed, inputs, upstream)
        reference_gradients = torch.autograd.grad(reference, reference_inputs, upstream.float())
        conv_errors = [_relative_error(a, b) for a, b in zip((observed, *gradients), (reference, *reference_gradients))]
    def counted(name, function):
        parameters = inspect.signature(function).parameters
        def call(*args, **kwargs):
            KERNEL_CALLS[name] += 1
            return function(*args, **{k: v for k, v in kwargs.items() if k in parameters})
        return call
    qwen.torch_chunk_gated_delta_rule = counted("linear_attention", chunk_gated_delta_rule)
    qwen.causal_conv1d_fn = counted("convolution", causal_conv1d_fn)
    KERNEL_CALLS.clear()
    print("GPU attention and convolution passed forward/backward checks.")
    return {"attention_relative_error": delta_error, "convolution_relative_errors": conv_errors}


def load_model(token=None, cache_dir=None):
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration
    with live_status("Download/load the pinned Qwen3.5-9B checkpoint (about 18 GB)"):
        # Transformers 5.17's Qwen2Tokenizer reconstructs an older pre-tokenizer
        # pattern. Keep Qwen3.5's published tokenizer.json, including Unicode marks.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token,
                                                            cache_dir=cache_dir, trust_remote_code=False)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=token, cache_dir=cache_dir,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False)
    if (base.config.text_config.num_hidden_layers, base.config.text_config.hidden_size) != (32, 4096):
        raise ValueError("Unexpected Qwen architecture; stop before training.")
    model = InternalFAST(base, tokenizer, gate_after=16, lora_rank=16, lora_alpha=32., max_length=2048)
    print(f"Qwen stays frozen. Trainable additions: {model.description()['lora_parameters']:,} LoRA + {model.description()['gate_parameters']:,} FAST parameters.")
    return model, tokenizer


def preflight(model, decision_rows, settings):
    """Exercise longest-input backward AND Adam allocation, then restore additions."""
    snapshot = {k: v.detach().cpu().clone() for k, v in model.adapters.state_dict().items()}
    versions = [p._version for p in model._original_parameters]
    selected = sorted(decision_rows, key=lambda r: len(r["input_ids"]), reverse=True)[:settings.micro_batch_size]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start, timings = time.monotonic(), []
    try:
        with torch.random.fork_rng(devices=[0]), live_status("Check longest-input training, memory and generation"):
            model.stage("lora")
            optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=settings.lora_lr)
            for _ in tqdm(range(2), desc="GPU training preflight", unit="step"):
                step = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                logits = model.decision_logits(selected, route="on")
                targets = torch.tensor([r["target_index"] for r in selected], device=model.device)
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite initial loss. Training has not started.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Non-finite initial gradient. Training has not started.")
                optimizer.step()
                torch.cuda.synchronize()
                timings.append(time.monotonic() - step)
            if not KERNEL_CALLS["convolution"] or not KERNEL_CALLS["linear_attention"]:
                raise RuntimeError("Accelerated training kernels were not used. Do not start a slow fallback run.")
            model.stage("evaluation")
            short = min(decision_rows, key=lambda r: len(r["input_ids"]))
            native = model.generate_request(short, mode="native", max_new_tokens=8)
            off = model.generate_request(short, mode="off", max_new_tokens=8)
            on = model.generate_request(short, mode="on", max_new_tokens=8)
            if native["generated_ids"] != off["generated_ids"]:
                raise AssertionError("Original generation differs with additions disabled. Stop before training.")
            if not on["trace"]["prefill_projection_calls"]:
                raise AssertionError("Always-on LoRA was not active during the prompt.")
            if len(on["generated_ids"]) > 1 and not on["trace"]["decode_projection_calls"]:
                raise AssertionError("Always-on LoRA was not active during generation.")
            peak = torch.cuda.max_memory_allocated() / 2**30
    finally:
        model.adapters.load_state_dict(snapshot)
        model.zero_grad(set_to_none=True)
        model.stage("evaluation")
        if "optimizer" in locals():
            del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if any(p.requires_grad or p.grad is not None or p._version != v for p, v in zip(model._original_parameters, versions)):
        raise AssertionError("An original Qwen parameter changed during preflight.")
    print(f"Preflight passed; peak allocated GPU memory {peak:.1f} GiB. Temporary adapter updates were undone.")
    print("Training progress bars will estimate remaining time from actual batches; first compilation is excluded.")
    return {"passed": True, "peak_allocated_gib": peak, "longest_batch_tokens": [len(r["input_ids"]) for r in selected],
            "training_step_seconds": timings, "elapsed_seconds": time.monotonic() - start,
            "native_off_tokens_equal": True, "kernel_calls": dict(KERNEL_CALLS)}
