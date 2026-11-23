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

from .release_data import MODEL_ID, MODEL_REVISION
from .model import FrozenQwen

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
    model = FrozenQwen(base, tokenizer, gate_after=16, max_length=2048)
    print("Original Qwen loaded unchanged; zero backbone adapters.")
    return model, tokenizer

