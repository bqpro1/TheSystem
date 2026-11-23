"""Inference-only Qwen execution using Transformers' native PyTorch operations.

The reference training/evaluation runtime is deliberately separate. This path
reuses Colab's PyTorch and does not require or compile optional CUDA extensions.
"""
from __future__ import annotations

import inspect
from types import FunctionType, MethodType

from .artifacts import Variant, read_release_manifest

TRANSFORMERS_VERSION = "5.17.0"
_NATIVE_FUNCTIONS = (
    "causal_conv1d_fn", "causal_conv1d_update",
    "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
)


def _check_transformers() -> None:
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"The inference demo requires transformers=={TRANSFORMERS_VERSION}; "
            "run Prepare runtime in a fresh Colab session."
        )


def verify_runtime() -> None:
    """Check inference hardware, without training dependencies or CUDA builds."""
    import torch
    _check_transformers()
    if not torch.cuda.is_available():
        raise RuntimeError("Select the latest Colab runtime with a native-BF16 NVIDIA GPU, then reconnect.")
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if (torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            f"This demo requires native BF16 support; "
            f"available: {torch.cuda.get_device_name(0)} ({memory:.1f} GiB)."
        )
    print(f"Inference runtime: {torch.cuda.get_device_name(0)} ({memory:.1f} GiB), "
          f"PyTorch {torch.__version__}, Transformers {TRANSFORMERS_VERSION}.", flush=True)


def _native_function(function, expected_name: str, module):
    """Select the documented Torch fallback beneath optional-kernel decorators."""
    native = inspect.unwrap(function)
    if (not isinstance(native, FunctionType) or native.__name__ != expected_name
            or native.__module__ != module.__name__
            or native.__code__.co_filename != module.__file__):
        raise RuntimeError(
            "Qwen's runtime functions were changed by another setup. Restart the "
            "Colab session and run only this demo's Prepare runtime cell."
        )
    return native


def use_native_inference(base) -> int:
    """Select native kernels on this model only, without changing its tensors.

    Qwen's forward looks up kernel functions in module globals. A private copy
    of that globals mapping avoids changing other models or the reference path.
    Configure this before FrozenQwen records the backbone's module identities.
    """
    _check_transformers()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    native = {name: _native_function(getattr(qwen, name), name, qwen)
              for name in _NATIVE_FUNCTIONS}
    original = _native_function(qwen.Qwen3_5GatedDeltaNet.forward, "forward", qwen)
    forward = FunctionType(original.__code__, {**original.__globals__, **native},
                           original.__name__, original.__defaults__, original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    # Qwen reads conv1d weights directly; retain upstream Accelerate hook handling.
    forward = qwen.force_accelerate_hooks("conv1d")(forward)
    norm_forward = _native_function(qwen.Qwen3_5RMSNormGated.forward, "forward", qwen)
    count = 0
    for module in base.modules():
        if isinstance(module, qwen.Qwen3_5GatedDeltaNet):
            module.forward = MethodType(forward, module)
            count += 1
        elif isinstance(module, qwen.Qwen3_5RMSNormGated):
            module.forward = MethodType(norm_forward, module)
    if not count:
        raise ValueError("Expected Qwen3.5 linear-attention layers")
    return count


def load_model(variant: Variant, token=None):
    """Load the pinned backbone and unchanged selected-model wrapper for demos."""
    if variant not in ("probe", "lora"):
        raise ValueError("variant must be 'probe' or 'lora'")
    verify_runtime()
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration

    backbone = read_release_manifest()["base_model"]
    common = dict(revision=backbone["revision"], token=token, trust_remote_code=False)
    print("Loading the pinned tokenizer and BF16 Qwen backbone…", flush=True)
    # This exact tokenizer class preserves the release's tokenizer.json contract.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(backbone["repo_id"], **common)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        backbone["repo_id"], **common, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", use_kernels=False,
    )
    config = base.config.text_config
    if (config.num_hidden_layers, config.hidden_size) != (32, 4096):
        raise ValueError("Expected the pinned 32-layer, 4096-unit Qwen3.5-9B backbone")
    count = use_native_inference(base)
    if variant == "probe":
        from ._vendor.probe.model import FrozenQwen
        reference = FrozenQwen(base, tokenizer, gate_after=16, max_length=2048)
    else:
        from ._vendor.lora.model import InternalFAST
        reference = InternalFAST(base, tokenizer, gate_after=16, lora_rank=16,
                                 lora_alpha=32.0, max_length=2048)
    print(f"Native PyTorch inference selected for {count} linear-attention layers; "
          "loading the selected release weights next.", flush=True)
    return reference, tokenizer
