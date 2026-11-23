"""Reference evaluation hardware checks, without a training memory cutoff."""
import importlib.metadata
import platform

import torch


def check_gpu():
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Select a native-BF16 NVIDIA GPU runtime, then reconnect.")
    gpu = torch.cuda.get_device_properties(0)
    if torch.version.cuda != "13.0":
        raise RuntimeError("Use the setup cell's PyTorch CUDA 13.0 build, then restart.")
    print(f"GPU: {gpu.name} | {gpu.total_memory / 2**30:.1f} GiB | BF16 | CUDA {torch.version.cuda}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB total; no capacity cutoff.", flush=True)
    return {"gpu": gpu.name, "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "cuda": torch.version.cuda, "precision": "bfloat16",
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("transformers", "huggingface-hub", "accelerate", "safetensors", "numpy",
                          "flash-linear-attention", "fla-core", "causal-conv1d")}}

