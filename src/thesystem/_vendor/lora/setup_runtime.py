"""Install the previously used G4 stack, with visible output and a restart boundary."""
import importlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def installed(name):
    try:
        return version(name).split("+")[0]
    except PackageNotFoundError:
        return None


def pins():
    return dict(line.split("==") for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def verify_runtime(gpu=True):
    """Checks the LIVE kernel too: correct files on disk do not remove stale imports."""
    importlib.invalidate_caches()
    import torch
    import transformers
    from transformers.generation import GenerationMixin
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if (torch.__version__.split("+")[0], transformers.__version__) != ("2.14.0", "5.17.0"):
        raise RuntimeError("Restart the Colab session/kernel, then Run all again: old modules are still loaded.")
    mismatches = {k: installed(k) for k, v in pins().items() if installed(k) != v}
    if mismatches:
        raise RuntimeError(f"Run the setup cell and restart; package mismatch: {mismatches}")
    if gpu:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Select the G4 GPU runtime with native BF16, then reconnect.")
        if torch.version.cuda != "13.0":
            raise RuntimeError("This G4 recipe requires the pinned PyTorch CUDA 13.0 build.")
    return {k: installed(k) for k in pins()}


def run_visible(command, title, env=None):
    stop = threading.Event()
    start = time.monotonic()
    def status():
        while not stop.wait(30):
            print(f"{title}: still running ({(time.monotonic()-start)/60:.1f} min).", flush=True)
    worker = threading.Thread(target=status, daemon=True)
    worker.start()
    print(title, flush=True)
    try:
        subprocess.run(command, check=True, env=env)
    finally:
        stop.set()
        worker.join()


def main():
    if sys.version_info < (3, 10):
        raise RuntimeError("Use a current Colab Python runtime (3.10 or newer).")
    expected = pins()
    base = {k: v for k, v in expected.items() if k != "causal-conv1d"}
    probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.version.cuda)"],
                           capture_output=True, text=True)
    wrong_torch_build = probe.returncode != 0 or probe.stdout.strip() != "13.0"
    changed = wrong_torch_build or any(installed(k) != v for k, v in base.items())
    extension_probe = subprocess.run([sys.executable, "-c", "from causal_conv1d import causal_conv1d_fn"],
                                     capture_output=True, text=True)
    conv_changed = changed or extension_probe.returncode != 0 or installed("causal-conv1d") != expected["causal-conv1d"]
    pip = [sys.executable, "-m", "pip"]
    if changed:
        unused = [k for k in ("torchao", "torchaudio", "torchtext", "librosa") if installed(k)]
        if unused:
            run_visible(pip + ["uninstall", "-y", *unused], "Remove incompatible optional runtime extensions")
        if wrong_torch_build or installed("torch") != base["torch"] or installed("torchvision") != base["torchvision"]:
            run_visible(pip + ["install", "--upgrade", "--force-reinstall", "--index-url", "https://download.pytorch.org/whl/cu130",
                               "torch==2.14.0", "torchvision==0.29.0"], "Install G4 PyTorch (CUDA 13.0)")
        run_visible(pip + ["install", "--upgrade", *[f"{k}=={v}" for k, v in base.items()
                                                       if k not in ("torch", "torchvision")]],
                    "Install pinned training packages")
    if conv_changed:
        run_visible(pip + ["install", "ninja", "packaging", "wheel", "setuptools"], "Prepare extension build tools")
        run_visible(pip + ["install", "--no-build-isolation", "--no-deps", "--force-reinstall", "--no-cache-dir",
                           "causal-conv1d==1.7.0"], "Install accelerated convolution; compilation may take several minutes",
                    env={**os.environ, "MAX_JOBS": "2"})
    run_visible([sys.executable, "-c", "from thesystem._vendor.lora.setup_runtime import verify_runtime; verify_runtime()"],
                "Check imports and CUDA in a fresh process")
    if changed or conv_changed:
        print("Setup completed. Restart the Colab session (or VS Code kernel), then Run all again.", flush=True)
        return 10
    print("Pinned packages and fresh-process imports passed. Continue with the live-kernel check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
