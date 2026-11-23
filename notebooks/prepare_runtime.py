"""Inference-only Colab setup: reuse PyTorch, install binary Python packages, never build CUDA."""
from contextlib import contextmanager
import csv
import hashlib
from importlib.metadata import distribution, PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time


WHEELS = {
    "probe": {
        "repo_id": "Lbuk/TheSystem-9B-Probe-256",
        "revision": "9b7ed1d287bc151d1db7e3c51d5d4d933195d976",
    },
    "lora": {
        "repo_id": "Lbuk/TheSystem-9B-LoRA",
        "revision": "d03d2d9475a3e1801403f8cffa7fb3d05b11187e",
    },
}
WHEEL_NAME = "thesystem-0.1.0rc3-py3-none-any.whl"
WHEEL_SHA256 = 'f315e2cf911edd113bbde3720ef36ba8dd33d2437b2d1e02c4e00ac40cca12c1'


class RestartRequired(RuntimeError):
    """Stop Run all before a model can use stale imports."""


def gpu_memory_status():
    """Best-effort feedback, including after an allocation failed."""
    torch = sys.modules.get("torch")
    try:
        free, total = torch.cuda.mem_get_info()
        return f"GPU memory: {free / 1024**3:.1f} GiB free of {total / 1024**3:.1f} GiB total."
    except Exception:
        return "GPU memory information is unavailable."


@contextmanager
def visible_progress(label, interval=20):
    """Immediate status, elapsed-time heartbeats, and an explicit outcome."""
    started, stopped = time.monotonic(), threading.Event()
    print(label, flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            print(f"  Still running: {label} — {time.monotonic() - started:.0f}s elapsed.", flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except RestartRequired:
        print("  Installation saved; a session/kernel restart is required.", flush=True)
        raise
    except BaseException as error:
        torch = sys.modules.get("torch")
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
        if isinstance(error, oom_type):
            message = (
                f"GPU memory ran out during: {label}. {gpu_memory_status()} "
                "This operation did not complete. If loading failed, restart the session to clear "
                "partially loaded weights, then load only one model. For a request, try a shorter "
                "prompt or fewer max_new_tokens; restart first if memory remains occupied. "
                "If it still does not fit, choose a GPU with more memory. "
                "No automatic quantization, CPU offload or input truncation was applied."
            )
            print(message, flush=True)
            raise RuntimeError(message) from None
        print(f"  Stopped: {label} — {type(error).__name__}. See the message below.", flush=True)
        raise
    else:
        print(f"  Done — {time.monotonic() - started:.1f}s.", flush=True)
    finally:
        stopped.set()
        worker.join()


def check_preflight():
    print("Python:", platform.python_version(), flush=True)
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError(
            "Use Python 3.12 or 3.13. In Colab, select Runtime → Change runtime type → "
            "the latest runtime version, reconnect, then rerun Prepare runtime. "
            "The recorded GPU runs used Python 3.13. No packages were installed."
        )
    if platform.system() != "Linux" or not shutil.which("nvidia-smi"):
        raise RuntimeError("Use a Linux NVIDIA GPU runtime. This is not a Mac/CPU demo.")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not gpu_id or gpu_id == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all GPUs. Select a visible NVIDIA GPU.")
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=30,
    )
    print("NVIDIA GPU — name, driver, total memory (MiB):", flush=True)
    print(result.stdout.strip(), flush=True)
    try:
        row = next(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
        memory_gib = float(row[2]) / 1024
    except (StopIteration, IndexError, ValueError) as error:
        raise RuntimeError("Could not read GPU memory. Inspect nvidia-smi output before installing.") from error
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise RuntimeError("Could not read a valid GPU memory capacity. Inspect nvidia-smi output.")
    print("No memory-size cutoff; native BF16 support will be verified next.", flush=True)


def run_command(command, log_path, allowed=(0,), env=None):
    """Keep detailed output on disk while the enclosing stage shows heartbeats."""
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} — {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            returncode = child.wait()
        except BaseException:
            print("Stopping setup subprocesses…", flush=True)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if child.poll() is not None:
                    break
                try:
                    os.killpg(child.pid, sig)
                    child.wait(timeout=5)
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise
    if returncode not in allowed:
        tail = Path(log_path).read_text(errors="replace")[-6000:]
        print("Last setup log lines:\n" + tail, flush=True)
        raise RuntimeError(f"Setup command failed (exit {returncode}). Full log: {log_path}")
    return returncode


INFERENCE_REQUIREMENTS = {
    "transformers": "transformers==5.17.0",
    "huggingface-hub": "huggingface-hub>=1.31,<2",
    "accelerate": "accelerate>=1.15,<2",
    "safetensors": "safetensors>=0.8,<1",
}


def missing_requirements():
    from packaging.requirements import Requirement
    missing = []
    for name, spec in INFERENCE_REQUIREMENTS.items():
        try:
            compatible = version(name) in Requirement(spec).specifier
        except PackageNotFoundError:
            compatible = False
        if not compatible:
            missing.append(spec)
    return missing


def install_inference_dependencies(log_path):
    missing = missing_requirements()
    if not missing:
        print("Inference libraries are already compatible; no installation needed.", flush=True)
        return
    print("Installing binary packages: " + ", ".join(missing), flush=True)
    # The resolver must not replace the working Colab PyTorch or NumPy indirectly.
    protected = {name: version(name) for name in ("torch", "numpy")}
    with tempfile.TemporaryDirectory(prefix="thesystem-inference-") as tmp:
        constraints = Path(tmp) / "constraints.txt"
        constraints.write_text("".join(f"{name}=={value}\n" for name, value in protected.items()))
        run_command([sys.executable, "-m", "pip", "install", "--only-binary=:all:",
                     "--upgrade-strategy", "only-if-needed", "--constraint", str(constraints), *missing], log_path)
    if any(version(name) != value for name, value in protected.items()):
        raise RuntimeError("The existing PyTorch/NumPy environment changed unexpectedly. Start a fresh runtime.")
    if missing_requirements():
        raise RuntimeError("Inference dependencies are still incompatible. Inspect the setup log.")
    check_loaded_versions()


def check_loaded_versions():
    """Colab may preload NumPy; replacing its files does not refresh those imports."""
    for name in ("numpy", "torch", "transformers", "huggingface_hub", "accelerate", "safetensors", "thesystem"):
        module = sys.modules.get(name)
        loaded = getattr(module, "__version__", None)
        if loaded and str(loaded).split("+")[0] != version(name.replace("_", "-")).split("+")[0]:
            raise RestartRequired(f"{name} is still loaded from an earlier installation.")


def install_wheel(variant, log_path):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    try:
        wheel = Path(hf_hub_download(
            **WHEELS[variant], filename=WHEEL_NAME, token=os.environ.get("HF_TOKEN") or False,
        ))
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            raise RuntimeError(
                "Hugging Face denied access. Rerun Prepare runtime and enter a valid read token "
                "for the private model repository. Browser login does not authenticate Colab."
            ) from None
        raise
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("Wheel checksum mismatch. Stop and obtain a fresh release notebook.")
    try:
        previous = json.loads(distribution("thesystem").read_text("direct_url.json") or "{}").get("archive_info", {})
    except PackageNotFoundError:
        previous = {}
    previous_sha = previous.get("hashes", {}).get("sha256") or previous.get("hash", "").removeprefix("sha256=")
    loaded = any(name == "thesystem" or name.startswith("thesystem.") for name in sys.modules)
    if previous_sha == WHEEL_SHA256:
        print("Release wheel checksum verified; matching wheel already installed.", flush=True)
        return
    if loaded:
        raise RestartRequired("Restart before replacing the already imported release package.")
    run_command([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)], log_path)
    print("Release wheel checksum verified and installed.", flush=True)


def verify_live_runtime():
    """Check the existing GPU stack without importing any training installer."""
    check_loaded_versions()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is missing or broken. Connect a fresh latest Colab NVIDIA GPU runtime.") from error
    if not re.match(r"^2\.14\.\d+(?:[+.]|$)", torch.__version__):
        raise RuntimeError(
            f"This demo supports Colab PyTorch 2.14.x; found {torch.__version__}. "
            "Use the latest Colab runtime. Setup will not replace PyTorch."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Select an NVIDIA GPU runtime and reconnect.")
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This demo requires native BF16 support. Select a compatible NVIDIA GPU; T4 is unsupported.")
    print(f"Verified GPU: {torch.cuda.get_device_name(0)} ({memory_gib:.1f} GiB), native BF16.", flush=True)
    print(gpu_memory_status(), flush=True)
    print(f"Keeping PyTorch {torch.__version__}, CUDA {torch.version.cuda} and NumPy {version('numpy')}.", flush=True)
    return torch


def prepare_runtime(variant):
    if variant not in WHEELS:
        raise ValueError("Choose probe or lora.")
    started = time.monotonic()
    log_path = Path.cwd() / "thesystem-prepare-runtime.log"
    name = "Probe-256" if variant == "probe" else "LoRA"
    print(f"Preparing {name} for inference. Long steps report every 20 seconds.\nDetailed setup log: {log_path}", flush=True)
    print("Uses built-in PyTorch operations; no CUDA compilation or training checks.", flush=True)
    try:
        with visible_progress("[1/4] Check the existing Colab GPU runtime"):
            check_preflight()
            verify_live_runtime()
            print("BF16 weights alone use about 18 GB; loading and requests need extra memory. "
                  "You may try a 24 GB GPU, but fit is not yet validated.", flush=True)
        with visible_progress("[2/4] Install only missing inference libraries"):
            install_inference_dependencies(log_path)
        with visible_progress("[3/4] Download and verify the release wheel"):
            install_wheel(variant, log_path)
        with visible_progress("[4/4] Verify the active runtime"):
            torch = verify_live_runtime()
    except RestartRequired:
        message = (
            "RESTART REQUIRED: An imported library needs a fresh kernel. In Colab choose Runtime → Restart session; "
            "in VS Code restart the kernel. Then rerun Prepare runtime and enter your read token again. "
            "Installed packages stay in place."
        )
        print(message, flush=True)
        raise RestartRequired(message) from None
    except BaseException:
        print("Setup is not ready. Resolve the error above, then rerun Prepare runtime.", flush=True)
        raise
    print(f"READY — inference runtime prepared in {time.monotonic() - started:.0f}s. "
          f"Continue to Load {name}; setup does not guarantee the model fits in memory.", flush=True)
    return torch
