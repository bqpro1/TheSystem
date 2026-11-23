"""Prepare the pinned evaluation stack and persistent output storage.

No model, dataset, prompt, scorer or numerical kernel is changed by this helper.
"""
import csv
import math
import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path

BOOTSTRAP = {
    "repo_id": "Lbuk/TheSystem-9B-Probe-256",
    "revision": "c533a07ef0606b5e08ade1522f6ae030247ae4a7",
    "filename": "thesystem-0.1.0rc4-py3-none-any.whl",
    "sha256": "255bad2fb39360aeb05ef497ba477c1cc9893107d27bce4343ff0d3d37aa0e00",
}

from contextlib import contextmanager
import threading
import time


class RestartRequired(RuntimeError):
    """The installed packages need a fresh kernel before evaluation."""


@contextmanager
def visible_progress(label):
    """Show elapsed time while a long load/download is active."""
    stopped = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stopped.wait(30):
            print(f"{label}: active for {(time.monotonic() - started) / 60:.1f} min; waiting for completion.", flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    print(label, flush=True)
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def check_preflight():
    print("Python:", platform.python_version(), flush=True)
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError(
            "Use Python 3.12 or 3.13. In Colab, select Runtime → Change runtime type → "
            "the latest runtime version, reconnect, then Run all again. "
            "The recorded GPU runs used Python 3.13. No packages were installed."
        )
    if platform.system() != "Linux":
        raise RuntimeError("Use a supported Linux NVIDIA GPU runtime. This notebook is not a Mac/CPU demo.")
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No NVIDIA GPU found. Select a GPU runtime, then run this cell again.")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not gpu_id or gpu_id == "-1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all GPUs. Select a visible NVIDIA GPU, then retry.")
    gpu_info = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    print("NVIDIA GPU — name, driver, total memory (MiB):", flush=True)
    print(gpu_info.stdout.strip(), flush=True)
    try:
        gpu_row = next(csv.reader(gpu_info.stdout.splitlines(), skipinitialspace=True))
        memory_gib = float(gpu_row[2]) / 1024
    except (StopIteration, IndexError, ValueError) as exc:
        raise RuntimeError("Could not read GPU memory from nvidia-smi. Check the output above before installing.") from exc
    if not math.isfinite(memory_gib) or memory_gib <= 0:
        raise RuntimeError("Could not read a valid GPU memory capacity. Inspect nvidia-smi output.")
    print("No memory-size cutoff. Setup will check native BF16 support.", flush=True)


def download_wheel():
    import hashlib
    import re
    from huggingface_hub import hf_hub_download

    if not re.fullmatch(r"[0-9a-f]{40}", BOOTSTRAP["revision"]):
        raise RuntimeError("This notebook's release revision has not been finalized. Obtain the pinned release notebook.")
    if not re.fullmatch(r"[0-9a-f]{64}", BOOTSTRAP["sha256"]):
        raise RuntimeError("This notebook's wheel checksum has not been finalized.")

    wheel = Path(hf_hub_download(
        repo_id=BOOTSTRAP["repo_id"], revision=BOOTSTRAP["revision"],
        filename=BOOTSTRAP["filename"], token=False,
    ))
    actual = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual != BOOTSTRAP["sha256"]:
        raise RuntimeError("Wheel checksum mismatch. Stop and obtain a fresh copy of the pinned release.")
    from importlib.metadata import distribution, PackageNotFoundError
    import json
    loaded_runtime = any(name == "thesystem" or name.startswith("thesystem.") for name in sys.modules)
    try:
        previous = json.loads(distribution("thesystem").read_text("direct_url.json") or "{}").get("archive_info", {})
    except PackageNotFoundError:
        previous = {}
    previous_sha = previous.get("hashes", {}).get("sha256") or previous.get("hash", "").removeprefix("sha256=")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps", str(wheel)], check=True)
    if loaded_runtime and previous_sha != BOOTSTRAP["sha256"]:
        raise RestartRequired("Restart the kernel to load the installed evaluation wheel.")
    print("Verified and installed the pinned TheSystem runtime wheel.")


def install_runtime():
    setup = subprocess.run([sys.executable, "-m", "thesystem._vendor.lora.setup_runtime"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if setup.returncode == 10:
        raise RestartRequired("Restart the kernel to load the installed GPU packages.")
    if setup.returncode != 0:
        print(setup.stdout, flush=True)
        raise RuntimeError(f"Runtime setup failed with exit code {setup.returncode}; inspect the setup output above.")
    # The installer just verified a fresh process. Colab may still hold older imports.
    from importlib.metadata import version
    for name in ("torch", "transformers"):
        loaded = sys.modules.get(name)
        if loaded is not None and loaded.__version__ != version(name):
            raise RestartRequired(f"Restart the kernel to load the installed {name}.")
    from thesystem._vendor.lora.setup_runtime import verify_runtime
    verify_runtime()
    from thesystem.evaluation.runtime import check_gpu
    check_gpu()


def prepare_runtime():
    print("Preparing the reference evaluation runtime. Long steps report every 30 seconds.", flush=True)
    try:
        with visible_progress("[1/4] Check Python and the GPU"):
            check_preflight()
        with visible_progress("[2/4] Install the pinned download client"):
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "huggingface-hub==1.31.0"], check=True)
        with visible_progress("[3/4] Download and verify the evaluation wheel"):
            download_wheel()
        with visible_progress("[4/4] Prepare and verify the reference GPU software"):
            install_runtime()
    except RestartRequired:
        return False
    print("READY — continue to step 2.", flush=True)
    return True


def prepare_output(RUN_NAME, USE_DRIVE=False):
    runtime = None
    try:
        from google.colab import drive, runtime
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB and USE_DRIVE:
        drive.mount("/content/drive")
        output_base = Path("/content/drive/MyDrive/TheSystem-runs")
    else:
        output_base = (Path("/content") if IN_COLAB else Path.cwd()) / "thesystem-runs"
    OUTPUT = Path(os.environ.get("THESYSTEM_OUTPUT", output_base / RUN_NAME)).expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Downloads and feature/model caches stay on fast local runtime disk.
    CACHE = Path(os.environ.get("THESYSTEM_MODEL_CACHE", "/content/thesystem-model-cache" if IN_COLAB else Path.home() / ".cache/thesystem/models"))
    CACHE.mkdir(parents=True, exist_ok=True)
    probe_file = OUTPUT / ".write-check"
    probe_file.write_text("Storage check\n")
    probe_file.unlink()
    print("New outputs:", OUTPUT)
    print("Local model cache:", CACHE)
    return OUTPUT, CACHE, IN_COLAB, runtime


def results_html(output):
    import html
    import json
    report = json.loads((Path(output) / "SUMMARY.json").read_text())
    if not report.get("complete"):
        raise RuntimeError("Evaluation is incomplete; download partial results and logs.")
    names = {"vitaminc-dev": "VitaminC", "paws": "PAWS", "squad2": "SQuAD2 answerability", "aegis2": "Aegis2", "pubmedqa": "PubMedQA"}
    source = html.escape(report["published"]["source_url"], quote=True)
    if not report["published"]["source_url"].startswith("https://"):
        raise ValueError("Expected an HTTPS published-results link")
    rows = []
    for key, name in names.items():
        values = report["decisions"][key]
        published = report["published"]["subsets"][key]
        cells = [f'<th>{name}<small>{values["qwen"]["rows"]} examples</small></th>']
        cells += [f'<td>{values[k]["accuracy"] * 100:.2f}%</td>' for k in ("lora_automatic", "probe256_automatic")]
        cells += [f'<td><a href="{source}" target="_blank" rel="noopener noreferrer">{published[k]["accuracy_percent"]:.1f}% ↗</a></td>' for k in ("nimble", "jev")]
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    label = 'QUICK CHECK — small subsets; not full benchmark scores' if report.get('smoke') else 'Benchmark accuracy — this evaluation run'
    if not report.get('inference_performed', True):
        label += ' · reconstructed evidence, no new inference'
    return '''<style>.ts-results{font:14px system-ui;color:#e8ebf0;background:#101318;border:1px solid #303640;border-radius:12px;padding:20px;overflow:auto}.ts-results table{width:100%;border-collapse:collapse}.ts-results th,.ts-results td{padding:16px;text-align:right;border-bottom:1px solid #303640}.ts-results th:first-child{text-align:left}.ts-results thead{background:#1b2028}.ts-results a{color:#9ac9ff}.ts-results small{display:block;color:#a7afbb;font-weight:400;margin-top:5px}</style>''' + f'''<div class="ts-results"><h3>{label}</h3><table><thead><tr><th>Benchmark</th><th>&lt;FAST&gt; + LoRA</th><th>&lt;FAST&gt; + Probe-256</th><th><a href="https://huggingface.co/bespokelabs/Bespoke-Nimble-9B">Bespoke-Nimble-9B</a></th><th><a href="https://docs.typesafe.ai/models#current-models">Jev 1.13.0</a></th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>Automatic accuracy includes routing errors. Nimble/Jev are published reference scores; prompt wrappers differ. Full Qwen, routing and generation results are in REPORT.html.</p></div>'''


def show_results(output):
    from IPython.display import HTML, display
    display(HTML(results_html(output)))


def finish_controls(output):
    import tempfile
    import ipywidgets as widgets
    from IPython.display import display, FileLink
    output = Path(output).resolve()
    download = widgets.Button(description="Download results ZIP", icon="download")
    saved = widgets.Checkbox(value=False, description="I confirmed that my results are saved", indent=False)
    shutdown = widgets.Button(description="Shut down runtime", disabled=True, button_style="warning")
    messages = widgets.Output()
    def download_results(_):
        saved.value = False
        with messages:
            archive = shutil.make_archive(str(Path(tempfile.mkdtemp(prefix="thesystem-results-")) / output.name), "zip", root_dir=output)
            try:
                from google.colab import files
            except ImportError:
                display(FileLink(archive))
            else:
                files.download(archive)
            print("Download requested. Verify the ZIP is saved before confirming shutdown.")
    def confirmation(change):
        shutdown.disabled = not change["new"]
    def stop(_):
        if not saved.value:
            return
        with messages:
            try:
                from google.colab import runtime
            except ImportError:
                print("This local kernel stays open. Close it when ready.")
            else:
                print("You confirmed that results are saved. Releasing Colab.", flush=True)
                runtime.unassign()
    download.on_click(download_results)
    saved.observe(confirmation, names="value")
    shutdown.on_click(stop)
    display(widgets.VBox([download, saved, shutdown, messages]))
    return download, saved, shutdown
