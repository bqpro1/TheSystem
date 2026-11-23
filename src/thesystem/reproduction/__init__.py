"""Repeat the selected training recipes in separate, resumable GPU processes."""
import os
from pathlib import Path
import subprocess
import sys


def _run(module, arguments, root, log):
    """Stream subprocess output while preserving a durable local log."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, THESYSTEM_RELEASE_ROOT=str(root))
    command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    print("RUN", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8", buffering=1) as handle:
        child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            code = child.wait()
        except BaseException:
            import signal
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    if code:
        raise RuntimeError(f"Worker exited {code}; saved log: {log}")


def _paths(root, output, cache):
    from thesystem.artifacts import resolve_release_resources
    root = Path(resolve_release_resources(root)).resolve()
    output = Path(output).expanduser().resolve()
    cache = Path(cache).expanduser().resolve() if cache else output.parent / "model-cache"
    for path in (output, cache):
        if path == root or root in path.parents or path in root.parents:
            raise ValueError("Outputs and cache must be outside immutable release resources")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return root, output, cache


def train_probe(root, output, cache=None, smoke=False):
    """Train seed-42 Probe-256 on frozen features and evaluate the new export.

    The released weights are never replaced. Reusing an output directory resumes
    the same immutable run identity. Requires the documented BF16 GPU environment.
    """
    root, output, cache = _paths(root, output, cache)
    _run("thesystem.reproduction._probe_runner", ["train", "--output", output / "run", "--cache", cache, *(["--smoke"] if smoke else [])],
         root, output / "training.log")
    from .common import read_json, write_json
    summary = read_json(output / "run/SUMMARY.json")
    write_json(output / "SUMMARY.json", dict(summary, smoke=smoke))
    return output


def train_lora(root, output, cache=None, smoke=False):
    """Repeat LoRA, original gate, diverse-gate continuation and evaluation."""
    root, output, cache = _paths(root, output, cache)
    worker = "thesystem.reproduction._lora_runner"
    original, diverse = output / "original", output / "diverse-gate"
    smoke_args = ["--smoke"] if smoke else []
    _run(worker, ["train", "--output", original, "--cache", cache, *smoke_args], root, output / "original.log")
    _run(worker, ["train-fast", "--output", diverse, "--cache", cache,
                  "--initial-gate", original / "exports/original-fast/weights.safetensors", *smoke_args],
         root, output / "diverse-gate.log")
    _run(worker, ["evaluate", "--output", output / "evaluation", "--cache", cache,
                  "--lora-weights", original / "exports/lora/weights.safetensors",
                  "--gate-weights", diverse / "export/weights.safetensors", *smoke_args],
         root, output / "evaluation.log")
    from .common import read_json, write_json
    summary = read_json(output / "evaluation/SUMMARY.json")
    write_json(output / "SUMMARY.json", dict(summary, smoke=smoke))
    return output
