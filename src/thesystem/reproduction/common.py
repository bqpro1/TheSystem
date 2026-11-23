"""Portable paths, immutable run identities and visible notebook subprocesses."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(os.environ.get("THESYSTEM_RELEASE_ROOT", ".")).expanduser().resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def output_path(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents or path in ROOT.parents:
        raise ValueError("Choose an output folder outside the immutable release package.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def implementation_fingerprint():
    package = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(package)): sha(path)
            for path in sorted(package.rglob("*.py"))}


def bind_run(folder, identity):
    folder = output_path(folder)
    identity = dict(identity, release_manifest_sha256=sha(ROOT / "RELEASE-MANIFEST.json"),
                    implementation_sha256=implementation_fingerprint())
    path = folder / "run-identity.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError(f"Run identity changed: {path}. Preserve it and choose a new output folder.")
    if not path.exists():
        if any(folder.iterdir()):
            raise ValueError(f"Output has files but no run identity: {folder}")
        write_json(path, identity)
    return identity


def load_data(name):
    folder = ROOT / "data" / name
    manifest = read_json(folder / ("manifest.json" if name == "original" else "data-manifest.json"))
    datasets = {}
    for key, record in manifest["files"].items():
        relative = record.get("path", key) if isinstance(record, dict) else key
        expected = record["sha256"] if isinstance(record, dict) else record
        path = folder / relative
        if sha(path) != expected:
            raise ValueError(f"Frozen data changed: {path}")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if isinstance(record, dict) and len(rows) != record["rows"]:
            raise ValueError(f"Wrong row count: {path}")
        datasets[Path(relative).stem.replace("-", "_")] = rows
    return datasets, manifest


@contextmanager
def heartbeat(folder, label):
    """Activity is not proof of progress: report committed files and their age."""
    stop, started = threading.Event(), time.monotonic()
    def report():
        files = [p for p in folder.rglob("*") if p.is_file() and
                 (p.suffix == ".ptc" or p.name.endswith(".commit.json") or p.suffix == ".jsonl")]
        latest = max((p.stat().st_mtime for p in files), default=None)
        value = {"stage": label, "utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": round(time.monotonic() - started), "committed_files": len(files),
                 "seconds_since_save": round(time.time() - latest) if latest else None}
        write_json(folder / "heartbeat.json", value)
        print("HEARTBEAT", json.dumps(value), flush=True)
    def watch():
        while not stop.wait(30):
            report()
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        report()
        yield
    finally:
        stop.set()
        thread.join()
