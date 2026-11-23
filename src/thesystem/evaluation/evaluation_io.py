"""Atomic result shards, immutable run identities and honest progress heartbeats."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__import__("os").environ.get("THESYSTEM_RELEASE_ROOT", ".")).expanduser().resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def outside_package(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents or path in ROOT.parents:
        raise ValueError("Outputs and cache must be outside the immutable evaluation package")
    return path


def bind_run(folder, identity):
    folder = outside_package(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "identity.json"
    if target.exists():
        if read_json(target) != identity:
            raise ValueError(f"Run identity changed: {folder}. Preserve it and use a new run name.")
    else:
        if any(folder.iterdir()):
            raise ValueError(f"Folder has files without an identity: {folder}")
        write_json(target, identity)


@contextmanager
def exclusive_run(folder):
    """An OS lock is released even if a kernel is killed; never run two writers."""
    with (folder / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluation is already writing to {folder}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def heartbeat(folder, state):
    stop = threading.Event()
    started = time.monotonic()
    def report():
        payload = dict(state, utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=round(time.monotonic() - started))
        if state.get("last_save"):
            payload["seconds_since_save"] = round(time.time() - state["last_save"])
        write_json(folder / "heartbeat.json", payload)
        print("HEARTBEAT", json.dumps(payload), flush=True)
    def watch():
        while not stop.wait(30):
            report()
    report()
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def evaluate_stage(rows, folder, identity, evaluate_batch, state, batch_size=1, shard_size=32):
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest(dict(identity=identity, rows=rows, batch_size=batch_size, shard_size=shard_size))
    expected = [f"part-{i:06d}.json" for i in range(0, len(rows), shard_size)]
    existing = sorted(p.name for p in folder.glob("part-*.json"))
    if existing != expected[:len(existing)]:
        raise ValueError(f"Missing or unexpected shards: {folder}")
    complete_path = folder / "COMPLETE.json"
    if complete_path.exists() and (existing != expected or read_json(complete_path)["signature"] != signature):
        raise ValueError(f"Completed stage changed: {folder}")
    state.update(stage=folder.name, completed=0, processed=0, planned=len(rows))
    results = []
    for start in range(0, len(rows), shard_size):
        subset, path = rows[start:start + shard_size], folder / f"part-{start:06d}.json"
        if path.exists():
            saved = read_json(path)
        else:
            values = []
            for offset in range(0, len(subset), batch_size):
                values.extend(evaluate_batch(subset[offset:offset + batch_size]))
                state["processed"] = start + len(values)
            saved = dict(signature=signature, rows=values, sha256=digest(values))
            write_json(path, saved)
        if (saved["signature"] != signature or saved["sha256"] != digest(saved["rows"])
                or [r["id"] for r in saved["rows"]] != [r["id"] for r in subset]):
            raise ValueError(f"Invalid saved evaluation shard: {path}")
        results.extend(saved["rows"])
        state.update(completed=len(results), processed=len(results), last_save=path.stat().st_mtime)
        print(f"{folder.name}: {len(results)}/{len(rows)} saved", flush=True)
    write_json(complete_path, dict(signature=signature, planned=len(rows), completed=len(results), complete=True))
    return results


def read_stage(folder):
    complete = read_json(folder / "COMPLETE.json")
    rows = []
    for path in sorted(folder.glob("part-*.json")):
        saved = read_json(path)
        if saved["signature"] != complete["signature"] or saved["sha256"] != digest(saved["rows"]):
            raise ValueError(f"Invalid results: {path}")
        rows.extend(saved["rows"])
    if len(rows) != complete["planned"] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f"Incomplete or duplicate results: {folder}")
    return rows
