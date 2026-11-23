"""Resolve immutable, checksum-verified model and dataset resources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Literal

Variant = Literal["probe", "lora"]


def read_release_manifest(root: str | Path | None = None) -> dict:
    """Read a local release manifest, or the manifest pinned in this wheel."""
    path = Path(root) / "RELEASE-MANIFEST.json" if root is not None else None
    if path is None or not path.is_file():
        path = Path(__file__).with_name("release_manifest.json")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or set(manifest.get("models", {})) != {"probe", "lora"}:
        raise ValueError(f"Unsupported release manifest: {path}")
    base = manifest.get("base_model", {})
    if base != {"repo_id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}:
        raise ValueError("Release manifest names a different backbone")
    return manifest


def _file_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError(f"Invalid resource path: {name}")
    return root.joinpath(*relative.parts)


def verify_files(root: str | Path, files: dict) -> None:
    """Reject missing or changed resources before loading any weights."""
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty resource inventory is required")
    for name, record in files.items():
        path = _file_path(Path(root), name)
        if not isinstance(record, dict) or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError(f"Invalid checksum record: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing release resource: {path}")
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Release resource size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Release resource checksum mismatch: {path}")


def _download(entry: dict, repo_type: str, revision: str | None = None) -> Path:
    commit = entry.get("revision", "") if revision is None else revision
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An immutable, full 40-character Hugging Face commit is required")
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(entry["repo_id"], repo_type=repo_type, revision=commit,
                                  allow_patterns=list(entry["files"]),
                                  token=os.environ.get("HF_TOKEN") or False))
    verify_files(root, entry["files"])
    return root


def resolve_model_resources(variant: Variant = "probe", revision: str | None = None,
                            root: str | Path | None = None) -> Path:
    """Use a verified local model snapshot or download its pinned Hub revision."""
    if variant not in ("probe", "lora"):
        raise ValueError("variant must be 'probe' or 'lora'")
    manifest = read_release_manifest(root)
    entry = manifest["models"][variant]
    if root is None:
        return _download(entry, "model", revision)
    if revision is not None and revision != entry.get("revision"):
        raise ValueError("A revision override cannot change an explicit local snapshot")
    root = Path(root).resolve()
    verify_files(root, entry["files"])
    return root


def _combined_files(manifest: dict) -> dict:
    files = {}
    for entry in [*manifest["models"].values(), manifest["dataset"]]:
        for name, record in entry["files"].items():
            _file_path(Path("."), name)
            if name in files and files[name] != record:
                raise ValueError(f"Conflicting model/dataset resource: {name}")
            files[name] = record
    return files


def _copy_resource(source: Path, target: Path) -> None:
    """An interrupted or concurrent copy must never expose a partial resource."""
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def resolve_release_resources(root: str | Path | None = None) -> Path:
    """Resolve both models and exact datasets into one verified resource directory."""
    manifest = read_release_manifest(root)
    files = _combined_files(manifest)
    if root is not None:
        root = Path(root).resolve()
        verify_files(root, files)
        return root
    from huggingface_hub.constants import HF_HUB_CACHE
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    identity = hashlib.sha256(serialized.encode()).hexdigest()
    combined = Path(HF_HUB_CACHE) / "thesystem-releases" / identity
    marker = combined / "RELEASE-MANIFEST.json"
    if marker.exists():
        if marker.read_text() != serialized:
            raise ValueError(f"Conflicting cached release manifest: {marker}")
        verify_files(combined, files)
        return combined
    entries = [(entry, "model") for entry in manifest["models"].values()]
    entries.append((manifest["dataset"], "dataset"))
    for entry, kind in entries:
        snapshot = _download(entry, kind)
        for name, record in entry["files"].items():
            target = _file_path(combined, name)
            if target.exists():
                verify_files(combined, {name: record})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copies keep editable training outputs from modifying the Hub's shared cache.
            _copy_resource(_file_path(snapshot, name), target)
    verify_files(combined, files)
    with tempfile.NamedTemporaryFile(mode="w", dir=combined, prefix=".manifest.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(serialized)
            stream.close()
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
    return combined
