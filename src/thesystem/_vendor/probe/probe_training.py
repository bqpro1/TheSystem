"""Checksummed frozen-feature caches and resumable answer-head training."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Callable
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .progress import tqdm
from .probe_model import HEAD_ARMS, head_spec, initialize_head


@dataclass(frozen=True)
class ProbeSettings:
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    checkpoint_every: int = 25
    seed: int = 42

    def validate(self) -> None:
        for name in ("batch_size", "max_epochs", "patience", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256(_json_bytes([str(tensor.dtype), list(tensor.shape)]))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_digest(module: nn.Module) -> str:
    return _digest({name: _tensor_digest(value) for name, value in module.state_dict().items()})


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value) + b"\n")


def _write_checked(path: Path, payload: dict) -> None:
    # One atomic file contains its checksum and payload, avoiding a two-file commit gap.
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    header = {"format": "thesystem-probe-checked-v1", "sha256": hashlib.sha256(raw).hexdigest()}
    _atomic_bytes(path, _json_bytes(header) + b"\n" + raw)


def _read_checked(path: Path) -> dict:
    try:
        header_bytes, raw = path.read_bytes().split(b"\n", 1)
        header = json.loads(header_bytes)
        if (header.get("format") != "thesystem-probe-checked-v1" or
                header.get("sha256") != hashlib.sha256(raw).hexdigest()):
            raise ValueError("checksum or format mismatch")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")
        return payload
    except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
        raise ValueError(f"Invalid saved artifact {path}: {error}. It was not replaced.") from error


def _open_folder(folder: Path, identity: dict) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "INVALID.json").exists():
        raise ValueError(f"A previous invariant failure invalidated {folder}; preserve it and use a new folder")
    path = folder / "identity.json"
    key = _digest(identity)
    if path.exists():
        if _digest(json.loads(path.read_bytes())) != key:
            raise ValueError(f"Identity changed in {folder}; restore inputs/settings or use a new folder")
    elif any(folder.iterdir()):
        raise ValueError(f"Files exist without an identity in {folder}; refusing to restart over them")
    else:
        _write_json(path, identity)
    pending = list(folder.glob("*.tmp-*"))
    if pending:
        print(f"{folder.name}: preserving {len(pending)} uncommitted temporary files; using only atomic completed files.", flush=True)
    return key


def _code_identity(model: object) -> dict[str, str]:
    paths = {Path(__file__).resolve()}
    for item in (model, model.reference):
        filename = inspect.getsourcefile(type(item))
        if filename:
            paths.add(Path(filename).resolve())
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


@contextmanager
def _frozen_reference(model: object, folder: Path):
    reference = model.reference
    parameters = tuple(reference.named_parameters())
    if any(p.requires_grad or p.grad is not None for _, p in parameters):
        raise ValueError("Qwen and FAST must be frozen with no gradients")
    stamp = tuple((name, id(p), p._version) for name, p in parameters)
    additions = {name: _state_digest(getattr(reference, name)) for name in ("gate",)
                 if hasattr(reference, name)}
    try:
        yield
    finally:
        current = tuple(reference.named_parameters())
        if (stamp != tuple((name, id(p), p._version) for name, p in current) or
                any(p.requires_grad or p.grad is not None for _, p in current)):
            _write_json(folder / "INVALID.json", {"reason": "frozen reference parameters or gradients changed"})
            raise RuntimeError("Frozen Qwen/FAST parameters or gradients changed")
        if any(_state_digest(getattr(reference, name)) != digest for name, digest in additions.items()):
            _write_json(folder / "INVALID.json", {"reason": "frozen FAST tensor values changed"})
            raise RuntimeError("Frozen FAST tensor values changed")


def _row_metadata(rows: list[dict], split: str) -> dict:
    if split not in {"train", "validation"}:
        raise ValueError("Feature caches are only for train or validation, never evaluation/test")
    if not rows:
        raise ValueError("A feature cache requires nonempty rows")
    ids, hashes, counts, targets, families = [], [], [], [], []
    for row in rows:
        if row.get("split", split) != split:
            raise ValueError(f"Row {row.get('id')} belongs to a different split")
        count, target = len(row.get("options", [])), row.get("target_index")
        if (not 2 <= count <= 18 or isinstance(target, bool) or
                not isinstance(target, int) or not 0 <= target < count):
            raise ValueError(f"Invalid candidate count or target for {row.get('id')}")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each feature row needs a nonempty ID")
        if not isinstance(row.get("source"), str) or not row["source"]:
            raise ValueError("Each feature row needs a source family")
        ids.append(row["id"])
        hashes.append(_digest(row))
        counts.append(count)
        targets.append(target)
        families.append(row["source"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate feature row IDs")
    return dict(ids=ids, row_hashes=hashes, counts=counts, targets=targets, families=families)


def _validate_shard(payload: dict, key: str, start: int, stop: int,
                    metadata: dict, widths: dict[str, int]) -> None:
    if (payload.get("identity_key"), payload.get("start"), payload.get("stop")) != (key, start, stop):
        raise ValueError("Feature shard identity or row boundaries changed")
    for name in ("ids", "row_hashes", "counts", "targets", "families"):
        if payload.get(name) != metadata[name][start:stop]:
            raise ValueError(f"Feature shard {name} does not match the requested rows")
    if set(payload.get("x", {})) != set(widths):
        raise ValueError("Feature shard has the wrong heads")
    for arm, width in widths.items():
        tensor = payload["x"][arm]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
                tensor.shape != (stop - start, width) or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"Invalid cached features for {arm}")
    if not math.isfinite(payload.get("extraction_seconds", -1)) or payload["extraction_seconds"] < 0:
        raise ValueError("Invalid extraction duration")


def extract_features(model: object, rows: list[dict], folder: str | Path, identity: dict,
                     batch_size: int = 4, shard_size: int = 64) -> dict:
    """Cache only frozen train/validation features; committed shards resume exactly.

    identity must include split='train' or 'validation' and the run's pinned inputs.
    Returns one shared CPU x["final"], counts, targets, families and cache identity.
    """
    if batch_size < 1 or shard_size < 1:
        raise ValueError("Feature batch/shard sizes must be positive")
    metadata = _row_metadata(rows, identity.get("split"))
    widths = {"final": model.heads[HEAD_ARMS[0]].in_features}
    if any(head.in_features != widths["final"] for head in model.heads.values()):
        raise ValueError("All heads must read the same final feature width")
    full_identity = dict(schema=1, caller=identity, split=identity["split"], rows=metadata,
                         widths=widths, batch_size=batch_size, shard_size=shard_size,
                         storage_dtype="float32", backbone_adapters=0,
                         feature_spec={"final": "decoder-final-normalized-final-prompt"},
                         code=_code_identity(model))
    folder = Path(folder)
    key = _open_folder(folder, full_identity)
    expected_names = {f"features-{start:08d}.ptc" for start in range(0, len(rows), shard_size)}
    existing_names = {path.name for path in folder.glob("features-*.ptc")}
    if existing_names - expected_names:
        raise ValueError("Unexpected feature shards; refusing to reuse a mixed cache")
    ordered_names = sorted(expected_names)
    if existing_names != set(ordered_names[:len(existing_names)]):
        raise ValueError("Feature cache has a missing committed shard")
    if (folder / "summary.json").exists():
        summary = json.loads((folder / "summary.json").read_bytes())
        if (summary.get("identity_key") != key or not summary.get("complete") or
                summary.get("rows") != len(rows) or existing_names != expected_names):
            raise ValueError("Completed feature cache is incomplete or has changed")
    print(f"Features {identity['split']}: reuse {len(existing_names)}/{len(expected_names)} saved shards", flush=True)
    shards = []
    with _frozen_reference(model, folder):
        model.reference.eval()
        for start in tqdm(range(0, len(rows), shard_size), desc=f"Features {identity['split']}", unit="shard"):
            stop = min(start + shard_size, len(rows))
            path = folder / f"features-{start:08d}.ptc"
            if path.exists():
                payload = _read_checked(path)
            else:
                began = time.monotonic()
                blocks = {arm: [] for arm in widths}
                for offset in range(start, stop, batch_size):
                    with torch.no_grad():
                        batch = model.features(rows[offset:min(offset + batch_size, stop)])
                    if set(batch) != set(widths):
                        raise ValueError("Feature extractor returned the wrong heads")
                    for arm in widths:
                        blocks[arm].append(batch[arm].detach().float().cpu().clone())
                payload = dict(identity_key=key, start=start, stop=stop,
                               x={arm: torch.cat(parts) for arm, parts in blocks.items()},
                               extraction_seconds=time.monotonic() - began,
                               **{name: values[start:stop] for name, values in metadata.items()})
                _validate_shard(payload, key, start, stop, metadata, widths)
                _write_checked(path, payload)
            _validate_shard(payload, key, start, stop, metadata, widths)
            shards.append(payload)
    result = dict(metadata, x={arm: torch.cat([part["x"][arm] for part in shards]) for arm in widths},
                  identity={"key": key, "split": identity["split"], "rows": len(rows)},
                  extraction_seconds=sum(part["extraction_seconds"] for part in shards))
    result["counts"] = torch.tensor(result["counts"], dtype=torch.int64)
    result["targets"] = torch.tensor(result["targets"], dtype=torch.int64)
    _write_json(folder / "summary.json", {"identity_key": key, "complete": True, "rows": len(rows),
                                          "extraction_seconds": result["extraction_seconds"]})
    return result


def masked_logits(head: nn.Module, features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Exclude absent choices from training, prediction and probability normalization."""
    if (counts.ndim != 1 or counts.shape[0] != features.shape[0] or
            bool(((counts < 2) | (counts > head.out_features)).any())):
        raise ValueError("Candidate counts do not fit this answer head")
    logits = head(features.float())
    valid = torch.arange(head.out_features, device=logits.device)[None, :] < counts.to(logits.device)[:, None]
    return logits.masked_fill(~valid, -torch.inf)


def _feature_identity(features: dict, arm: str, split: str, width: int) -> dict:
    if features.get("identity", {}).get("split") != split:
        raise ValueError(f"Expected {split} features")
    x, counts, targets = features["x"]["final"], features["counts"], features["targets"]
    n = len(features["ids"])
    if (n == 0 or len(set(features["ids"])) != n or x.shape != (n, width) or
            x.device.type != "cpu" or x.dtype != torch.float32 or x.requires_grad or
            not bool(torch.isfinite(x).all()) or counts.shape != (n,) or targets.shape != (n,) or
            counts.dtype != torch.int64 or targets.dtype != torch.int64 or
            counts.device.type != "cpu" or targets.device.type != "cpu" or
            bool(((counts < 2) | (counts > 18) | (targets < 0) | (targets >= counts)).any()) or
            len(features["families"]) != n or len(features["row_hashes"]) != n):
        raise ValueError("Invalid cached training/validation feature data")
    return dict(cache=features["identity"], x=_tensor_digest(x), counts=_tensor_digest(counts),
                targets=_tensor_digest(targets), ids=features["ids"], families=features["families"],
                row_hashes=features["row_hashes"])


@torch.no_grad()
def evaluate_probe(head: nn.Module, features: dict, arm: str, batch_size: int = 256) -> dict:
    device = next(head.parameters()).device
    head.eval()
    losses, correct = [], []
    for start in range(0, len(features["ids"]), batch_size):
        end = min(start + batch_size, len(features["ids"]))
        logits = masked_logits(head, features["x"]["final"][start:end].to(device),
                               features["counts"][start:end].to(device))
        target = features["targets"][start:end].to(device)
        loss = F.cross_entropy(logits, target, reduction="none")
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Nonfinite probe validation loss")
        losses.extend(loss.cpu().tolist())
        correct.extend(logits.argmax(-1).eq(target).cpu().tolist())
    families = sorted(set(features["families"]))
    per_family = {family: sum(int(c) for c, f in zip(correct, features["families"]) if f == family) /
                  features["families"].count(family) for family in families}
    return {"macro_accuracy": sum(per_family.values()) / len(per_family),
            "accuracy": sum(correct) / len(correct), "nll": sum(losses) / len(losses),
            "per_family": per_family}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def _defer_interrupt():
    pending = {"requested": False}
    main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if main else None
    if main:
        def request_pause(signum, frame):
            pending["requested"] = True
        signal.signal(signal.SIGINT, request_pause)
    try:
        yield pending
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)


def _save_training(folder: Path, key: str, head: nn.Module, optimizer: torch.optim.Optimizer,
                   generator: torch.Generator, state: dict) -> None:
    existing = sorted(folder.glob("checkpoint-*.ptc"))
    sequence = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
    path = folder / f"checkpoint-{sequence:08d}.ptc"
    _write_checked(path, dict(identity_key=key, schema=1, weights=_cpu_state(head),
                             optimizer=optimizer.state_dict(), rng=_rng_state(),
                             generator=generator.get_state(), state=state))
    _write_json(folder / "status.json", {"identity_key": key, "checkpoint": path.name,
                                          "completed": state["completed"], "updates": state["updates"]})
    for old in existing[:-2]:
        old.unlink()
    _write_json(folder / "curves.json", state["curves"])


def _validate_progress(state: dict, n: int, settings: ProbeSettings) -> None:
    order = state.get("order")
    if (not 0 <= state.get("epoch", -1) <= settings.max_epochs or
            not 0 <= state.get("cursor", -1) <= n or
            len(state.get("curves", [])) != state["epoch"] or
            not isinstance(order, torch.Tensor) or order.dtype != torch.int64 or
            order.shape != (n,) or not torch.equal(order.sort().values, torch.arange(n)) or
            state.get("updates", -1) < 0 or state.get("examples_seen", -1) < 0):
        raise ValueError("Invalid checkpoint training position")


def train_probe(model: object, arm: str, train_features: dict, validation_features: dict,
                folder: str | Path, identity: dict, settings: ProbeSettings | None = None,
                *, should_stop: Callable[[], bool] | None = None,
                stop_after_updates: int | None = None) -> dict:
    """Fit one head and restore its selected weights. SIGINT pauses after an update.

    Only atomic optimizer-update boundaries are saved; an interrupted validation is
    repeated from its completed training epoch. A hard loss resumes the last save.
    """
    settings = settings or ProbeSettings()
    settings.validate()
    if arm not in HEAD_ARMS:
        raise ValueError("Unknown probe arm")
    if stop_after_updates is not None and (not isinstance(stop_after_updates, int) or stop_after_updates < 1):
        raise ValueError("stop_after_updates must be a positive integer")
    head = model.heads[arm]
    if head.out_features != 18 or any(p.dtype != torch.float32 for p in head.parameters()):
        raise ValueError("Answer head must be float32 with 18 outputs")
    train_id = _feature_identity(train_features, arm, "train", head.in_features)
    validation_id = _feature_identity(validation_features, arm, "validation", head.in_features)
    if set(train_features["ids"]) & set(validation_features["ids"]):
        raise ValueError("Train and validation feature IDs overlap")
    full_identity = dict(schema=1, caller=identity, arm=arm, settings=asdict(settings),
                         train=train_id, validation=validation_id, code=_code_identity(model),
                         head=head_spec(head))
    folder = Path(folder)
    key = _open_folder(folder, full_identity)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings.learning_rate,
                                 weight_decay=settings.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(settings.seed)
    n = len(train_features["ids"])
    files = sorted(folder.glob("checkpoint-*.ptc"))
    if (folder / "status.json").exists():
        status = json.loads((folder / "status.json").read_bytes())
        if (status.get("identity_key") != key or not files or
                status.get("checkpoint", "") > files[-1].name):
            raise ValueError("Training checkpoint is missing or the saved status identity changed")
    if files:
        saved = _read_checked(files[-1])
        if saved.get("identity_key") != key or saved.get("schema") != 1:
            raise ValueError("Training checkpoint identity changed")
        state = saved["state"]
        _validate_progress(state, n, settings)
        head.load_state_dict(saved["weights"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        _restore_rng(saved["rng"])
        status = "COMPLETE: loading selected weights" if state["completed"] else f"epoch {state['epoch'] + 1}, row {state['cursor']}"
        print(f"RESUME {arm} seed {settings.seed}: {status}; updates {state['updates']}", flush=True)
    else:
        if (folder / "summary.json").exists():
            raise ValueError("Training summary exists without a checkpoint; refusing a restart")
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        with torch.no_grad():
            initialize_head(head)
        print(f"NEW {arm} seed {settings.seed}: {sum(p.numel() for p in head.parameters()):,} trainable parameters", flush=True)
        state = dict(epoch=0, cursor=0, order=torch.randperm(n, generator=generator), updates=0,
                     examples_seen=0, epoch_loss_sum=0., epochs_without_improvement=0,
                     best_epoch=None, best_metrics=None, best_weights=None, curves=[], completed=False,
                     training_seconds=0., peak_memory_bytes=0)
    device = next(head.parameters()).device
    for name, module in model.heads.items():
        module.requires_grad_(name == arm)
        module.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _frozen_reference(model, folder), _defer_interrupt() as pending:
        if not files:
            _save_training(folder, key, head, optimizer, generator, state)
        began, previous_seconds = time.monotonic(), state["training_seconds"]

        def save() -> None:
            state["training_seconds"] = previous_seconds + time.monotonic() - began
            if device.type == "cuda":
                state["peak_memory_bytes"] = max(state["peak_memory_bytes"], torch.cuda.max_memory_allocated(device))
            _save_training(folder, key, head, optimizer, generator, state)

        def pause_if_requested() -> None:
            if (pending["requested"] or (should_stop is not None and should_stop()) or
                    (stop_after_updates is not None and state["updates"] >= stop_after_updates)):
                save()
                raise KeyboardInterrupt(f"{arm} paused at a saved optimizer boundary; rerun to resume")

        try:
            while not state["completed"]:
                head.train()
                pause_if_requested()
                for start in tqdm(range(state["cursor"], n, settings.batch_size),
                                  desc=f"{arm} epoch {state['epoch'] + 1}", unit="batch"):
                    end = min(start + settings.batch_size, n)
                    indices = state["order"][start:end]
                    optimizer.zero_grad(set_to_none=True)
                    logits = masked_logits(head, train_features["x"]["final"][indices].to(device),
                                           train_features["counts"][indices].to(device))
                    loss = F.cross_entropy(logits, train_features["targets"][indices].to(device))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite training loss; last safe checkpoint retained")
                    loss.backward()
                    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                        raise FloatingPointError("Nonfinite training gradients; last safe checkpoint retained")
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
                        raise FloatingPointError("Nonfinite updated head; last safe checkpoint retained")
                    optimizer.zero_grad(set_to_none=True)
                    state["cursor"] = end
                    state["updates"] += 1
                    state["examples_seen"] += end - start
                    state["epoch_loss_sum"] += float(loss.detach()) * (end - start)
                    if state["updates"] % settings.checkpoint_every == 0 or end == n:
                        save()
                    pause_if_requested()
                metrics = evaluate_probe(head, validation_features, arm, settings.batch_size)
                epoch = state["epoch"] + 1
                state["curves"].append(dict(epoch=epoch, train_nll=state["epoch_loss_sum"] / n,
                                            validation=metrics, updates=state["updates"]))
                best = state["best_metrics"]
                better = best is None or (metrics["macro_accuracy"], -metrics["nll"]) > (best["macro_accuracy"], -best["nll"])
                if better:
                    state.update(best_epoch=epoch, best_metrics=metrics, best_weights=_cpu_state(head),
                                 epochs_without_improvement=0)
                else:
                    state["epochs_without_improvement"] += 1
                print(f"{arm} seed {settings.seed} epoch {epoch}: validation {metrics['macro_accuracy']:.2%}; best epoch {state['best_epoch']}", flush=True)
                state["epoch"] = epoch
                state["completed"] = epoch >= settings.max_epochs or state["epochs_without_improvement"] >= settings.patience
                state["cursor"], state["epoch_loss_sum"] = 0, 0.
                if not state["completed"]:
                    state["order"] = torch.randperm(n, generator=generator)
                save()
            head.load_state_dict(state["best_weights"], strict=True)
        finally:
            optimizer.zero_grad(set_to_none=True)
            for module in model.heads.values():
                module.requires_grad_(False)
    summary = dict(arm=arm, seed=settings.seed, architecture=head_spec(head), completed=True, identity_key=key, selected_epoch=state["best_epoch"],
                   selected_metrics=state["best_metrics"], selected_weights_sha256=_state_digest(head),
                   epochs_completed=state["epoch"], updates=state["updates"],
                   examples_seen=state["examples_seen"], training_seconds=state["training_seconds"],
                   peak_memory_bytes=state["peak_memory_bytes"],
                   feature_extraction_seconds=train_features.get("extraction_seconds", 0.) + validation_features.get("extraction_seconds", 0.),
                   curves=state["curves"], settings=asdict(settings))
    _write_json(folder / "summary.json", summary)
    return summary
