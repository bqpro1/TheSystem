"""Two frozen-backbone training stages with verified, resumable checkpoints.

Only adapter/gate tensors are saved. A checkpoint is usable only after its SHA-256
commit record is written. Test data is never passed to either training function.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import signal
import threading
import uuid

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

CHECKPOINT_ERRORS = (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError)


@dataclass(frozen=True)
class Settings:
    lora_epochs: int = 3
    fast_epochs: int = 10
    micro_batch_size: int = 1
    accumulation: int = 16
    validation_batch_size: int = 2
    fast_batch_size: int = 256
    lora_lr: float = 2e-4
    fast_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    checkpoint_every: int = 25
    retain_checkpoints: int = 3
    lora_patience: int = 2
    fast_patience: int = 3
    seed: int = 42

    def validate(self):
        integers = (self.lora_epochs, self.fast_epochs, self.micro_batch_size,
                    self.accumulation, self.validation_batch_size, self.fast_batch_size,
                    self.checkpoint_every, self.lora_patience, self.fast_patience)
        if any(value < 1 for value in integers) or self.retain_checkpoints < 2:
            raise ValueError("Training counts must be positive; retain at least two checkpoints")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if any(not math.isfinite(value) or value <= 0 for value in (self.lora_lr, self.fast_lr)):
            raise ValueError("Learning rates must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def cpu_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def weights_sha(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temporary, "wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    _atomic(path, lambda handle: handle.write(json.dumps(value, indent=2, ensure_ascii=False).encode()))


def _atomic_torch(path, value):
    _atomic(path, lambda handle: torch.save(value, handle))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume needs the same number of CUDA devices")
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class Checkpoints:
    """Never replace the only good checkpoint with a partly written file."""

    def __init__(self, folder, identity, retain=3):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.key = json_sha(identity)
        self.retain = retain
        identity_file = self.folder / "identity.json"
        if identity_file.exists():
            if json_sha(json.loads(identity_file.read_text())) != self.key:
                raise ValueError(f"Run identity changed in {self.folder}. Restore settings or use a NEW run name.")
        elif any(self.folder.iterdir()):
            raise ValueError(f"Checkpoint folder has files but no identity: {self.folder}. Do not restart over it.")
        else:
            atomic_json(identity_file, identity)

    def _valid(self, commit):
        metadata = json.loads(commit.read_text())
        if metadata.get("identity_key") != self.key:
            raise ValueError("checkpoint identity mismatch")
        path = self.folder / metadata["file"]
        if path.name != metadata["file"] or file_sha(path) != metadata["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("schema") != 1 or saved.get("identity_key") != self.key:
            raise ValueError("checkpoint schema/identity mismatch")
        return saved

    def latest(self, required=False):
        commits = sorted(self.folder.glob("checkpoint-*.commit.json"), reverse=True)
        rejected = []
        for commit in commits:
            try:
                saved = self._valid(commit)
            except CHECKPOINT_ERRORS as exc:
                rejected.append((commit.name, str(exc)))
                continue
            if rejected:
                print(f"WARNING: recovered {commit.name}; skipped invalid newer checkpoint(s): {rejected}. "
                      "Corrupt files were preserved.", flush=True)
            return saved
        artifacts = list(self.folder.glob("checkpoint-*"))
        if required or commits or artifacts or (self.folder / "status.json").exists():
            raise ValueError(f"No valid committed checkpoint in {self.folder}; refusing a silent restart. {rejected}")
        return None

    def save(self, payload):
        existing = list(self.folder.glob("checkpoint-*"))
        numbers = [int(path.name.split("-")[1].split(".")[0]) for path in existing
                   if path.name.split("-")[1].split(".")[0].isdigit()]
        sequence = max(numbers, default=-1) + 1
        stem = f"checkpoint-{sequence:07d}"
        path = self.folder / (stem + ".pt")
        _atomic_torch(path, dict(payload, schema=1, identity_key=self.key))
        atomic_json(self.folder / (stem + ".commit.json"),
                    {"identity_key": self.key, "file": path.name, "sha256": file_sha(path),
                     "epoch": payload["state"]["epoch"], "updates": payload["state"]["updates"],
                     "examples_seen": payload["state"]["examples_seen"],
                     "cursor": payload["state"]["cursor"],
                     "partial_accumulation_examples": payload["state"]["micro_in_window"],
                     "completed": payload["state"]["completed"],
                     "best_epoch": payload["state"]["best_epoch"]})
        # Retention happens only after the new transaction is committed. Corrupt or
        # incomplete artifacts remain for diagnosis, never silently get overwritten.
        valid = []
        for commit in sorted(self.folder.glob("checkpoint-*.commit.json"), reverse=True):
            try:
                self._valid(commit)
                valid.append(commit)
            except CHECKPOINT_ERRORS:
                pass
        for commit in valid[self.retain:]:
            metadata = json.loads(commit.read_text())
            commit.unlink()
            (self.folder / metadata["file"]).unlink()
        atomic_json(self.folder / "status.json", {
            key: value for key, value in payload["state"].items() if key != "best_weights"
        })


def resume_overview(run_dir):
    """Cheap pre-download check. Training verifies hashes before restoring."""
    output = {}
    for stage in ("stage1-lora", "stage2-fast"):
        folder = Path(run_dir) / stage
        commits = sorted(folder.glob("checkpoint-*.commit.json"), reverse=True)
        if not commits:
            status = "incomplete files found" if list(folder.glob("checkpoint-*")) else "not started"
            output[stage] = {"status": status, "folder": str(folder)}
        else:
            try:
                metadata = json.loads(commits[0].read_text())
                output[stage] = {
                    key: metadata.get(key) for key in ("epoch", "updates", "examples_seen", "cursor",
                                                       "partial_accumulation_examples", "completed", "best_epoch")
                }
                output[stage].update(status="checkpoint found", folder=str(folder),
                                     checkpoint_file_exists=(folder / metadata["file"]).is_file(),
                                     checksum_status="not yet checked; verified when training resumes")
            except CHECKPOINT_ERRORS as exc:
                output[stage] = {"status": "unreadable latest commit", "error": str(exc), "folder": str(folder)}
        print(f"{stage}: {json.dumps(output[stage], ensure_ascii=False)}", flush=True)
    return output


def _stage_identity(identity, settings, stage, lora_sha=None):
    common_keys = ("weight_decay", "warmup_fraction", "seed")
    keys = common_keys + (("lora_epochs", "micro_batch_size", "accumulation", "lora_lr", "lora_patience")
                          if stage == "lora" else ("fast_epochs", "fast_batch_size", "fast_lr", "fast_patience"))
    result = {"common": identity, "stage": stage,
              "settings": {key: getattr(settings, key) for key in keys},
              "selection": "source_macro_accuracy" if stage == "lora" else "source_macro_balanced_accuracy"}
    if lora_sha is not None:
        result["selected_lora_sha256"] = lora_sha
    return result


def _state():
    return {"epoch": 0, "cursor": 0, "updates": 0, "examples_seen": 0,
            "epoch_loss_sum": 0., "epoch_examples": 0, "micro_in_window": 0,
            "window_size": 0, "best_score": None, "best_loss": None,
            "best_epoch": None, "best_weights": None, "best_weights_sha256": None,
            "bad_epochs": 0, "history": [], "completed": False, "completion_reason": None}


def _save(store, module, optimizer, state, total_updates):
    store.save({"weights": cpu_state(module), "optimizer": optimizer.state_dict(),
                "gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                              for name, p in module.named_parameters()},
                "state": state, "rng": _rng(),
                "scheduler": {"kind": "warmup_cosine", "total_updates": total_updates,
                              "next_update": state["updates"]}})


def _start(store, module, lr, weight_decay, total_updates):
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    saved = store.latest()
    if saved is None:
        state = _state()
        _save(store, module, optimizer, state, total_updates)
        print(f"NEW {store.folder.name}: epoch 1, update 0, 0 examples. Checkpoint folder: {store.folder}", flush=True)
    else:
        if saved["scheduler"]["total_updates"] != total_updates:
            raise ValueError("Training schedule changed; use the original settings or a new run")
        module.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        for name, parameter in module.named_parameters():
            grad = saved["gradients"][name]
            parameter.grad = None if grad is None else grad.to(parameter.device, parameter.dtype)
        state = saved["state"]
        _restore_rng(saved["rng"])
        print(f"RESUME {store.folder.name}: completed epochs={state['epoch']}, "
              f"next epoch position={state['cursor']}, updates={state['updates']}, "
              f"examples seen={state['examples_seen']}, partial accumulation={state['micro_in_window']}, "
              f"completed={state['completed']}. Folder: {store.folder}", flush=True)
    return optimizer, state


def _learning_rate(update, total, peak, warmup_fraction):
    warmup = max(1, math.ceil(total * warmup_fraction))
    if update < warmup:
        return peak * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


@contextmanager
def _graceful_interrupt():
    # Finish the active backward pass before saving its accumulated gradients.
    # A kernel crash/hard kill instead restores the last committed checkpoint.
    flag = {"requested": False}
    can_handle = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if can_handle else None
    if can_handle:
        def handler(signum, frame):
            flag["requested"] = True
        signal.signal(signal.SIGINT, handler)
    try:
        yield flag
    finally:
        if can_handle:
            signal.signal(signal.SIGINT, previous)


def _finish_epoch(state, metrics, module, epochs, patience):
    score, loss = metrics["selection_score"], metrics["loss"]
    if not math.isfinite(score) or not math.isfinite(loss):
        raise FloatingPointError("Validation has non-finite values")
    better = (state["best_score"] is None or score > state["best_score"] or
              (score == state["best_score"] and loss < state["best_loss"]))
    if better:
        weights = cpu_state(module)
        state.update(best_score=score, best_loss=loss, best_epoch=state["epoch"] + 1,
                     best_weights=weights, best_weights_sha256=weights_sha(weights), bad_epochs=0)
    else:
        state["bad_epochs"] += 1
    state["history"].append(dict(metrics, epoch=state["epoch"] + 1,
                                 train_loss=state["epoch_loss_sum"] / state["epoch_examples"],
                                 updates=state["updates"], train_examples=state["epoch_examples"],
                                 selected_best=better))
    state.update(epoch=state["epoch"] + 1, cursor=0, epoch_loss_sum=0., epoch_examples=0)
    if state["bad_epochs"] >= patience:
        state.update(completed=True, completion_reason="early_stopping")
    elif state["epoch"] >= epochs:
        state.update(completed=True, completion_reason="epoch_limit")
    print(f"Validation: source-macro score={score:.4f}; loss={loss:.4f}; "
          f"selected epoch={state['best_epoch']}; completed={state['completed']}", flush=True)


@torch.no_grad()
def validate_lora(model, rows, batch_size=2, interrupt=None):
    model.eval()
    totals, loss_sum = {}, 0.
    if not rows:
        raise ValueError("Validation data is empty")
    for start in tqdm(range(0, len(rows), batch_size), desc="LoRA validation", unit="batch"):
        if interrupt is not None and interrupt["requested"]:
            raise KeyboardInterrupt("Validation paused; the completed training epoch is already saved")
        batch = rows[start:start + batch_size]
        logits = model.decision_logits(batch, route="on")
        target = torch.tensor([row["target_index"] for row in batch], device=logits.device)
        loss = F.cross_entropy(logits.float(), target, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite decision validation loss")
        prediction = logits.argmax(-1).cpu().tolist()
        loss_sum += float(loss)
        for row, pred in zip(batch, prediction):
            source = row["source"]
            correct, count = totals.get(source, (0, 0))
            totals[source] = (correct + int(pred == row["target_index"]), count + 1)
    per_source = {key: correct / count for key, (correct, count) in sorted(totals.items())}
    return {"loss": loss_sum / len(rows), "selection_score": float(np.mean(list(per_source.values()))),
            "accuracy": sum(pair[0] for pair in totals.values()) / len(rows), "per_source": per_source}


def train_lora(model, train_rows, validation_rows, run_dir, settings, identity,
               stop_after_updates=None, stop_after_microbatches=None):
    """Train only LoRA. Optional stop arguments are for CPU resume checks."""
    settings.validate()
    if not train_rows or not validation_rows:
        raise ValueError("Both training and validation data are required")
    model.stage("lora")
    frozen_gate = weights_sha(cpu_state(model.gate))
    store = Checkpoints(Path(run_dir) / "stage1-lora", _stage_identity(identity, settings, "lora"),
                        settings.retain_checkpoints)
    window = settings.micro_batch_size * settings.accumulation
    total = math.ceil(len(train_rows) / window) * settings.lora_epochs
    optimizer, state = _start(store, model.adapters, settings.lora_lr, settings.weight_decay, total)
    processed_microbatches = 0
    with _graceful_interrupt() as interrupt:
        while not state["completed"]:
            order = list(range(len(train_rows)))
            random.Random(settings.seed + state["epoch"]).shuffle(order)
            model.stage("lora")
            bar = tqdm(total=len(order), initial=state["cursor"],
                       desc=f"LoRA epoch {state['epoch'] + 1}/{settings.lora_epochs}", unit="example")
            try:
                while state["cursor"] < len(order):
                    if state["micro_in_window"] == 0:
                        optimizer.zero_grad(set_to_none=True)
                        state["window_size"] = min(window, len(order) - state["cursor"])
                    indices = order[state["cursor"]:state["cursor"] + settings.micro_batch_size]
                    batch = [train_rows[index] for index in indices]
                    logits = model.decision_logits(batch, route="on")
                    targets = torch.tensor([row["target_index"] for row in batch], device=logits.device)
                    loss = F.cross_entropy(logits.float(), targets, reduction="sum")
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite forward loss; no unsafe checkpoint saved")
                    (loss / state["window_size"]).backward()
                    state["cursor"] += len(batch)
                    state["examples_seen"] += len(batch)
                    state["epoch_examples"] += len(batch)
                    state["epoch_loss_sum"] += float(loss.detach())
                    state["micro_in_window"] += len(batch)
                    processed_microbatches += 1
                    bar.update(len(batch))
                    if state["micro_in_window"] == state["window_size"]:
                        lr = _learning_rate(state["updates"], total, settings.lora_lr, settings.warmup_fraction)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        norm = torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.)
                        if not bool(torch.isfinite(norm)):
                            raise FloatingPointError("Non-finite LoRA gradients; last valid checkpoint retained")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state.update(updates=state["updates"] + 1, micro_in_window=0, window_size=0)
                        if state["updates"] % settings.checkpoint_every == 0 or state["cursor"] == len(order):
                            _save(store, model.adapters, optimizer, state, total)
                        bar.set_postfix(update=state["updates"], loss=f"{float(loss.detach()) / len(batch):.3f}", lr=f"{lr:.1g}")
                    stop = ((stop_after_updates is not None and state["updates"] >= stop_after_updates) or
                            (stop_after_microbatches is not None and processed_microbatches >= stop_after_microbatches))
                    if stop or interrupt["requested"]:
                        _save(store, model.adapters, optimizer, state, total)
                        print("Paused at a safe boundary. Rerun the cell to restore weights, optimizer, and partial gradients.", flush=True)
                        return state
            finally:
                bar.close()
            # The end-of-epoch checkpoint precedes validation. An interrupted
            # validation is retried on resume without repeating the training epoch.
            metrics = validate_lora(model, validation_rows, settings.validation_batch_size, interrupt=interrupt)
            _finish_epoch(state, metrics, model.adapters, settings.lora_epochs, settings.lora_patience)
            _save(store, model.adapters, optimizer, state, total)
            if interrupt["requested"] and not state["completed"]:
                return state
    model.adapters.load_state_dict(state["best_weights"])
    if weights_sha(cpu_state(model.gate)) != frozen_gate:
        raise AssertionError("Stage 1 unexpectedly changed FAST")
    return state


@torch.no_grad()
def cache_features(model, rows, cache_dir, identity, batch_size=2, chunk_size=256):
    """Extract frozen-prefix states once; committed chunks survive disconnects."""
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    key = json_sha({"identity": identity, "rows": [(r["id"], r["prompt"], r["mode_target"], r["source"]) for r in rows],
                    "chunk_size": chunk_size, "feature_dtype": "float32"})
    index_file = folder / "identity.json"
    if index_file.exists():
        if json.loads(index_file.read_text())["identity_key"] != key:
            raise ValueError("Feature cache identity changed; use a new cache directory")
    else:
        if any(folder.iterdir()):
            raise ValueError("Feature cache has files but no identity")
        atomic_json(index_file, {"identity_key": key, "rows": len(rows)})
    if not rows or batch_size < 1 or chunk_size < 1:
        raise ValueError("Feature extraction requires nonempty rows and positive batch/chunk sizes")
    model.eval()
    parts = []
    with tqdm(total=len(rows), desc=f"FAST features: {folder.name}", unit="example") as bar:
        for begin in range(0, len(rows), chunk_size):
            batch_rows = rows[begin:begin + chunk_size]
            path = folder / f"features-{begin:07d}.pt"
            commit = folder / f"features-{begin:07d}.commit.json"
            if commit.exists():
                metadata = json.loads(commit.read_text())
                if metadata["identity_key"] != key or file_sha(path) != metadata["sha256"]:
                    raise ValueError(f"Feature cache checksum failed: {path}. Preserve it and use a new cache folder.")
                part = torch.load(path, map_location="cpu", weights_only=True)
                if part["ids"] != [r["id"] for r in batch_rows]:
                    raise ValueError("Feature chunk rows do not match")
                bar.update(len(batch_rows))
            else:
                features = []
                for offset in range(0, len(batch_rows), batch_size):
                    subset = batch_rows[offset:offset + batch_size]
                    values = model.gate_features(subset).detach().float().cpu()
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("Non-finite prefix features")
                    features.append(values)
                    bar.update(len(subset))
                part = {"x": torch.cat(features), "y": torch.tensor([r["mode_target"] for r in batch_rows]).float(),
                        "sources": [r["source"] for r in batch_rows], "ids": [r["id"] for r in batch_rows]}
                # Uncommitted files may be leftovers from a crash. Preserve them.
                if path.exists():
                    path.rename(path.with_name(path.name + ".incomplete-" + uuid.uuid4().hex))
                _atomic_torch(path, part)
                atomic_json(commit, {"identity_key": key, "sha256": file_sha(path)})
            parts.append(part)
    return {"x": torch.cat([part["x"] for part in parts]), "y": torch.cat([part["y"] for part in parts]),
            "sources": sum([part["sources"] for part in parts], []),
            "ids": sum([part["ids"] for part in parts], []), "identity_key": key}


def _fast_metrics(logits, y, sources, threshold):
    prediction = (logits.sigmoid() >= threshold).float()
    per_source = {}
    for source in sorted(set(sources)):
        mask = torch.tensor([value == source for value in sources])
        # Some natural sources contain one class; average over labels present.
        per_source[source] = float(torch.stack([(prediction[mask & (y == label)] == label).float().mean()
                                               for label in y[mask].unique()]).mean())
    return {"loss": float(F.binary_cross_entropy_with_logits(logits, y)),
            "accuracy": float((prediction == y).float().mean()),
            "selection_score": float(np.mean(list(per_source.values()))), "per_source": per_source,
            "false_fast": int(((prediction == 1) & (y == 0)).sum()),
            "missed_fast": int(((prediction == 0) & (y == 1)).sum())}


@torch.no_grad()
def validate_fast(model, features, batch_size=256, interrupt=None):
    model.eval()
    outputs = []
    for batch in tqdm(features["x"].split(batch_size), desc="FAST validation", unit="batch"):
        if interrupt is not None and interrupt["requested"]:
            raise KeyboardInterrupt("FAST validation paused; training progress is already saved")
        outputs.append(model.gate_logits(batch).detach().float().cpu())
    logits = torch.cat(outputs)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Non-finite FAST validation logits")
    return _fast_metrics(logits, features["y"], features["sources"], model.threshold)


def _selected_stage(run_dir, identity, settings, stage, lora_sha=None):
    folder = "stage1-lora" if stage == "lora" else "stage2-fast"
    store = Checkpoints(Path(run_dir) / folder, _stage_identity(identity, settings, stage, lora_sha),
                        settings.retain_checkpoints)
    saved = store.latest(required=True)
    state = saved["state"]
    if not state["completed"] or state["best_weights"] is None:
        raise ValueError(f"Complete {stage} training and validation before continuing")
    if weights_sha(state["best_weights"]) != state["best_weights_sha256"]:
        raise ValueError("Selected weight digest does not match")
    return state


def train_fast(model, training_features, validation_features, run_dir, settings, identity,
               stop_after_updates=None):
    """Freeze selected LoRA; fit only the binary FAST gate on frozen states."""
    settings.validate()
    lora = _selected_stage(run_dir, identity, settings, "lora")
    model.adapters.load_state_dict(lora["best_weights"])
    model.stage("gate")
    stage_identity = _stage_identity(identity, settings, "fast", lora["best_weights_sha256"])
    # Cache identities are checked separately to avoid accidentally mixing splits.
    stage_identity["features"] = [training_features["identity_key"], validation_features["identity_key"]]
    store = Checkpoints(Path(run_dir) / "stage2-fast", stage_identity, settings.retain_checkpoints)
    x, y = training_features["x"], training_features["y"]
    if set(y.tolist()) != {0., 1.} or not len(validation_features["y"]):
        raise ValueError("FAST needs both training classes and nonempty validation")
    total = math.ceil(len(y) / settings.fast_batch_size) * settings.fast_epochs
    optimizer, state = _start(store, model.gate, settings.fast_lr, settings.weight_decay, total)
    with _graceful_interrupt() as interrupt:
        while not state["completed"]:
            order = list(range(len(y)))
            random.Random(settings.seed + 1000 + state["epoch"]).shuffle(order)
            with tqdm(total=len(y), initial=state["cursor"],
                      desc=f"FAST epoch {state['epoch'] + 1}/{settings.fast_epochs}", unit="example") as bar:
                while state["cursor"] < len(order):
                    indices = order[state["cursor"]:state["cursor"] + settings.fast_batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.gate_logits(x[indices])
                    loss = F.binary_cross_entropy_with_logits(logits.float(), y[indices].to(logits.device))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite FAST loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 1.)
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError("Non-finite FAST gradients")
                    lr = _learning_rate(state["updates"], total, settings.fast_lr, settings.warmup_fraction)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    state.update(cursor=state["cursor"] + len(indices), updates=state["updates"] + 1,
                                 examples_seen=state["examples_seen"] + len(indices),
                                 epoch_examples=state["epoch_examples"] + len(indices),
                                 epoch_loss_sum=state["epoch_loss_sum"] + float(loss.detach()) * len(indices))
                    bar.update(len(indices))
                    bar.set_postfix(update=state["updates"], loss=f"{float(loss.detach()):.3f}")
                    if state["updates"] % settings.checkpoint_every == 0 or state["cursor"] == len(order):
                        _save(store, model.gate, optimizer, state, total)
                    if interrupt["requested"] or (stop_after_updates is not None and state["updates"] >= stop_after_updates):
                        _save(store, model.gate, optimizer, state, total)
                        return state
            metrics = validate_fast(model, validation_features, settings.fast_batch_size, interrupt=interrupt)
            _finish_epoch(state, metrics, model.gate, settings.fast_epochs, settings.fast_patience)
            _save(store, model.gate, optimizer, state, total)
            if interrupt["requested"] and not state["completed"]:
                return state
    model.gate.load_state_dict(state["best_weights"])
    if weights_sha(cpu_state(model.adapters)) != lora["best_weights_sha256"]:
        raise AssertionError("FAST training unexpectedly changed LoRA")
    return state


def load_selected(model, run_dir, identity, settings, training_features=None, validation_features=None):
    """Load validation-selected tensors and report exact identities for tests."""
    lora = _selected_stage(run_dir, identity, settings, "lora")
    stage_identity = _stage_identity(identity, settings, "fast", lora["best_weights_sha256"])
    if training_features is not None and validation_features is not None:
        stage_identity["features"] = [training_features["identity_key"], validation_features["identity_key"]]
    else:
        saved_identity = json.loads((Path(run_dir) / "stage2-fast" / "identity.json").read_text())
        # Feature identity is frozen at fitting time. Every other field must match.
        stage_identity["features"] = saved_identity.get("features")
    store = Checkpoints(Path(run_dir) / "stage2-fast", stage_identity, settings.retain_checkpoints)
    fast = store.latest(required=True)["state"]
    if not fast["completed"] or fast["best_weights"] is None:
        raise ValueError("Complete FAST training before comparison tests")
    if weights_sha(fast["best_weights"]) != fast["best_weights_sha256"]:
        raise ValueError("FAST selected weight digest does not match")
    model.adapters.load_state_dict(lora["best_weights"])
    model.gate.load_state_dict(fast["best_weights"])
    model.stage("evaluation")
    details = lambda state: {"selected_epoch": state["best_epoch"], "score": state["best_score"],
                             "loss": state["best_loss"], "completed": state["completed"]}
    return {"lora_sha256": lora["best_weights_sha256"], "fast_sha256": fast["best_weights_sha256"],
            "lora": details(lora), "fast": details(fast)}
