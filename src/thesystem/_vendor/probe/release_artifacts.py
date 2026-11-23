"""Small, checksummed inference exports with the unchanged diverse-42 gate."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save

from .probe_training import _atomic_bytes, _cpu_state, _state_digest, _write_json
from .release_data import MODEL_ID, MODEL_REVISION, sha256

GATE_ARTIFACT_SHA = "20357ba838cc66e4c899ed2cb2b77511d85f1e341ecf9da3ae5a9d3ad2516397"
GATE_TENSOR_SHA = "ae23854dbfdb9502bec475b11661f6f1f7e779a0e6d5f9208bff7e27e6d7fba2"


def gate_digest(weights):
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.shape).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prepare_gate(source, folder):
    if sha256(source) != GATE_ARTIFACT_SHA:
        raise ValueError("The exact diverse-42 source export is missing or changed")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(source, map_location="cpu", weights_only=True)
    weights, metadata = payload["weights"], payload["metadata"]
    expected = {"candidate_name": "diverse-42", "seed": 42, "selected_epoch": 9,
                "model": MODEL_ID, "revision": MODEL_REVISION, "gate_after": 16,
                "hidden_size": 4096, "threshold": 0.5, "selected_gate_sha256": GATE_TENSOR_SHA,
                "promotion_authorized": False}
    if any(metadata.get(k) != value for k, value in expected.items()):
        raise ValueError("Gate source identity changed")
    validate_gate(weights)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    raw = save(weights)
    path = folder / "fast.safetensors"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Refusing to replace a different gate artifact")
    _atomic_bytes(path, raw)
    record = {**expected, "source_artifact_sha256": GATE_ARTIFACT_SHA,
              "safetensors_sha256": sha256(path), "status": "experimental candidate",
              "known_test_writing_misroute_rate": 0.0215, "required_maximum": 0.01,
              "all_acceptance_checks_passed": False}
    _write_json(folder / "fast.json", record)
    return record


def validate_gate(weights):
    if set(weights) != {"weight", "bias"}:
        raise ValueError("Unexpected gate tensors")
    for key, value in weights.items():
        if (tuple(value.shape) != ((1, 4096) if key == "weight" else (1,))
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Invalid gate tensor shape, dtype or values")
    if gate_digest(weights) != GATE_TENSOR_SHA:
        raise ValueError("Diverse-42 tensor checksum mismatch")


def load_gate(folder, reference):
    folder = Path(folder)
    metadata = json.loads((folder / "fast.json").read_text())
    if (metadata["selected_gate_sha256"] != GATE_TENSOR_SHA
            or metadata["safetensors_sha256"] != sha256(folder / "fast.safetensors")):
        raise ValueError("Gate file identity mismatch")
    weights = load_file(str(folder / "fast.safetensors"))
    validate_gate(weights)
    reference.gate.load_state_dict(weights, strict=True)
    reference.stage("evaluation")
    return metadata


def export_head(model, arm, folder, identity, training, summary):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    validate_gate(_cpu_state(model.reference.gate))
    tensors = {f"fast.{k}": v for k, v in _cpu_state(model.reference.gate).items()}
    tensors.update({f"answer.{k}": v for k, v in _cpu_state(model.heads[arm]).items()})
    raw = save(tensors)
    path = folder / "heads.safetensors"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Export already exists with different weights")
    _atomic_bytes(path, raw)
    config = {"schema": 1, "model": MODEL_ID, "revision": MODEL_REVISION,
              "architecture": "frozen original Qwen + FAST + answer classifier",
              "backbone_adapters": 0, "arm": arm, "head": model.head_specs()[arm],
              "gate_after": 16, "threshold": 0.5, "max_choices": 18, "max_input_tokens": 2048,
              "system_prompt": "You are a helpful assistant.", "enable_thinking": False,
              "answer_features": "final normalized state at last nonpadding prompt token",
              "fast_features": "RMS-normalized last prompt state after layer 16, epsilon=1e-6",
              "seed": 42, "selected_epoch": training["selected_epoch"],
              "gate_tensor_sha256": GATE_TENSOR_SHA, "sha256": sha256(path),
              "answer_state_digest": _state_digest(model.heads[arm]),
              "training_identity": identity, "checks": summary,
              "status": "experimental release candidate; not promoted", "automatic_promotion": False}
    _write_json(folder / "config.json", config)
    return config


def load_export(model, folder):
    """Load one published head into a plain ProbeModel built from the pinned base."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if (config["model"] != MODEL_ID or config["revision"] != MODEL_REVISION
            or config["backbone_adapters"] != 0 or config["threshold"] != 0.5
            or config["sha256"] != sha256(folder / "heads.safetensors")):
        raise ValueError("Export identity or file checksum mismatch")
    arm = config["arm"]
    if config["head"] != model.head_specs()[arm]:
        raise ValueError("Classifier architecture differs from export")
    weights = load_file(str(folder / "heads.safetensors"))
    expected = {f"fast.{k}" for k in model.reference.gate.state_dict()}
    expected |= {f"answer.{k}" for k in model.heads[arm].state_dict()}
    if set(weights) != expected:
        raise ValueError("Unexpected export tensors")
    gate = {k[5:]: v for k, v in weights.items() if k.startswith("fast.")}
    validate_gate(gate)
    model.reference.gate.load_state_dict(gate, strict=True)
    model.heads[arm].load_state_dict({k[7:]: v for k, v in weights.items() if k.startswith("answer.")}, strict=True)
    model.requires_grad_(False).eval()
    if _state_digest(model.heads[arm]) != config["answer_state_digest"]:
        raise ValueError("Reloaded answer-head checksum mismatch")
    return config


def load_classifier(base, tokenizer, folder):
    """Construct exactly one answer head beside the original vocabulary head."""
    from .model import FrozenQwen
    from .probe_model import ProbeModel
    config = json.loads((Path(folder) / "config.json").read_text())
    model = ProbeModel(FrozenQwen(base, tokenizer))
    return model, load_export(model, folder)
