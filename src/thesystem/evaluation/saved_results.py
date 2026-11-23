"""Reconstruct measured release results from bundled per-case evidence; no inference."""
import argparse
from pathlib import Path

from .evaluation_data import ROOT, load_suites, read_rows
from .evaluation_io import bind_run, digest, evaluate_stage, exclusive_run, read_json, sha, write_json
from .build_report import build


def reconstruct(output):
    from thesystem.artifacts import resolve_release_resources
    resolve_release_resources(ROOT)
    output = Path(output).expanduser().resolve()
    if output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError("Reconstructed reports must be outside immutable release resources")
    evidence = ROOT / "evidence/reference-results"
    metadata = read_json(evidence / "provenance.json")
    suites = load_suites()
    stages = {**{"decisions/" + name: rows for name, rows in suites["decisions"].items()},
              **{name: suites[name] for name in ("paired_writing", "routing_historical", "writing")}}
    for arm in ("probe", "lora"):
        folder = output / arm
        identity = dict(package_sha256=sha(ROOT / "RELEASE-MANIFEST.json"), arm=arm,
                        batch_size=1, generation_caps=[512, 1024], seed=42, smoke=False,
                        source_sha256={"reference_evidence": sha(evidence / "provenance.json")},
                        packages=metadata["packages"], origin="saved_evidence", inference_performed=False)
        bind_run(folder, identity)
        with exclusive_run(folder):
            state = {}
            for name, expected in stages.items():
                rows = read_rows(evidence / arm / (name + ".jsonl"))
                if [r["id"] for r in rows] != [r["id"] for r in expected]:
                    raise ValueError(f"Reference rows differ from the frozen evaluation inputs: {arm}/{name}")
                for actual, source in zip(rows, expected):
                    if actual["group_id"] != source["group_id"]:
                        raise ValueError("Reference context groups differ")
                    if name.startswith("decisions/") and (actual["target"] != source["target_index"] or actual["option_keys"] != source["option_keys"]):
                        raise ValueError("Reference labels or candidate order differ")
                    if name in ("paired_writing", "routing_historical") and actual["target"] != source["mode_target"]:
                        raise ValueError("Reference routing label differs")
                    if name == "writing" and (actual["prompt"] != source["prompt"] or actual["suite"] != source["writing_suite"]):
                        raise ValueError("Reference writing prompt differs")
                lookup = {row["id"]: row for row in rows}
                evaluate_stage(expected, folder / name, identity,
                               lambda batch: [lookup[row["id"]] for row in batch], state,
                               shard_size=1 if name == "writing" else 32)
            write_json(folder / "COMPLETE.json", dict(complete=True, identity=digest(identity)))
    build(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    reconstruct(parser.parse_args().output)
