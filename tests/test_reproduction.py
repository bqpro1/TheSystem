import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thesystem.reproduction import _paths, train_lora, train_probe
from thesystem.reproduction.data import load_collection, smoke_rows, tokenize_shared
from thesystem.reproduction.common import sha, write_json


class ReproductionTests(unittest.TestCase):
    def test_exact_dataset_bytes_and_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/fast"
            folder.mkdir(parents=True)
            data = folder / "rows.jsonl"
            data.write_text('{"id":"one","prompt":"request"}\n')
            write_json(folder / "data-manifest.json", {"files": {"rows.jsonl": sha(data)}})
            rows, _ = load_collection(root, "fast", ("rows",))
            self.assertEqual(rows["rows"][0]["id"], "one")
            with self.assertRaisesRegex(ValueError, "Missing required"):
                load_collection(root, "fast", ("missing",))
            data.write_text(data.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Frozen dataset changed"):
                load_collection(root, "fast")

    def test_shared_answer_and_routing_rows_tokenize_once(self):
        rows = {"fresh": [{"id": "shared", "prompt": "choose", "mode_target": 1}],
                "routing": [{"id": "shared", "prompt": "choose", "mode_target": 1},
                            {"id": "write", "prompt": "explain", "mode_target": 0}]}
        calls = []
        def tokenize(datasets, *args):
            calls.extend(row["id"] for row in datasets["unique"])
            for row in datasets["unique"]:
                row["input_ids"] = [1, 2]
        tokenize_shared(rows, None, {}, None, tokenize)
        self.assertEqual(calls, ["shared", "write"])
        self.assertEqual(rows["routing"][0]["input_ids"], [1, 2])
        rows["routing"][0]["prompt"] = "different"
        with self.assertRaisesRegex(ValueError, "Conflicting repeated input"):
            tokenize_shared(rows, None, {}, None, tokenize)

    def test_smoke_retains_decision_and_writing(self):
        rows = [{"id": str(n), "mode_target": int(n < 50)} for n in range(100)]
        selected = smoke_rows(rows, 4)
        self.assertEqual([r["id"] for r in selected], ["0", "1", "50", "51"])

    def test_outputs_cannot_mutate_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "resources"
            root.mkdir()
            with patch("thesystem.artifacts.resolve_release_resources", return_value=root):
                for output in (root, root / "outputs", root.parent):
                    with self.assertRaises(ValueError):
                        _paths(root, output, root.parent / "cache")

    def test_training_exports_feed_gate_continuation_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, cache = (Path(tmp) / name for name in ("resources", "run", "cache"))
            calls = []
            def worker(module, args, resources, log):
                calls.append(args)
                if args[0] == "evaluate":
                    write_json(output / "evaluation/SUMMARY.json", {"complete": True})
            with patch("thesystem.reproduction._paths", return_value=(root, output, cache)), patch("thesystem.reproduction._run", side_effect=worker):
                train_lora(root, output, smoke=True)
            self.assertEqual(calls[1][calls[1].index("--initial-gate") + 1], output / "original/exports/original-fast/weights.safetensors")
            self.assertEqual(calls[2][calls[2].index("--gate-weights") + 1], output / "diverse-gate/export/weights.safetensors")
            self.assertTrue(all("--smoke" in args for args in calls))
            self.assertTrue(json.loads((output / "SUMMARY.json").read_text())["smoke"])

    def test_resume_rejects_changed_run_identity(self):
        from thesystem.reproduction import common
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "resources"
            root.mkdir()
            write_json(root / "RELEASE-MANIFEST.json", {"version": "fixture"})
            output = root.parent / "run"
            with patch.object(common, "ROOT", root):
                first = common.bind_run(output, {"seed": 42})
                self.assertEqual(common.bind_run(output, {"seed": 42}), first)
                with self.assertRaisesRegex(ValueError, "Run identity changed"):
                    common.bind_run(output, {"seed": 43})

    def test_failed_training_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output, cache = (Path(tmp) / name for name in ("resources", "run", "cache"))
            with patch("thesystem.reproduction._paths", return_value=(root, output, cache)), patch("thesystem.reproduction._run", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    train_probe(root, output)
            self.assertFalse((output / "SUMMARY.json").exists())
