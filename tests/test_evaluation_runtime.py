import io
import json
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from thesystem.evaluation import runtime, evaluation_models, evaluation_worker


class EvaluationRuntimeTests(unittest.TestCase):
    def test_smaller_gpu_is_allowed_without_changing_precision_or_cuda(self):
        gpu = SimpleNamespace(name="24 GB GPU", total_memory=24 * 2**30)
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
                patch.object(torch.cuda, "get_device_properties", return_value=gpu), \
                patch.object(torch.cuda, "mem_get_info", return_value=(23 * 2**30, 24 * 2**30)), \
                patch.object(torch.version, "cuda", "13.0"), \
                patch.object(runtime.importlib.metadata, "version", return_value="pinned"), \
                redirect_stdout(io.StringIO()) as output:
            environment = runtime.check_gpu()
        self.assertEqual(environment["precision"], "bfloat16")
        self.assertEqual(environment["cuda"], "13.0")
        self.assertIn("23.0 GiB free", output.getvalue())

    def test_emulated_bf16_hardware_is_rejected(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            with self.assertRaisesRegex(RuntimeError, "native-BF16"):
                runtime.check_gpu()

    def test_both_arms_use_evaluation_hardware_check(self):
        for arm in ("probe", "lora"):
            vendor = Mock()
            with patch("importlib.import_module", return_value=vendor), \
                    patch.object(evaluation_models, "check_gpu", side_effect=RuntimeError("stop before model load")) as check:
                with self.assertRaisesRegex(RuntimeError, "stop before model load"):
                    evaluation_models.load_backend(arm, Path("cache"), {})
            check.assert_called_once()
            vendor.verify_runtime.assert_called_once()
            vendor.check_gpu.assert_not_called()

    def test_memory_failure_is_saved_and_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = SimpleNamespace(arm="probe")
            with patch.object(evaluation_worker, "heartbeat", return_value=nullcontext()), \
                    patch.object(evaluation_worker, "load_backend", side_effect=torch.cuda.OutOfMemoryError("allocation failed")), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Keep benchmark prompts, precision and generation limits unchanged"):
                    evaluation_worker.run_locked(args, folder, folder / "cache", {})
            status = json.loads((folder / "STATUS.json").read_text())
            self.assertEqual(status["error"], "OutOfMemoryError")
            self.assertIn("did not complete", status["detail"])
            self.assertFalse((folder / "COMPLETE.json").exists())
