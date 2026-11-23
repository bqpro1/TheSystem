"""Offline checks of demo setup; no install, GPU connection or real token use."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
spec = importlib.util.spec_from_file_location("demo_setup", NOTEBOOKS / "prepare_runtime.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
DEMOS = ("01-Try-Probe-256.ipynb", "02-Try-LoRA.ipynb")


def source(name):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    return next(c["source"] for c in notebook["cells"] if c["id"] == "prepare-runtime")


class DemoSetupTests(unittest.TestCase):
    def test_demos_have_one_setup_cell_before_model_loading(self):
        for name in DEMOS:
            cells = json.loads((NOTEBOOKS / name).read_text())["cells"]
            loading = next(i for i, cell in enumerate(cells) if cell["id"] == "load-model")
            code = [cell for cell in cells[:loading] if cell["cell_type"] == "code"]
            self.assertEqual([cell["id"] for cell in code], ["prepare-runtime"])
            self.assertIn("# @title Prepare runtime", code[0]["source"])

    def test_notebook_replaces_stale_token_and_rejects_unverified_code(self):
        for name in DEMOS:
            response = Mock()
            response.__enter__ = Mock(return_value=io.BytesIO(b"raise AssertionError('must not execute')"))
            response.__exit__ = Mock(return_value=False)
            output = io.StringIO()
            with patch("sys.version_info", (3, 12)), patch("getpass.getpass", return_value="test-new-secret"), \
                    patch.dict(os.environ, {"HF_TOKEN": "test-old-secret"}), \
                    patch("urllib.request.urlopen", return_value=response) as download, redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "helper checksum mismatch"):
                    exec(source(name), {})
                self.assertEqual(os.environ["HF_TOKEN"], "test-new-secret")
                self.assertEqual(download.call_args.args[0].get_header("Authorization"), "Bearer test-new-secret")
            self.assertNotIn("test-new-secret", output.getvalue())

    def test_notebook_auth_error_is_actionable_and_does_not_execute_helper(self):
        for name in DEMOS:
            error = HTTPError("https://huggingface.co/test", 401, "Unauthorized", {}, None)
            with patch("sys.version_info", (3, 13)), patch("getpass.getpass", return_value="test-token"), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch("urllib.request.urlopen", side_effect=error), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "rerun this cell and paste a valid read token"):
                    exec(source(name), {})

    def test_notebook_rejects_wrong_python_before_token_or_network(self):
        with patch("sys.version_info", (3, 14)), patch("getpass.getpass") as prompt, \
                patch("urllib.request.urlopen") as download, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Python 3.12"):
                exec(source(DEMOS[0]), {})
            prompt.assert_not_called()
            download.assert_not_called()

    def test_ready_is_only_reported_after_all_checks(self):
        output = io.StringIO()
        torch = object()
        with patch.object(setup, "check_preflight"), patch.object(setup, "install_inference_dependencies"), \
                patch.object(setup, "run_command", return_value=0), \
                patch.object(setup, "install_wheel"), patch.object(setup, "verify_live_runtime", return_value=torch), \
                redirect_stdout(output):
            self.assertIs(setup.prepare_runtime("probe"), torch)
        for number in range(1, 5):
            self.assertIn(f"[{number}/4]", output.getvalue())
        self.assertIn("READY", output.getvalue())

    def test_failure_stops_before_later_stages(self):
        output = io.StringIO()
        with patch.object(setup, "check_preflight", side_effect=RuntimeError("bad GPU")), \
                patch.object(setup, "run_command") as command, redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, "bad GPU"):
                setup.prepare_runtime("lora")
            command.assert_not_called()
        self.assertIn("Setup is not ready", output.getvalue())
        self.assertNotIn("READY", output.getvalue())

    def test_changed_imported_dependency_stops_before_model_wheel(self):
        output = io.StringIO()
        with patch.object(setup, "check_preflight"), patch.object(setup, "verify_live_runtime"), \
                patch.object(setup, "install_inference_dependencies", side_effect=setup.RestartRequired("old imports")), \
                patch.object(setup, "install_wheel") as wheel, redirect_stdout(output):
            with self.assertRaisesRegex(setup.RestartRequired, "Restart session"):
                setup.prepare_runtime("lora")
            wheel.assert_not_called()
        self.assertNotIn("READY —", output.getvalue())

    def test_compatible_dependencies_do_not_run_pip(self):
        with patch.object(setup, "missing_requirements", return_value=[]), \
                patch.object(setup, "run_command") as command, redirect_stdout(io.StringIO()):
            setup.install_inference_dependencies(Path("setup.log"))
            command.assert_not_called()

    def test_dependency_install_protects_existing_torch_numpy_and_uses_only_wheels(self):
        versions = {"torch": "2.14.1+cu130", "numpy": "2.4.6"}
        observed = []
        def install(command, log):
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            observed.append((command, constraints))
        with patch.object(setup, "missing_requirements", side_effect=[["transformers==5.17.0"], []]), \
                patch.object(setup, "version", side_effect=versions.__getitem__), \
                patch.object(setup, "check_loaded_versions"), patch.object(setup, "run_command", side_effect=install), \
                redirect_stdout(io.StringIO()):
            setup.install_inference_dependencies(Path("setup.log"))
        command, constraints = observed[0]
        self.assertIn("--only-binary=:all:", command)
        self.assertIn("torch==2.14.1+cu130", constraints)
        self.assertIn("numpy==2.4.6", constraints)
        self.assertEqual(command[-1], "transformers==5.17.0")
        self.assertNotIn("torch==2.14.1+cu130", command)

    def test_dependency_resolver_cannot_silently_change_torch(self):
        calls = {"torch": 0}
        def installed(name):
            if name == "numpy": return "2.4.6"
            calls["torch"] += 1
            return "2.14.1+cu130" if calls["torch"] == 1 else "2.15.0"
        with patch.object(setup, "missing_requirements", return_value=["transformers==5.17.0"]), \
                patch.object(setup, "version", side_effect=installed), \
                patch.object(setup, "run_command"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "changed unexpectedly"):
                setup.install_inference_dependencies(Path("setup.log"))

    def test_requirements_select_only_missing_or_incompatible_libraries(self):
        versions = {"transformers": "5.22.1", "huggingface-hub": "1.33.0", "accelerate": "1.15.0", "safetensors": "0.8.0"}
        with patch.object(setup, "version", side_effect=versions.__getitem__):
            self.assertEqual(setup.missing_requirements(), ["transformers==5.17.0"])

    def test_stale_numpy_requires_restart_before_live_import_checks(self):
        with patch.dict(sys.modules, {"numpy": SimpleNamespace(__version__="2.5.4")}), \
                patch.object(setup, "version", return_value="2.3.3"):
            with self.assertRaisesRegex(setup.RestartRequired, "numpy is still loaded"):
                setup.verify_live_runtime()

    def test_failed_or_unrun_setup_blocks_model_loading(self):
        for name in DEMOS:
            cells = json.loads((NOTEBOOKS / name).read_text())["cells"]
            load = next(c["source"] for c in cells if c["id"] == "load-model")
            for ready in (False, None):
                with self.assertRaisesRegex(RuntimeError, "Prepare runtime.*READY"):
                    exec(load, {} if ready is None else {"RUNTIME_READY": ready})

    def test_live_runtime_still_rejects_non_native_bf16(self):
        for gib, capability, bf16 in ((24, 7, True), (80, 8, False)):
            cuda = SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _: SimpleNamespace(total_memory=gib * 1024**3),
                get_device_capability=lambda _: (capability, 0), is_bf16_supported=lambda: bf16,
            )
            with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda, __version__="2.14.1+cu130")}), \
                    patch.object(setup, "check_loaded_versions"):
                with self.assertRaisesRegex(RuntimeError, "native BF16 support"):
                    setup.verify_live_runtime()

    def test_24_gb_gpu_is_allowed_and_reports_free_memory(self):
        torch = SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                get_device_properties=lambda _: SimpleNamespace(total_memory=24 * 1024**3),
                get_device_capability=lambda _: (8, 9), get_device_name=lambda _: "NVIDIA L4",
                mem_get_info=lambda: (22 * 1024**3, 24 * 1024**3)))
        with patch.dict(sys.modules, {"torch": torch}), patch.object(setup, "check_loaded_versions"), \
                patch.object(setup, "version", return_value="2.4.6"), redirect_stdout(io.StringIO()) as output:
            self.assertIs(setup.verify_live_runtime(), torch)
        self.assertIn("22.0 GiB free of 24.0 GiB total", output.getvalue())

    def test_loading_and_request_oom_give_recovery_feedback_without_success(self):
        import torch
        for label in ("Download and load the selected model", "Run this request"):
            with self.subTest(label=label), patch.object(torch.cuda, "mem_get_info", return_value=(0, 24 * 1024**3)), \
                    redirect_stdout(io.StringIO()) as output:
                with self.assertRaisesRegex(RuntimeError, "GPU memory ran out") as stopped:
                    with setup.visible_progress(label):
                        raise torch.cuda.OutOfMemoryError("test allocation failure")
            self.assertIn(label, str(stopped.exception))
            self.assertIn("0.0 GiB free of 24.0 GiB total", str(stopped.exception))
            self.assertIn("restart the session", str(stopped.exception))
            self.assertIn("shorter prompt", str(stopped.exception))
            self.assertNotIn("Done", output.getvalue())

    def test_unrelated_errors_are_preserved(self):
        error = ValueError("Invalid prompt")
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as stopped:
            with setup.visible_progress("Run this request"):
                raise error
        self.assertIs(stopped.exception, error)

    def test_repeated_load_reuses_model_and_different_model_stops(self):
        for variant, name in zip(("probe", "lora"), DEMOS):
            cells = json.loads((NOTEBOOKS / name).read_text())["cells"]
            load = next(c["source"] for c in cells if c["id"] == "load-model")
            loader, resolver = Mock(), Mock()
            with patch.dict(sys.modules, {"thesystem": SimpleNamespace(load_model=loader),
                    "thesystem.artifacts": SimpleNamespace(resolve_model_resources=resolver)}), \
                    redirect_stdout(io.StringIO()):
                model = SimpleNamespace(variant=variant)
                env = {"RUNTIME_READY": True, "model": model}
                exec(load, env)
                self.assertIs(env["model"], model)
                with self.assertRaisesRegex(RuntimeError, "different model"):
                    exec(load, {"RUNTIME_READY": True, "model": SimpleNamespace(variant="other")})
            loader.assert_not_called()
            resolver.assert_not_called()

    def test_unsupported_torch_stops_without_installation(self):
        with patch.dict(sys.modules, {"torch": SimpleNamespace(__version__="2.15.0")}), \
                patch.object(setup, "check_loaded_versions"), patch.object(setup, "run_command") as command:
            with self.assertRaisesRegex(RuntimeError, "will not replace PyTorch"):
                setup.verify_live_runtime()
            command.assert_not_called()

    def test_long_step_emits_elapsed_heartbeat_and_finishes(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with setup.visible_progress("Test stage", interval=0.01):
                time.sleep(0.04)
        self.assertIn("Still running: Test stage", output.getvalue())
        self.assertIn("elapsed", output.getvalue())
        self.assertIn("Done", output.getvalue())

    def test_command_failure_retains_log_and_reports_tail(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            log = Path(tmp) / "setup.log"
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                setup.run_command([sys.executable, "-c", "print('test diagnostic'); raise SystemExit(7)"], log)
            self.assertIn("test diagnostic", log.read_text())
            self.assertIn("test diagnostic", output.getvalue())

    def test_wheel_verification_blocks_install_and_loaded_changes_require_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / setup.WHEEL_NAME
            wheel.write_bytes(b"test-wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(wheel)))
            errors = SimpleNamespace(HfHubHTTPError=type("HfHubHTTPError", (Exception,), {}))
            previous = SimpleNamespace(read_text=lambda _: json.dumps({"archive_info": {"hashes": {"sha256": "old"}}}))
            with patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors, "thesystem": object()}), \
                    patch.object(setup, "run_command") as command, patch.object(setup, "distribution", return_value=previous), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Wheel checksum mismatch"):
                    setup.install_wheel("probe", Path(tmp) / "log")
                command.assert_not_called()
                with patch.object(setup, "WHEEL_SHA256", digest):
                    with self.assertRaises(setup.RestartRequired):
                        setup.install_wheel("probe", Path(tmp) / "log")
                command.assert_not_called()

    def test_verified_matching_wheel_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / setup.WHEEL_NAME
            wheel.write_bytes(b"test-wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(wheel)))
            errors = SimpleNamespace(HfHubHTTPError=type("HfHubHTTPError", (Exception,), {}))
            previous = SimpleNamespace(read_text=lambda _: json.dumps({"archive_info": {"hashes": {"sha256": digest}}}))
            with patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors}), \
                    patch.object(setup, "WHEEL_SHA256", digest), patch.object(setup, "distribution", return_value=previous), \
                    patch.object(setup, "run_command") as command, redirect_stdout(io.StringIO()):
                setup.install_wheel("probe", Path(tmp) / "log")
                command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
