"""Notebook safety and executable-cell checks; no GPU or network is used here."""
import ast
import hashlib
import importlib.metadata
import importlib.util
from contextlib import redirect_stdout, nullcontext
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
EXPECTED = (
    "01-Try-Probe-256.ipynb", "02-Try-LoRA.ipynb", "03-Reproduce-Evaluation.ipynb",
)

spec = importlib.util.spec_from_file_location("demo_setup", NOTEBOOKS / "prepare_runtime.py")
demo_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_setup)
eval_spec = importlib.util.spec_from_file_location("eval_setup", NOTEBOOKS / "prepare_evaluation.py")
eval_setup = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(eval_setup)


def document(name):
    return json.loads((NOTEBOOKS / name).read_text())


def source(name, cell_id):
    return next(c["source"] for c in document(name)["cells"] if c["id"] == cell_id)


class NotebookTests(unittest.TestCase):
    def test_evaluation_has_five_ordered_steps(self):
        cells = [c for c in document(EXPECTED[2])["cells"] if c["cell_type"] == "code"]
        self.assertEqual([c["id"] for c in cells],
                         ["prepare-runtime", "configure-job", "release-resources", "run-and-shutdown", "download-and-finish"])
        for cell in cells:
            self.assertTrue(cell["source"].startswith("# @title"))
        env = {"RUNTIME_READY": False}
        for cell in cells[1:]:
            with self.assertRaisesRegex(RuntimeError, "Prepare runtime|steps 1–3|output folder"):
                exec(cell["source"], env)

    def test_evaluation_setup_reaches_ready_only_after_all_checks(self):
        order = []
        with patch.object(eval_setup, "visible_progress", return_value=nullcontext()), \
                patch.object(eval_setup, "check_preflight", side_effect=lambda: order.append("hardware")), \
                patch.object(eval_setup.subprocess, "run", side_effect=lambda *a, **k: order.append("client")), \
                patch.object(eval_setup, "download_wheel", side_effect=lambda: order.append("wheel")), \
                patch.object(eval_setup, "install_runtime", side_effect=lambda: order.append("runtime")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertTrue(eval_setup.prepare_runtime())
        self.assertEqual(order, ["hardware", "client", "wheel", "runtime"])
        self.assertIn("READY — continue", output.getvalue())

    def test_failed_evaluation_preflight_does_not_install(self):
        with patch.object(eval_setup, "visible_progress", return_value=nullcontext()), \
                patch.object(eval_setup, "check_preflight", side_effect=RuntimeError("no GPU")), \
                patch.object(eval_setup.subprocess, "run") as install, \
                redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "no GPU"):
                eval_setup.prepare_runtime()
        install.assert_not_called()
        self.assertNotIn("READY — continue", output.getvalue())

    def test_reference_restart_stops_before_live_imports(self):
        with patch.object(eval_setup.subprocess, "run", return_value=SimpleNamespace(returncode=10)):
            with patch.object(eval_setup, "check_preflight"), patch.object(eval_setup, "download_wheel"), \
                    redirect_stdout(io.StringIO()) as output:
                self.assertFalse(eval_setup.prepare_runtime())
            self.assertNotIn("READY", output.getvalue())

    def test_failed_gpu_install_keeps_diagnostics(self):
        failed = SimpleNamespace(returncode=1, stdout="GPU extension build failed")
        with patch.object(eval_setup.subprocess, "run", return_value=failed), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "exit code 1"):
                eval_setup.install_runtime()
        self.assertIn(failed.stdout, output.getvalue())

    def test_run_all_waits_for_an_actual_session_restart(self):
        cells = [c["source"] for c in document(EXPECTED[2])["cells"] if c["cell_type"] == "code"]
        helper = b"from setup_fixture import prepare_runtime, visible_progress"
        setup = SimpleNamespace(prepare_runtime=Mock(side_effect=[False, True]), visible_progress=None)
        display = SimpleNamespace(Markdown=str, clear_output=Mock(), display=Mock())
        pinned = hashlib.sha256((NOTEBOOKS / "prepare_evaluation.py").read_bytes()).hexdigest()
        prepare = cells[0].replace(pinned, hashlib.sha256(helper).hexdigest())
        with patch.dict(sys.modules, {"setup_fixture": setup, "IPython.display": display}), \
                patch.dict(os.environ), patch("sys.version_info", (3, 13)), \
                patch("getpass.getpass") as token, \
                patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(helper)), \
                redirect_stdout(io.StringIO()):
            session = {}
            exec(prepare, session)
            for cell in cells[1:]:
                exec(cell, session)
            self.assertFalse(session["RUNTIME_READY"] or session["OUTPUT_READY"] or session["RESOURCES_READY"])
            self.assertNotIn("OUTPUT", session)
            self.assertIn("Runtime → Restart session", display.display.call_args.args[0])
            exec(prepare, session)
            token.assert_not_called()
            setup.prepare_runtime.assert_called_once()

            fresh_session = {}
            exec(prepare, fresh_session)
            self.assertTrue(fresh_session["RUNTIME_READY"])
            self.assertFalse(fresh_session["RESTART_PENDING"])
            token.assert_not_called()

    def preflight(self, name, python=(3, 12, 13), gpu="NVIDIA A100 80GB, 580.95.05, 81920\n"):
        output = io.StringIO()
        query = Mock(return_value=SimpleNamespace(stdout=gpu))
        error = None
        with patch("sys.version_info", python), \
                patch("platform.python_version", return_value=".".join(map(str, python))), \
                patch("platform.system", return_value="Linux"), \
                patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch("subprocess.run", query), redirect_stdout(output):
            try:
                if name in EXPECTED[:2]:
                    demo_setup.check_preflight()
                else:
                    eval_setup.check_preflight()
            except RuntimeError as exc:
                error = exc
        return output.getvalue(), query, error

    def test_preflight_rejects_wrong_python_before_commands(self):
        for name in EXPECTED:
            for version in ((3, 14, 0), (3, 11, 9)):
                with self.subTest(name=name, version=version):
                    output, query, error = self.preflight(name, python=version)
                    self.assertIn("Python:", output)
                    self.assertIsNotNone(error)
                    self.assertIn("latest runtime", str(error))
                    query.assert_not_called()

    def test_preflight_prints_hardware_and_only_runs_nvidia_smi(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                output, query, error = self.preflight(name)
                self.assertIsNone(error)
                self.assertIn("NVIDIA A100 80GB, 580.95.05, 81920", output)
                self.assertIn("native BF16 support", output)
                query.assert_called_once()
                self.assertEqual(query.call_args.args[0][0], "nvidia-smi")
                self.assertIn("--query-gpu=name,driver_version,memory.total", query.call_args.args[0])
                self.assertTrue(query.call_args.kwargs["capture_output"])

    def test_recorded_python313_runtime_is_accepted(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                output, query, error = self.preflight(name, python=(3, 13, 16))
                self.assertIsNone(error)
                self.assertIn("3.13.16", output)
                query.assert_called_once()

    def test_evaluation_preflight_allows_smaller_gpu_capacities(self):
        for mib in (16384, 23034, 24576, 40959, 40960):
            with self.subTest(memory_mib=mib):
                output, query, error = self.preflight(EXPECTED[2], gpu=f"Test GPU, 580.95.05, {mib}\n")
                self.assertIsNone(error)
                self.assertIn("No memory-size cutoff", output)
                query.assert_called_once()

    def test_demo_preflight_allows_smaller_gpu_capacities(self):
        for name in EXPECTED[:2]:
            for mib in (16384, 23034, 24576, 40959):
                with self.subTest(name=name, memory_mib=mib):
                    output, query, error = self.preflight(name, gpu=f"Test GPU, 580.95.05, {mib}\n")
                    self.assertIsNone(error)
                    self.assertIn("No memory-size cutoff", output)
                    query.assert_called_once()

    def test_preflight_rejects_unknown_memory(self):
        for name in EXPECTED:
            for memory in ("N/A", "nan", "inf"):
                with self.subTest(name=name, memory=memory):
                    _, _, error = self.preflight(name, gpu=f"Test GPU, 580.95.05, {memory}\n")
                    self.assertIsInstance(error, RuntimeError)

    def test_all_notebooks_have_unique_cells_and_compile_without_saved_outputs(self):
        self.assertEqual(sorted(p.name for p in NOTEBOOKS.glob("*.ipynb")), list(EXPECTED))
        for name in EXPECTED:
            doc = document(name)
            self.assertEqual((doc["nbformat"], doc["nbformat_minor"]), (4, 5))
            ids = [c["id"] for c in doc["cells"]]
            self.assertEqual(len(ids), len(set(ids)))
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile(cell["source"], f"{name}:{cell['id']}", "exec")

    def test_bootstrap_pins_are_immutable_or_explicitly_unfinished(self):
        for name in EXPECTED:
            env = {}
            if name in EXPECTED[:2]:
                variant = "probe" if name == EXPECTED[0] else "lora"
                pin = dict(demo_setup.WHEELS[variant], filename=demo_setup.WHEEL_NAME, sha256=demo_setup.WHEEL_SHA256)
                node = next(n for n in ast.parse(source(name, "prepare-runtime")).body
                            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "PREPARE")
                helper = ast.literal_eval(node.value)
                self.assertEqual(helper["repo_id"], "bqpro1/TheSystem")
                self.assertEqual(helper["sha256"], hashlib.sha256((NOTEBOOKS / "prepare_runtime.py").read_bytes()).hexdigest())
                if document(name)["metadata"]["thesystem"]["bootstrap_finalized"] or os.environ.get("THESYSTEM_REQUIRE_FINAL_PINS") == "1":
                    self.assertRegex(helper["revision"], r"^[0-9a-f]{40}$")
                else:
                    self.assertEqual(helper["revision"], "__PREPARE_REVISION__")
            else:
                pin = eval_setup.BOOTSTRAP
                node = next(n for n in ast.parse(source(name, "prepare-runtime")).body
                            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "PREPARE")
                helper = ast.literal_eval(node.value)
                self.assertEqual(helper["repo_id"], "bqpro1/TheSystem")
                if document(name)["metadata"]["thesystem"]["bootstrap_finalized"] or os.environ.get("THESYSTEM_REQUIRE_FINAL_PINS") == "1":
                    self.assertRegex(helper["revision"], r"^[0-9a-f]{40}$")
                    self.assertEqual(helper["sha256"], hashlib.sha256((NOTEBOOKS / "prepare_evaluation.py").read_bytes()).hexdigest())
            self.assertIn(pin["repo_id"], ("Lbuk/TheSystem-9B-Probe-256", "Lbuk/TheSystem-9B-LoRA"))
            self.assertEqual(pin["filename"], "thesystem-0.1.0rc3-py3-none-any.whl" if name in EXPECTED[:2] else "thesystem-0.1.0rc4-py3-none-any.whl")
            final = name in EXPECTED[:2] or document(name)["metadata"]["thesystem"]["bootstrap_finalized"]
            if final or os.environ.get("THESYSTEM_REQUIRE_FINAL_PINS") == "1":
                self.assertRegex(pin["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")
            else:
                self.assertEqual(pin["revision"], "__WHEEL_REVISION__")
                self.assertEqual(pin["sha256"], "__WHEEL_SHA256__")

    def test_demo_run_all_keeps_model_loaded(self):
        for name in EXPECTED[:2]:
            model = object()
            env = {"model": model, "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))}
            with redirect_stdout(io.StringIO()):
                exec(source(name, "cleanup"), env)
            self.assertIs(env["model"], model)
            env["torch"].cuda.empty_cache.assert_not_called()

    def execute_bootstrap(self, wheel, previous_sha, pin_sha):
        hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(wheel)))
        previous = SimpleNamespace(read_text=lambda _: json.dumps({"archive_info": {"hashes": {"sha256": previous_sha}}}))
        installer = Mock()
        pin = {"repo_id": "Lbuk/TheSystem-9B-Probe-256", "revision": "a" * 40,
               "filename": wheel.name, "sha256": pin_sha}
        with patch.dict(sys.modules, {"huggingface_hub": hub, "thesystem": SimpleNamespace()}), \
                patch.dict(os.environ, {}, clear=True), patch("getpass.getpass") as prompt, \
                patch("importlib.metadata.distribution", return_value=previous), \
                patch.object(eval_setup, "BOOTSTRAP", pin), \
                patch.object(eval_setup.subprocess, "run", installer), \
                redirect_stdout(io.StringIO()):
            eval_setup.download_wheel()
        prompt.assert_not_called()
        hub.hf_hub_download.assert_called_once_with(repo_id=pin["repo_id"], revision=pin["revision"],
                                                    filename=pin["filename"], token=False)
        return installer

    def test_same_version_wheel_is_reinstalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "thesystem-0.1.0rc1-py3-none-any.whl"
            wheel.write_bytes(b"test-wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            installer = self.execute_bootstrap(wheel, digest, digest)
            command = installer.call_args.args[0]
            self.assertIn("--force-reinstall", command)
            self.assertIn("--no-deps", command)

    def test_changed_imported_wheel_requires_kernel_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "thesystem-0.1.0rc1-py3-none-any.whl"
            wheel.write_bytes(b"test-wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
                self.execute_bootstrap(wheel, "b" * 64, digest)

    def test_changed_wheel_bytes_fail_before_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "thesystem-0.1.0rc1-py3-none-any.whl"
            wheel.write_bytes(b"changed-wheel")
            with self.assertRaisesRegex(RuntimeError, "Wheel checksum mismatch"):
                self.execute_bootstrap(wheel, "b" * 64, "b" * 64)

    def test_demo_layouts_route_automatically_and_explain_the_returned_choice(self):
        def respond(prompt, *, choices, mode, max_new_tokens):
            self.assertEqual(mode, "auto")
            if choices is None:
                return {"mode": "generation", "gate_probability": 0.1, "answer": "Generated text"}
            code = next(reversed(choices))
            return {"mode": "decision", "gate_probability": 0.9, "selected_code": code, "answer": choices[code]}

        def progress(label):
            print(f"Running {label}")
            return nullcontext()

        for name in EXPECTED[:2]:
            with self.subTest(name=name):
                model = Mock()
                model.respond.side_effect = respond
                env = {"model": model, "visible_progress": progress}
                cells = document(name)["cells"]
                start = next(i for i, cell in enumerate(cells) if cell["id"] == "decision-intro")
                end = next(i for i, cell in enumerate(cells) if cell["id"] == "cleanup-intro")
                displayed_modes = []
                for cell in cells[start:end]:
                    if cell["cell_type"] != "code":
                        continue
                    with redirect_stdout(io.StringIO()) as output:
                        exec(cell["source"], env)
                    if cell["id"] == "display-response":
                        continue
                    answer, metadata = output.getvalue().split("\nMetadata\n")
                    self.assertTrue(answer.startswith("Answer\n"))
                    self.assertNotIn("<FAST> decision probability:", answer)
                    modes = re.findall(r"^Selected mode: (\w+)$", answer, re.MULTILINE)
                    self.assertEqual(metadata.count("<FAST> decision probability:"), len(modes))
                    self.assertIn("Response time:", metadata)
                    displayed_modes.extend(modes)

                calls = model.respond.call_args_list
                self.assertEqual(len(calls), 12)
                flat, array, nested, named = [json.loads(call.args[0]) for call in calls[:4]]
                maps = [
                    flat["alternatives"],
                    {item["code"]: item["label"] for item in array["options"]},
                    nested["assessment_job"]["response_menu"],
                    named["candidate_codes"],
                ]
                self.assertEqual(maps, [call.kwargs["choices"] for call in calls[:4]])
                for call in calls[4:8]:
                    self.assertIsNone(call.kwargs["choices"])
                for call in calls[4:6]:
                    self.assertFalse(call.args[0].startswith("{"))
                for call in calls[6:8]:
                    self.assertIn("request", json.loads(call.args[0]))

                for original, decision, explanation in zip(calls[:2], calls[8::2], calls[9::2]):
                    self.assertEqual(original, decision)
                    self.assertIsNone(explanation.kwargs["choices"])
                    record = json.loads(explanation.args[0])["decision"]
                    self.assertEqual(record["context"], json.loads(decision.args[0])["context"])
                    choices = decision.kwargs["choices"]
                    code = next(reversed(choices))
                    self.assertEqual((record["selected_code"], record["selected_answer"]), (code, choices[code]))
                self.assertEqual(displayed_modes, ["decision"] * 4 + ["generation"] * 4 + ["decision", "generation"] * 2)

    def test_demo_loop_skips_explanation_when_the_decision_misroutes(self):
        for name in EXPECTED[:2]:
            run_example = Mock(return_value={"mode": "generation", "answer": "Generated text"})
            env = dict(run_example=run_example, team_request={}, team_choices={}, priority_request={}, priority_choices={})
            with self.subTest(name=name), redirect_stdout(io.StringIO()) as output:
                exec(source(name, "editable-example"), env)
                self.assertEqual(run_example.call_count, 2)
                self.assertTrue(all(call.kwargs["expected_mode"] == "decision" for call in run_example.call_args_list))
                self.assertEqual(output.getvalue().count("Skipping the explanation"), 2)

    def test_smoke_setting_reaches_every_long_workflow(self):
        for name in EXPECTED[2:]:
            tree = ast.parse(source(name, "configure-job"))
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
            call_name = fn.body[0].value.func.id
            job = Mock(return_value=Path("result"))
            env = {call_name: job, "RESOURCE_ROOT": "inputs", "OUTPUT": "outputs", "CACHE": "cache", "SMOKE": True}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), env)
            env["run_job"]()
            job.assert_called_once_with("inputs", "outputs", cache="cache", smoke=True)

    def run_final_cell(self, output, job, shutdown):
        env = {"RUNTIME_READY": True, "OUTPUT_READY": True, "RESOURCES_READY": True,
               "_setup": {"show_results": Mock()}, "OUTPUT": output, "RUN_NAME": "test", "run_job": job,
               "IN_COLAB": True, "AUTO_SHUTDOWN": True, "runtime": SimpleNamespace(unassign=shutdown),
               "SMOKE": True, "Path": Path, "datetime": datetime, "timezone": timezone,
               "json": json, "os": os}
        with redirect_stdout(io.StringIO()):
            exec(source(EXPECTED[2], "run-and-shutdown"), env)

    def test_success_is_saved_without_colab_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "SUMMARY.json").write_text('{"complete": true}')
            observed = []
            def shutdown():
                observed.append(json.loads((out / "notebook-status.json").read_text())["status"])
            self.run_final_cell(out, lambda: out, shutdown)
            self.assertEqual(observed, [])
            self.assertEqual(json.loads((out / "notebook-status.json").read_text())["status"], "completed")

    def test_job_failure_is_recorded_without_colab_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stopped = Mock()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_final_cell(out, Mock(side_effect=RuntimeError("interrupted")), stopped)
            self.assertEqual(json.loads((out / "notebook-status.json").read_text())["status"], "failed")
            stopped.assert_not_called()

    def test_failed_persistent_save_does_not_disconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "SUMMARY.json").write_text('{"complete": true}')
            stopped = Mock()
            with patch("os.fsync", side_effect=OSError("storage unavailable")):
                with self.assertRaisesRegex(OSError, "storage unavailable"):
                    self.run_final_cell(out, lambda: out, stopped)
            stopped.assert_not_called()

    def test_incomplete_summary_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "SUMMARY.json").write_text('{"complete": false}')
            stopped = Mock()
            with self.assertRaisesRegex(RuntimeError, "did not report complete"):
                self.run_final_cell(out, lambda: out, stopped)
            self.assertEqual(json.loads((out / "notebook-status.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
