import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('output_setup', ROOT / 'notebooks/prepare_evaluation.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

class OutputTests(unittest.TestCase):
    def test_drive_is_only_mounted_when_requested(self):
        for use_drive in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                drive = SimpleNamespace(mount=Mock())
                colab = SimpleNamespace(drive=drive, runtime=Mock())
                with patch.dict(sys.modules, {'google.colab': colab}), patch.dict(os.environ, {
                    'THESYSTEM_OUTPUT': str(Path(tmp)/'out'), 'THESYSTEM_MODEL_CACHE': str(Path(tmp)/'cache')}):
                    out, cache, in_colab, runtime = setup.prepare_output('test', use_drive)
                self.assertEqual(drive.mount.call_count, int(use_drive))
                self.assertTrue(out.is_dir())

    def test_default_storage_is_local_colab(self):
        import inspect
        self.assertFalse(inspect.signature(setup.prepare_output).parameters['USE_DRIVE'].default)
        self.assertIn('Path("/content") if IN_COLAB', inspect.getsource(setup.prepare_output))

    def test_table_uses_automatic_scores_and_website_order(self):
        subsets = ('vitaminc-dev', 'paws', 'squad2', 'aegis2', 'pubmedqa')
        decisions = {
            'qwen': {'rows': 8},
            'lora_automatic': {'accuracy': 0.75},
            'probe256_automatic': {'accuracy': 0.875},
        }
        references = {'nimble': {'accuracy_percent': 62.5}, 'jev': {'accuracy_percent': 50.0}}
        report = {
            'complete': True,
            'decisions': {key: decisions for key in subsets},
            'published': {
                'source_url': 'https://example.org/benchmark',
                'subsets': {key: references for key in subsets},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / 'SUMMARY.json'
            summary.write_text(json.dumps(report))
            markup = setup.results_html(tmp)
            self.assertIn('Benchmark accuracy — this evaluation run', markup)
            headers = ['Benchmark', '&lt;FAST&gt; + LoRA', '&lt;FAST&gt; + Probe-256', '>Bespoke-Nimble-9B<', '>Jev 1.13.0<']
            header = markup.split('<thead>', 1)[1].split('</thead>', 1)[0]
            positions = [header.index(label) for label in headers]
            self.assertEqual(positions, sorted(positions))
            for row in markup.split('<tbody>', 1)[1].split('</tbody>', 1)[0].split('</tr>')[:-1]:
                positions = [row.index(score) for score in ('75.00%', '87.50%', '62.5%', '50.0%')]
                self.assertEqual(positions, sorted(positions))
            self.assertEqual(markup.count('8 examples'), len(subsets))

            report['smoke'] = True
            summary.write_text(json.dumps(report))
            self.assertIn('QUICK CHECK', setup.results_html(tmp))

    @unittest.skipUnless(importlib.util.find_spec('ipywidgets'), 'widget audit environment required')
    def test_download_never_shuts_down_and_confirmation_is_required(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'REPORT.html').write_text('report')
            files=SimpleNamespace(download=Mock());runtime=SimpleNamespace(unassign=Mock())
            with patch.dict(sys.modules, {'google.colab': SimpleNamespace(files=files,runtime=runtime)}), patch('IPython.display.display'):
                download,saved,shutdown=setup.finish_controls(tmp)
                self.assertTrue(shutdown.disabled)
                shutdown.click(); runtime.unassign.assert_not_called()
                download.click(); runtime.unassign.assert_not_called()
                archive=files.download.call_args.args[0]
                with zipfile.ZipFile(archive) as z:self.assertEqual(z.read('REPORT.html'), b'report')
                saved.value=True; self.assertFalse(shutdown.disabled)
                download.click();self.assertFalse(saved.value);self.assertTrue(shutdown.disabled)
                saved.value=True;shutdown.click();runtime.unassign.assert_called_once()
