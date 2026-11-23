"""Checksums, immutable revision pins, and collision-safe release assembly."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thesystem.artifacts import resolve_model_resources, resolve_release_resources, verify_files


def manifest(files):
    entry = {'repo_id':'test/model','revision':'a'*40,'files':files}
    return {'schema':1,'version':'test','base_model':{'repo_id':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
            'models':{'probe':entry,'lora':dict(entry)},'dataset':dict(entry,repo_id='test/data')}


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.file = self.root/'checkpoints/value.bin'
        self.file.parent.mkdir()
        self.file.write_bytes(b'frozen weights')
        self.files = {'checkpoints/value.bin':{'size':self.file.stat().st_size,'sha256':hashlib.sha256(self.file.read_bytes()).hexdigest()}}
        self.manifest = manifest(self.files)
        (self.root/'RELEASE-MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_local_resolve_does_not_need_hub(self):
        with patch('thesystem.artifacts._download', side_effect=AssertionError('unexpected network')):
            self.assertEqual(resolve_model_resources('probe', root=self.root), self.root.resolve())
            self.assertEqual(resolve_release_resources(root=self.root), self.root.resolve())

    def test_corruption_and_missing_files_fail(self):
        self.file.write_bytes(b'changed weights')
        with self.assertRaises(ValueError): verify_files(self.root, self.files)
        self.file.unlink()
        with self.assertRaises(FileNotFoundError): verify_files(self.root, self.files)

    def test_unsafe_paths_and_revision_changes_fail(self):
        with self.assertRaises(ValueError): verify_files(self.root, {'../outside':next(iter(self.files.values()))})
        with self.assertRaises(ValueError): resolve_model_resources('probe', 'main', root=self.root)
        with self.assertRaises(ValueError): resolve_model_resources('unknown', root=self.root)

    def test_download_uses_locked_revision_and_file_allowlist(self):
        with patch.dict('os.environ', {'HF_TOKEN':'test-token'}), patch('thesystem.artifacts.read_release_manifest', return_value=self.manifest), patch('huggingface_hub.snapshot_download', return_value=str(self.root)) as download:
            self.assertEqual(resolve_model_resources('probe'), self.root)
            self.assertEqual(download.call_args.kwargs['revision'], 'a'*40)
            self.assertEqual(download.call_args.kwargs['allow_patterns'], list(self.files))
            self.assertEqual(download.call_args.kwargs['token'], 'test-token')
            with self.assertRaises(ValueError): resolve_model_resources('probe', 'main')

    def test_conflicting_shared_resources_fail_before_download(self):
        other = dict(self.files['checkpoints/value.bin'], sha256='0'*64)
        self.manifest['models']['lora'] = dict(self.manifest['models']['lora'], files={'checkpoints/value.bin':other})
        (self.root/'RELEASE-MANIFEST.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'Conflicting'): resolve_release_resources(root=self.root)

    def test_assembly_verifies_and_reuses_own_cache(self):
        with patch('thesystem.artifacts.read_release_manifest', return_value=self.manifest), patch('huggingface_hub.constants.HF_HUB_CACHE', str(self.root/'cache')), patch('thesystem.artifacts._download', return_value=self.root) as download:
            combined = resolve_release_resources()
            self.assertEqual(download.call_count, 3)
            self.assertEqual((combined/'checkpoints/value.bin').read_bytes(), self.file.read_bytes())
            self.assertEqual(resolve_release_resources(), combined)
            self.assertEqual(download.call_count, 3)
            (combined/'checkpoints/value.bin').write_bytes(b'tampered')
            self.assertEqual(self.file.read_bytes(), b'frozen weights')
            with self.assertRaises(ValueError): resolve_release_resources()

    def test_interrupted_copy_can_be_retried(self):
        def interrupted(source, target):
            target.write(b'partial')
            raise InterruptedError('simulated disconnect')
        with patch('thesystem.artifacts.read_release_manifest', return_value=self.manifest), patch('huggingface_hub.constants.HF_HUB_CACHE', str(self.root/'cache')), patch('thesystem.artifacts._download', return_value=self.root):
            with patch('thesystem.artifacts.shutil.copyfileobj', side_effect=interrupted):
                with self.assertRaises(InterruptedError): resolve_release_resources()
            combined = resolve_release_resources()
            self.assertEqual((combined/'checkpoints/value.bin').read_bytes(), b'frozen weights')

    def test_concurrent_assembly_produces_one_verified_release(self):
        with patch('thesystem.artifacts.read_release_manifest', return_value=self.manifest), patch('huggingface_hub.constants.HF_HUB_CACHE', str(self.root/'cache')), patch('thesystem.artifacts._download', return_value=self.root):
            with ThreadPoolExecutor(max_workers=2) as pool:
                roots = list(pool.map(lambda _: resolve_release_resources(), range(2)))
            self.assertEqual(roots[0], roots[1])
            verify_files(roots[0], self.files)


if __name__ == '__main__':
    unittest.main()
