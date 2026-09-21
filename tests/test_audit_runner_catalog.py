"""Offline runner integration: catalog identity survives a multi-fit batch.

The runners, filesystem snapshots and hash checks are real. Dataset preparation,
model construction and fitting are replaced so no model or network is used.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import luna_benchmark, release_benchmark


class CloudSectionComplete(Exception):
    """Stop legacy mixed runner before entering its out-of-scope local section."""


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RunnerCatalogAuditTests(unittest.TestCase):
    def exercise(self, module, *, tamper=False, wrong_backend_hash=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'docs').mkdir()
            (root / 'docs/EXPERIMENT_PROTOCOL_RELEASE.md').write_text('Offline test protocol')
            data = root / 'data'
            (data / 'inputs').mkdir(parents=True)
            (data / 'inputs/manifest.json').write_text('{}')
            api_key_file = root / 'offline-key.txt'
            api_key_file.write_text('offline-no-request')
            source = root / 'user_catalog.json'
            original = b'{"snapshot_date":"2026-09-07","providers":{}}\n'
            source.write_bytes(original)
            out = root / 'batch'
            constructors = []
            fitted = []

            def make_backend(**kwargs):
                constructors.append(kwargs)
                path = Path(kwargs.get('catalog_path', source))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                return SimpleNamespace(
                    client=SimpleNamespace(with_options=lambda **opts: object()),
                    _request=lambda: None,
                    generation_config={'catalog_sha256': 'incorrect' if wrong_backend_hash else digest},
                )

            def fit(path, kind, dataset, seed, *args):
                fitted.append(seed)
                path.mkdir(parents=True)
                (path / 'complete.json').write_text('{}')
                frozen = out / 'source_snapshot/model_catalog.json'
                if tamper and len(fitted) == 1:
                    frozen.parent.mkdir(parents=True, exist_ok=True)
                    frozen.write_text('{"providers":{},"tampered":true}')

            with ExitStack() as stack:
                stack.enter_context(working_directory(root))
                stack.enter_context(patch.dict(os.environ, {'HYBRID_TOPIC_MODEL_CATALOG': str(source)}))
                stack.enter_context(patch.object(module, 'DATASETS', ['offline']))
                stack.enter_context(patch.object(module, 'SEEDS', [11, 23]))
                stack.enter_context(patch.object(module, 'prepare_reuse', return_value=({'offline': {}}, {})))
                stack.enter_context(patch.object(module, 'validate_cached_embeddings', return_value=([], [])))
                stack.enter_context(patch.object(module, 'SentenceTransformerEmbedder', return_value=object()))
                stack.enter_context(patch.object(module, 'OpenAITopicGenerator', side_effect=make_backend))
                stack.enter_context(patch.object(module, 'run_one', side_effect=fit))
                stack.enter_context(patch.object(module.importlib.metadata, 'version', return_value='offline'))
                if module is luna_benchmark:
                    stack.enter_context(patch.object(module, 'PLANNED_FITS', 2))
                    execute = lambda: module.execute_luna(data, root / 'cache', out, api_key_file)
                else:
                    stack.enter_context(patch.object(module, 'MLXTopicGenerator', side_effect=CloudSectionComplete))
                    execute = lambda: module.execute(data, root / 'cache', out, api_key_file, root / 'unused-local')

                if tamper or wrong_backend_hash:
                    with self.assertRaises((RuntimeError, ValueError)):
                        execute()
                elif module is release_benchmark:
                    with self.assertRaises(CloudSectionComplete):
                        execute()
                else:
                    execute()

            preflight = json.loads((out / 'preflight.json').read_text())
            self.assertIn('model_catalog', preflight)
            record = preflight['model_catalog']
            frozen = out / 'source_snapshot/model_catalog.json'
            self.assertEqual(Path(record['path']).resolve(), frozen.resolve())
            self.assertEqual(Path(record['source_path']).resolve(), source.resolve())
            self.assertEqual(record['sha256'], hashlib.sha256(original).hexdigest())
            expected_count = 1 if tamper or wrong_backend_hash else 2
            self.assertEqual(len(constructors), expected_count)
            for kwargs in constructors:
                self.assertEqual(Path(kwargs['catalog_path']).resolve(), frozen.resolve())
            if wrong_backend_hash:
                self.assertEqual(fitted, [], 'mismatched backend configuration must fail before fitting')
            elif tamper:
                self.assertEqual(fitted, [11], 'catalog tampering must stop before the next fit')
            else:
                self.assertEqual(fitted, [11, 23])
                self.assertEqual(frozen.read_bytes(), original)

    def test_luna_every_fit_uses_catalog_recorded_in_preflight(self):
        self.exercise(luna_benchmark)

    def test_release_cloud_every_fit_uses_catalog_recorded_in_preflight(self):
        self.exercise(release_benchmark)

    def test_luna_rejects_catalog_tampering_before_next_backend(self):
        self.exercise(luna_benchmark, tamper=True)

    def test_release_rejects_catalog_tampering_before_next_backend(self):
        self.exercise(release_benchmark, tamper=True)

    def test_luna_rejects_backend_using_different_catalog_before_fit(self):
        self.exercise(luna_benchmark, wrong_backend_hash=True)

    def test_release_rejects_backend_using_different_catalog_before_fit(self):
        self.exercise(release_benchmark, wrong_backend_hash=True)


if __name__ == '__main__':
    unittest.main()
