import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.freeze_cloud_release import freeze


class ReleaseFreezeTests(unittest.TestCase):
    def test_freeze_copies_only_completed_cloud_runs_and_writes_new_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'mixed'
            run = source / 'cloud/bills_seed11'
            (run / 'requests').mkdir(parents=True)
            (run / 'result.txt').write_text('result')
            (run / 'complete.json').write_text(json.dumps({'files': {'result.txt': 'placeholder'}}))
            (run / 'requests/request_0001.json').write_text(json.dumps({'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}))
            (source / 'source_snapshot').mkdir()
            (source / 'source_snapshot/source.py').write_text('pass')
            (source / 'preflight.json').write_text(json.dumps({'backends': ['cloud', 'local'], 'planned_fits': 40}))
            output = root / 'cloud'
            with patch('experiments.freeze_cloud_release.DATASETS', ('bills',)), patch(
                'experiments.freeze_cloud_release.SEEDS', (11,)
            ):
                result = freeze(source, output)
            self.assertEqual(result['completed'], 1)
            preflight = json.loads((output / 'preflight.json').read_text())
            self.assertEqual(preflight['backends'], ['cloud'])
            self.assertEqual(preflight['planned_fits'], 1)
            frozen = json.loads((output / 'predictions_frozen.json').read_text())
            self.assertEqual(frozen['failures'], {})
            self.assertEqual(frozen['planned'], 1)
            usage = json.loads((output / 'usage_summary.json').read_text())
            self.assertEqual(usage['cloud_prompt_tokens'], 2)
            self.assertTrue((source / 'cloud/bills_seed11/result.txt').is_file())


if __name__ == '__main__':
    unittest.main()
