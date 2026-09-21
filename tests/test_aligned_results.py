import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.build_aligned_paper_results import build_aligned


class AlignedResultsTests(unittest.TestCase):
    def fixture(self, directory):
        rows = [dict(generator=g, dataset=d, seed=s, method=m,
                     n_documents=10, harmonic_purity=72/85,
                     topicgpt_weighted_best_f1=8/9, coverage=.9, ari=.4, nmi=.5)
                for g in ['GPT-4.1 mini', 'Luna']
                for d in ['bills', 'sib200', 'massive', 'ag_news']
                for s in [11, 23, 37, 53, 71]
                for m in ['hybrid', 'initial', 'direct_hybrid', 'direct_initial', 'bertopic', 'kmeans']]
        path = Path(directory)/'scores.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_uses_topicgpt_metric_and_retains_legacy_scores(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(d)
            original = source.read_bytes()
            out = Path(d)/'aligned'
            summary = build_aligned(source, out)
            self.assertTrue(((summary.harmonic_purity_mean - 8/9).abs() < 1e-14).all())
            rows = pd.read_csv(out/'per_run_metrics.csv')
            self.assertTrue(((rows.legacy_global_hmp - 72/85).abs() < 1e-14).all())
            self.assertTrue((rows.coverage == .9).all())
            self.assertTrue((rows.ari == .4).all())
            self.assertEqual(source.read_bytes(), original)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['metric_version'], 'topicgpt-weighted-best-f1-v1')

    def test_rejects_legacy_only_and_invalid_metric(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(d)
            frame = pd.read_csv(source)
            frame.drop(columns='topicgpt_weighted_best_f1').to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'TopicGPT'):
                build_aligned(source, Path(d)/'missing')
            frame.loc[0, 'topicgpt_weighted_best_f1'] = float('inf')
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'finite'):
                build_aligned(source, Path(d)/'invalid')

    def test_missing_planned_position_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.fixture(d)
            pd.read_csv(source).iloc[1:].to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, 'accounting'):
                build_aligned(source, Path(d)/'missing')


if __name__ == '__main__':
    unittest.main()
