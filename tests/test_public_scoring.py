import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.score_public_predictions import score_bundle


class PublicScoringTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'bundle'
        root.mkdir()
        rows = []
        for gen in ['GPT-4.1 mini', 'Luna']:
            for dataset in ['bills','sib200','massive','ag_news']:
                for seed in [11,23,37,53,71]:
                    for index, gold, eligible, predicted in [(0,0,1,0),(1,1,1,-1),(2,1,0,0)]:
                        row = dict(generator=gen,dataset=dataset,seed=seed,doc_index=index,gold=gold,eligible=eligible)
                        row.update({m:predicted for m in ['hybrid','initial','direct_hybrid','direct_initial','bertopic','kmeans']})
                        rows.append(row)
        pd.DataFrame(rows).to_csv(root/'predictions.csv',index=False)
        self.reseal(root)
        return root

    def reseal(self, root):
        frame = pd.read_csv(root/'predictions.csv')
        manifest = {'schema':'public-partition-evidence-v1',
                    'predictions_sha256':hashlib.sha256((root/'predictions.csv').read_bytes()).hexdigest(),
                    'counts':{d:{'original':3,'eligible':2} for d in ['bills','sib200','massive','ag_news']}}
        (root/'manifest.json').write_text(json.dumps(manifest))

    def test_scores_only_eligible_and_keeps_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            result = score_bundle(root,Path(d)/'out')
            self.assertEqual(len(result),240)
            self.assertTrue((result.n_documents == 2).all())
            self.assertTrue((result.coverage == .5).all())
            self.assertTrue((result.harmonic_purity == 1).all())
            self.assertTrue((result.topicgpt_weighted_best_f1 == 1).all())

    def test_rejects_tampered_file_before_writing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            with (root/'predictions.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'hash'):
                score_bundle(root,Path(d)/'out')
            self.assertFalse((Path(d)/'out').exists())

    def test_rejects_missing_run_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            f = pd.read_csv(root/'predictions.csv')
            f.iloc[3:].to_csv(root/'predictions.csv',index=False)
            self.reseal(root)
            with self.assertRaisesRegex(ValueError,'planned'):
                score_bundle(root,Path(d)/'out')

    def test_rejects_duplicate_document_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            f = pd.read_csv(root/'predictions.csv');f.loc[1,'doc_index']=0
            f.to_csv(root/'predictions.csv',index=False);self.reseal(root)
            with self.assertRaisesRegex(ValueError,'document'):
                score_bundle(root,Path(d)/'out')


if __name__ == '__main__':
    unittest.main()
