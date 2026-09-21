import unittest
import pandas as pd
from experiments.paper_tables import summarize

class PaperTableTests(unittest.TestCase):
 def test_missing_seed_cannot_silently_shrink_average(self):
  frame=pd.DataFrame([{'dataset':'bills','method':'hybrid','seed':11}])
  with self.assertRaisesRegex(ValueError,'planned'):summarize(frame)
 def test_actual_frozen_run_completeness_and_means(self):
  from pathlib import Path
  for batch in ['release_cloud_20260906','release_luna_20260906']:
   source=Path('runs',batch,'evaluation/all_metrics.csv')
   if not source.exists():self.skipTest('private frozen run artifacts not distributed')
   frame=pd.read_csv(source);frame=frame[(frame['split']=='test')&(frame['sample_scope']=='eligible')]
   summary=summarize(frame)
   self.assertEqual(len(summary),24)
   self.assertTrue((summary['runs']==5).all())
   known=pd.read_csv(source.parent/'test_summary.csv')
   merged=summary.merge(known,on=['dataset','method'])
   for a,b in [('harmonic_purity_mean','hmp_mean'),('harmonic_purity_std','hmp_std')]:
    self.assertLess(float((merged[a]-merged[b]).abs().max()),1e-12)
