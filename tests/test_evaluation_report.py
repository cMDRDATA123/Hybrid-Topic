import json
import unittest

import numpy as np

from experiments.evaluation_report import build_evaluation_report
from hybrid_topic.evaluation import evaluate


class EvaluationReportTests(unittest.TestCase):
    def test_partial_rejection_preserves_full_partition_scores(self):
        gold, pred = [0, 0, 1, 1], [7, 7, -1, -1]
        report = build_evaluation_report(gold, pred)
        self.assertEqual(report['coverage'], 0.5)
        self.assertEqual(report['n_assigned_topics'], 1)
        self.assertEqual(report['n_partitions_including_rejection'], 2)
        self.assertEqual(report['n_rejected'], 2)
        for key, value in report['all_documents']['metrics'].items():
            self.assertEqual(value, evaluate(np.array(gold), np.array(pred))[key])
        # Rejection happens to coincide with a gold class: the legacy score is 1.
        # The report must expose the 50% rejection, not silently repair the score.
        self.assertEqual(report['all_documents']['metrics']['harmonic_purity'], 1)
        self.assertEqual(report['assigned_documents']['n_documents'], 2)

    def test_all_rejected_has_no_assigned_metrics(self):
        report = build_evaluation_report([0, 1], [-1, -1])
        self.assertEqual(report['coverage'], 0)
        self.assertEqual(report['n_assigned_topics'], 0)
        self.assertIsNone(report['assigned_documents']['metrics'])
        self.assertIn('no_assigned_documents', report['warnings'])
        json.dumps(report, allow_nan=False)

    def test_full_assignment_matches_both_scopes(self):
        report = build_evaluation_report([0, 0, 1], [8, 8, 4])
        self.assertEqual(report['coverage'], 1)
        self.assertEqual(report['all_documents']['metrics'], report['assigned_documents']['metrics'])
        self.assertEqual(report['n_assigned_topics'], 2)

    def test_single_assigned_document_is_flagged(self):
        report = build_evaluation_report([0, 1], [4, -1])
        self.assertIn('single_assigned_document', report['warnings'])
        self.assertIn('single_assigned_gold_class', report['warnings'])

    def test_invalid_inputs_are_not_silently_cast(self):
        for gold, pred in [([], []), ([0], []), ([[0]], [0]),
                           ([0], [0.8]), ([0.5], [0]), ([0], ['1']),
                           ([0], [True]), ([True], [0]), ([0, 1], [0, True]),
                           ([0, True], [0, 1]), ([0], [-2]),
                           ([0], [float('nan')])]:
            with self.subTest(gold=gold, pred=pred), self.assertRaises(ValueError):
                build_evaluation_report(gold, pred)

    def test_does_not_mutate_input_arrays(self):
        gold, pred = np.array([0, 1]), np.array([5, -1])
        build_evaluation_report(gold, pred)
        np.testing.assert_array_equal(gold, [0, 1])
        np.testing.assert_array_equal(pred, [5, -1])


if __name__ == '__main__':
    unittest.main()
