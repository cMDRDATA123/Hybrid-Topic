import unittest

import numpy as np

from hybrid_topic.evaluation import clean_text, evaluate


class EvaluationTests(unittest.TestCase):
    def test_clean_text_removes_headers_and_quotes(self) -> None:
        text = "Subject: old thread\n> quoted text\nUseful content\n\nMore content"
        self.assertEqual(clean_text(text), "Useful content More content")

    def test_evaluate_perfect_cluster_permutation(self) -> None:
        metrics = evaluate(np.asarray([0, 0, 1, 1]), np.asarray([3, 3, 2, 2]))
        self.assertEqual(metrics["harmonic_purity"], 1.0)
        self.assertEqual(metrics["mapped_accuracy"], 1.0)
        self.assertEqual(metrics["ari"], 1.0)
        self.assertEqual(metrics["nmi"], 1.0)

    def test_evaluate_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(np.asarray([]), np.asarray([]))


if __name__ == "__main__":
    unittest.main()
