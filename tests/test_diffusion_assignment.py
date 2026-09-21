import unittest
import numpy as np
from hybrid_topic.diffusion_assignment import diffuse_scores

class DiffusionAssignmentTests(unittest.TestCase):
    def test_diffusion_uses_neighbor_evidence_not_direct_prototype_only(self) -> None:
        similarity = np.asarray(
            [
                [1.0, 0.92, 0.10],
                [0.92, 1.0, 0.15],
                [0.10, 0.15, 1.0],
            ],
            dtype=np.float64,
        )
        seed_scores = np.asarray(
            [
                [0.99, 0.01],
                [0.20, 0.05],
                [0.01, 0.95],
            ],
            dtype=np.float64,
        )

        result = diffuse_scores(seed_scores, similarity, p=0.90, rounds=2, seed_per_topic=1)

        self.assertEqual(result.assignments.tolist(), [0, 0, 1])
        self.assertGreater(result.scores[1, 0], seed_scores[1, 0])


    def test_diffusion_reports_natural_coverage_without_forced_fallback(self) -> None:
        similarity = np.asarray(
            [
                [1.0, 0.90, 0.10, 0.10],
                [0.90, 1.0, 0.10, 0.10],
                [0.10, 0.10, 1.0, 0.20],
                [0.10, 0.10, 0.20, 1.0],
            ],
            dtype=np.float64,
        )
        seed_scores = np.asarray(
            [
                [0.90, 0.10],
                [0.80, 0.10],
                [0.20, 0.70],
                [0.10, 0.60],
            ],
            dtype=np.float64,
        )

        result = diffuse_scores(seed_scores, similarity, p=0.90, rounds=2, seed_per_topic=1)

        self.assertEqual(result.assignments.tolist(), [0, 0, 1, -1])
        self.assertEqual(result.reached.tolist(), [True, True, True, False])
        self.assertEqual(result.natural_coverage, 0.75)


    def test_diffusion_can_stop_before_the_ten_iteration_cap_when_converged(self) -> None:
        similarity = np.eye(3, dtype=np.float64)
        seed_scores = np.asarray([[0.9, 0.1], [0.8, 0.1], [0.1, 0.7]], dtype=np.float64)

        result = diffuse_scores(
            seed_scores,
            similarity,
            p=0.90,
            rounds=10,
            seed_per_topic=1,
            convergence_tolerance=1.0,
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.iterations_run, 1)




