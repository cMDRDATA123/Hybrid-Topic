import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_topic import HybridTopic
from tests.test_model import StaticGenerator, WordEmbedder


class DefaultConfigurationTests(unittest.TestCase):
    documents = ["weather rain", "rain today", "music song", "music today", "finance tax"]

    def test_omitted_parameters_match_the_documented_fixed_profile(self):
        default = HybridTopic(embedding_model=WordEmbedder(), generator=StaticGenerator())
        explicit = HybridTopic(embedding_model=WordEmbedder(), generator=StaticGenerator(),
                               p=0.95, seed_per_topic=5, diffusion_rate=0.70,
                               rounds=10, max_residual_rounds=2)
        default.fit(self.documents)
        explicit.fit(self.documents)
        self.assertTrue(default.get_document_info().equals(explicit.get_document_info()))
        self.assertEqual(default.metadata_["parameter_defaults_version"], "empirical-defaults-v1")
        self.assertEqual(default.metadata_["configuration"], {
            "p": 0.95, "seed_per_topic": 5, "diffusion_rate": 0.70,
            "rounds": 10, "max_residual_rounds": 2,
        })
        self.assertEqual(default.metadata_["residual_generation_calls"], 0)

    def test_manual_override_and_saved_configuration_are_preserved(self):
        model = HybridTopic(embedding_model=WordEmbedder(), generator=StaticGenerator(),
                            p=0.99, seed_per_topic=1, diffusion_rate=0.55,
                            rounds=7, max_residual_rounds=0).fit(self.documents)
        expected = model.transform(["rain tomorrow", "finance tax"])
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = HybridTopic.load(tmp, embedding_model=WordEmbedder())
            self.assertEqual(loaded.metadata_["configuration"], model.metadata_["configuration"])
            self.assertEqual(loaded.metadata_["configuration"]["p"], 0.99)
            actual = loaded.transform(["rain tomorrow", "finance tax"])
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_missing_saved_parameter_is_not_replaced_by_a_new_default(self):
        model = HybridTopic(embedding_model=WordEmbedder(), generator=StaticGenerator(), p=.99).fit(self.documents)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            path = Path(tmp) / "model.json"
            payload = json.loads(path.read_text())
            del payload["configuration"]["p"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "configuration"):
                HybridTopic.load(tmp, embedding_model=WordEmbedder())


if __name__ == "__main__":
    unittest.main()
