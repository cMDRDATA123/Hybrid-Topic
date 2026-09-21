import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_topic import HybridTopic
from hybrid_topic.types import GenerationResult, Topic


WORDS = ("weather", "music", "finance", "sport", "politics")


def topic(word):
    return Topic(word, word.title(), f"Discussion of {word}", (word,))


class KeywordEmbedder:
    model_name = "residual-test-v1"

    def encode(self, texts):
        return np.asarray([[float(w in text.lower()) for w in WORDS] for text in texts])


class ResidualGenerator:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def generate_initial(self, documents):
        return GenerationResult((topic("weather"), topic("music")), "initial", "fixture")

    def generate_residual(self, existing_codebook, residual_documents):
        self.calls.append((tuple(t.name for t in existing_codebook), list(residual_documents)))
        proposal = self.proposals[len(self.calls) - 1]
        if isinstance(proposal, Exception):
            raise proposal
        return GenerationResult(tuple(proposal), "residual", "fixture")


class BoundedResidualTests(unittest.TestCase):
    def setUp(self):
        self.documents = [f"{w} {i}" for w in WORDS for i in range(2)]

    def model(self, generator, **kwargs):
        return HybridTopic(embedding_model=KeywordEmbedder(), generator=generator,
                           p=0.95, seed_per_topic=1, **kwargs)

    def test_default_adds_two_rounds_and_passes_only_current_residuals(self):
        generator = ResidualGenerator([[topic("finance")], [topic("sport")], [topic("politics")]])
        model = self.model(generator).fit(self.documents)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(generator.calls[0][1], self.documents[4:])
        self.assertEqual(generator.calls[1][1], self.documents[6:])
        self.assertIn("Finance", generator.calls[1][0])
        self.assertEqual(model.get_document_info()["topic_id"].tolist(), [0, 0, 1, 1, 2, 2, 3, 3, -1, -1])
        self.assertEqual(model.metadata_["generation_calls"], 3)
        self.assertEqual(model.metadata_["residual_stop_reason"], "max_residual_rounds_reached")
        self.assertEqual([s["accepted_topics"] for s in model.metadata_["residual_history"]], [1, 1])
        queries = ["finance new", "politics new", "sport new"]
        ids, scores = model.transform(queries)
        self.assertEqual(ids.tolist(), [2, -1, 3])
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = HybridTopic.load(tmp, embedding_model=KeywordEmbedder())
            actual = loaded.transform(queries)
            np.testing.assert_array_equal(actual[0], ids)
            np.testing.assert_array_equal(actual[1], scores)
            self.assertEqual(loaded.metadata_, model.metadata_)

    def test_empty_or_duplicate_proposals_stop_early(self):
        for proposals, reason in [
            ([], "no_new_topics"),
            ([topic("weather"), Topic("W2", "Rain", "weather", ("weather",))],
             "only_duplicate_or_unsupported_topics"),
        ]:
            generator = ResidualGenerator([proposals])
            model = self.model(generator).fit(self.documents)
            self.assertEqual(len(generator.calls), 1)
            self.assertEqual(len(model.get_topic_info()), 2)
            self.assertEqual(model.metadata_["residual_stop_reason"], reason)

    def test_small_residual_and_disabled_expansion_make_no_extra_calls(self):
        for documents, options in [(self.documents[:4], {}), (self.documents[:5], {}),
                                   (self.documents, {"max_residual_rounds": 0})]:
            generator = ResidualGenerator([])
            self.model(generator, **options).fit(documents)
            self.assertEqual(generator.calls, [])

    def test_fixed_codebook_does_not_trigger_generation(self):
        generator = ResidualGenerator([RuntimeError("must not call")])
        model = self.model(generator).fit(self.documents, codebook=[topic("weather"), topic("music")])
        self.assertEqual(generator.calls, [])
        self.assertEqual(model.metadata_["generation_calls"], 0)

    def test_residual_error_leaves_previously_fitted_model_unchanged(self):
        generator = ResidualGenerator([RuntimeError("backend failed")])
        model = self.model(generator).fit(self.documents[:4])
        before = model.get_document_info()
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            model.fit(self.documents)
        self.assertTrue(model.get_document_info().equals(before))

    def test_round_limit_rejects_invalid_values(self):
        for value in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                self.model(ResidualGenerator([]), max_residual_rounds=value)

    def test_one_round_override_is_honored(self):
        generator = ResidualGenerator([[topic("finance")], [topic("sport")]])
        model = self.model(generator, max_residual_rounds=1).fit(self.documents)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(model.get_topic_info()), 3)
        self.assertEqual(model.metadata_["residual_stop_reason"], "max_residual_rounds_reached")

    def test_initial_only_saved_models_remain_loadable_without_relabelling(self):
        model = self.model(ResidualGenerator([])).fit(self.documents[:4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model.save(path / "old")
            saved = path / "old/model.json"
            payload = json.loads(saved.read_text())
            old_workflow = "initial-codebook-reference-diffusion-v1"
            payload["workflow"] = payload["metadata"]["workflow"] = old_workflow
            payload["configuration"].pop("max_residual_rounds", None)
            saved.write_text(json.dumps(payload))
            loaded = HybridTopic.load(path / "old", embedding_model=KeywordEmbedder())
            loaded.save(path / "resaved")
            self.assertEqual(json.loads((path / "resaved/model.json").read_text())["workflow"], old_workflow)
            self.assertEqual(loaded.transform(["weather new"])[0].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
