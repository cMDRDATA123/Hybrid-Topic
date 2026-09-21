import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from hybrid_topic import HybridTopic
from hybrid_topic.types import GenerationResult, Topic


class WordEmbedder:
    model_name = "offline-word-embedding-v1"

    def encode(self, texts):
        vectors = []
        for text in texts:
            words = text.lower()
            vectors.append([
                float("rain" in words or "weather" in words),
                float("music" in words or "song" in words),
                float("finance" in words or "tax" in words),
            ])
        return np.asarray(vectors)


class StaticGenerator:
    def __init__(self):
        self.calls = 0

    def generate_initial(self, documents):
        self.calls += 1
        return GenerationResult((
            Topic("T1", "Weather", "Weather and rain", ("rain", "weather")),
            Topic("T2", "Music", "Music and songs", ("music", "song")),
        ), "offline prompt", "offline response")

    def generate_residual(self, existing_codebook, residual_documents):
        return GenerationResult((), "offline residual prompt", "no new topics")


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.documents = ["weather rain", "rain today", "music song", "music today", "finance tax"]
        self.generator = StaticGenerator()
        self.model = HybridTopic(
            generator=self.generator, embedding_model=WordEmbedder(), p=0.95,
            seed_per_topic=1,
        )

    def test_fit_produces_inspectable_assignments_and_preserves_rejection(self):
        topics, scores = self.model.fit_transform(self.documents)
        self.assertEqual(topics.tolist(), [0, 0, 1, 1, -1])
        self.assertEqual(scores.shape, (5,))
        self.assertEqual(self.generator.calls, 1)
        self.assertEqual(self.model.get_topic_info()["count"].tolist(), [2, 2])
        self.assertEqual(self.model.get_document_info(topic_id=-1)["text"].tolist(), ["finance tax"])
        self.assertEqual(self.model.get_topic(0)["name"], "Weather")
        self.assertEqual(self.model.metadata_["workflow"], "bounded-residual-reference-diffusion-v2")

    def test_bad_input_or_unfitted_queries_do_not_call_generator(self):
        for bad in ([], [""], [None], ["rain"]):
            with self.assertRaises(ValueError):
                self.model.fit(bad)
        self.assertEqual(self.generator.calls, 0)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            self.model.get_topic_info()

    def test_failed_refit_does_not_replace_previous_model(self):
        self.model.fit(self.documents)
        old = self.model.get_document_info()
        # The fake embedder produces a zero vector for this unsupported text.
        with self.assertRaisesRegex(ValueError, "zero"):
            self.model.fit(["unsupported", "unsupported"])
        self.assertTrue(old.equals(self.model.get_document_info()))

    def test_explicit_codebook_can_be_used_without_generator(self):
        model = HybridTopic(embedding_model=WordEmbedder(), p=0.95, seed_per_topic=1)
        codebook = StaticGenerator().generate_initial([]).topics
        model.fit(self.documents, codebook=codebook)
        self.assertEqual(model.get_topic_info()["count"].tolist(), [2, 2])
        with self.assertRaisesRegex(ValueError, "generator"):
            HybridTopic(embedding_model=WordEmbedder(), p=0.95).fit(self.documents)

    def test_save_preserves_topic_state_without_backend_or_raw_text(self):
        self.model.fit(self.documents)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model"
            self.model.save(path)
            payload = json.loads((path / "model.json").read_text())
            self.assertEqual(payload["schema"], "hybrid_topic_model_v1")
            self.assertEqual(payload["embedding_model_id"], WordEmbedder.model_name)
            self.assertEqual(len(payload["topic_embeddings"]), 2)
            self.assertNotIn("offline response", (path / "model.json").read_text())
            self.assertNotIn("finance tax", (path / "model.json").read_text())
            with self.assertRaises(FileExistsError):
                self.model.save(path)

    def test_loaded_model_recovers_topics_and_rejects_different_embedding_backend(self):
        self.model.fit(self.documents)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            loaded = HybridTopic.load(tmp, embedding_model=WordEmbedder())
            self.assertTrue(loaded.get_topic_info().equals(self.model.get_topic_info()))
            with self.assertRaisesRegex(RuntimeError, "training documents"):
                loaded.get_document_info()
            with self.assertRaisesRegex(ValueError, "embedding"):
                HybridTopic.load(tmp, embedding_model=WordEmbedder(), embedding_model_id="different")

    def test_diffusion_configuration_is_validated_before_generation(self):
        for kwargs in ({"p": 1.0}, {"p": 0.95, "seed_per_topic": 0}, {"p": 0.95, "rounds": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridTopic(generator=self.generator, embedding_model=WordEmbedder(), **kwargs)

    def test_transform_is_independent_of_batch_composition_order_and_duplicates(self):
        self.model.fit(self.documents)
        before = self.model.get_document_info()
        alone_ids, alone_scores = self.model.transform(["rain tomorrow"])
        together_ids, together_scores = self.model.transform(["music tomorrow", "rain tomorrow", "finance tax", "rain tomorrow"])
        self.assertEqual(alone_ids.tolist(), [0])
        self.assertEqual(together_ids.tolist(), [1, 0, -1, 0])
        self.assertEqual(alone_scores[0], together_scores[1])
        self.assertEqual(alone_scores[0], together_scores[3])
        self.assertEqual(self.generator.calls, 1)
        self.assertTrue(before.equals(self.model.get_document_info()))
        inspected = self.model.transform_result(["finance tax"])
        self.assertEqual(inspected.get_document_info()["topic_id"].tolist(), [-1])

    def test_reference_predictions_survive_save_load_without_a_generator(self):
        self.model.fit(self.documents)
        texts = ["rain tomorrow", "song tomorrow", "finance tax"]
        expected_ids, expected_scores = self.model.transform(texts)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            loaded = HybridTopic.load(tmp, embedding_model=WordEmbedder())
            actual_ids, actual_scores = loaded.transform(texts)
            np.testing.assert_array_equal(actual_ids, expected_ids)
            np.testing.assert_array_equal(actual_scores, expected_scores)
            payload = json.loads((Path(tmp) / "model.json").read_text())
            self.assertEqual(len(payload["reference_embeddings"]), len(self.documents))
            self.assertNotIn("adjacency", payload)

    def test_transform_encodes_each_query_in_isolation(self):
        class BatchSensitiveEmbedder(WordEmbedder):
            def __init__(self):
                self.calls = []

            def encode(self, texts):
                self.calls.append(list(texts))
                vectors = super().encode(texts)
                # Exaggerate a batch/padding-dependent encoder discrepancy.
                return vectors[:, [1, 0, 2]] if len(texts) > 1 else vectors

        self.model.fit(self.documents)
        encoder = BatchSensitiveEmbedder()
        self.model.embedding_model = encoder
        single = self.model.transform(["rain tomorrow"])
        mixed = self.model.transform(["song tomorrow", "rain tomorrow", "finance tax"])
        self.assertEqual(single[0][0], mixed[0][1])
        self.assertEqual(single[1][0], mixed[1][1])
        self.assertTrue(all(len(call) == 1 for call in encoder.calls))

    def test_reference_shape_corruption_is_rejected_on_load(self):
        self.model.fit(self.documents)
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(tmp)
            path = Path(tmp) / "model.json"
            payload = json.loads(path.read_text())
            payload["reference_scores"] = [[1.0]]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "reference"):
                HybridTopic.load(tmp, embedding_model=WordEmbedder())

    def test_high_dimensional_finite_embeddings_do_not_emit_arithmetic_warnings(self):
        class HighDimensionalEmbedder:
            def encode(self, texts):
                return np.random.default_rng(42).normal(size=(len(texts), 1024))

        model = HybridTopic(embedding_model=HighDimensionalEmbedder(), generator=StaticGenerator(), p=0.95)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit([f"document {i}" for i in range(60)])
            _, scores = model.transform(["query"])
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
