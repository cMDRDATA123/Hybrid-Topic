import unittest
import numpy as np
from types import ModuleType
from unittest.mock import Mock, patch
from hybrid_topic.backends import SentenceTransformerEmbedder

class RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.max_seq_length = 4096
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 3), dtype=np.float32)



class SentenceTransformerEmbedderTests(unittest.TestCase):
    def test_caps_sequence_length_and_uses_small_embedding_batches(self) -> None:
        model = RecordingSentenceTransformer()
        embedder = SentenceTransformerEmbedder("test-model", model=model)

        vectors = embedder.encode(["x"] * 17)

        self.assertEqual(model.max_seq_length, 512)
        self.assertEqual(vectors.shape, (17, 3))
        self.assertEqual(model.calls[0][1]["batch_size"], 16)
        self.assertTrue(model.calls[0][1]["normalize_embeddings"])

    def test_public_embedding_adapter_accepts_device_and_cache_policy(self):
        from hybrid_topic.backends import SentenceTransformerEmbedder as PublicEmbedder

        module = ModuleType("sentence_transformers")
        factory = Mock(return_value=RecordingSentenceTransformer())
        module.SentenceTransformer = factory
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            embedder = PublicEmbedder("test-model", device="cpu", local_files_only=False)
        self.assertEqual(factory.call_args.kwargs["device"], "cpu")
        self.assertFalse(factory.call_args.kwargs["local_files_only"])
        self.assertEqual(embedder.model_name, "test-model")

