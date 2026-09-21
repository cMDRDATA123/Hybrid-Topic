import builtins
import unittest
from unittest.mock import patch

from hybrid_topic.backends import (
    MLXTopicGenerator, OpenAITopicGenerator, SentenceTransformerEmbedder,
)


class OptionalBackendTests(unittest.TestCase):
    def test_missing_backend_explains_the_required_extra(self):
        original_import = builtins.__import__
        cases = [
            ("sentence_transformers", "embeddings", lambda: SentenceTransformerEmbedder("offline")),
            ("openai", "openai", lambda: OpenAITopicGenerator(model="offline", api_key="unused", context_window=32768)),
            ("mlx_lm", "mlx", lambda: MLXTopicGenerator(model="offline", context_window=32768)),
        ]
        for missing, extra, create in cases:
            def without_backend(name, *args, **kwargs):
                if name == missing:
                    raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
                return original_import(name, *args, **kwargs)

            with self.subTest(extra=extra), patch("builtins.__import__", side_effect=without_backend):
                with self.assertRaisesRegex(ImportError, rf"\[\s*{extra}\s*\]"):
                    create()

    def test_broken_transitive_dependency_is_not_misreported_as_missing_extra(self):
        with patch("builtins.__import__", side_effect=ModuleNotFoundError("broken dependency", name="broken")):
            with self.assertRaisesRegex(ModuleNotFoundError, "broken dependency"):
                SentenceTransformerEmbedder("offline")


if __name__ == "__main__":
    unittest.main()
