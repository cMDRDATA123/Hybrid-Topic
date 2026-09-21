import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hybrid_topic.backends import (
    AnthropicTopicGenerator,
    OpenAICompatibleTopicGenerator,
    TransformersTopicGenerator,
)
from hybrid_topic.generation_budget import resolve_model_limits


GOOD = json.dumps({
    "topics": [{
        "topic_id": "T1",
        "name": "Weather",
        "definition": "Weather reports",
        "core_keywords": ["rain"],
    }]
})


class ProviderCompatibilityTests(unittest.TestCase):
    def test_known_luna_capabilities_are_resolved_without_manual_context(self):
        limits = resolve_model_limits("gpt-5.6-luna", provider="openai")
        self.assertEqual(limits.context_window, 1_050_000)
        self.assertEqual(limits.max_output_tokens, 128_000)
        self.assertIn("low", limits.reasoning_efforts)
        self.assertEqual(limits.tokenizer_encoding, "o200k_base")

    def test_unknown_model_requires_explicit_context_and_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "context_window"):
            resolve_model_limits("unknown-model", provider="openai")
        limits = resolve_model_limits(
            "unknown-model", provider="openai", context_window=32_768,
            tokenizer_encoding="cl100k_base",
        )
        self.assertEqual(limits.context_window, 32_768)
        self.assertEqual(limits.tokenizer_encoding, "cl100k_base")
        self.assertEqual(limits.source, "explicit_configuration")

    def test_openai_compatible_adapter_records_provider_and_reasoning(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=GOOD, refusal=None),
            )],
            usage=None,
            model="grok-3-mini",
        )
        with patch("openai.OpenAI") as factory:
            factory.return_value.chat.completions.create.return_value = response
            generator = OpenAICompatibleTopicGenerator(
                provider="xai", model="grok-4.6", api_key="secret",
                base_url="https://api.x.ai/v1", context_window=131_072,
                input_token_counter=lambda request: 100, token_count_source="test_counter", reasoning_effort="low",
            )
            result = generator.generate_initial(["rain today"])
            self.assertEqual(result.topics[0].name, "Weather")
            self.assertEqual(generator.generation_config["provider"], "xai")
            self.assertEqual(generator.generation_config["reasoning_effort"], "low")
            self.assertEqual(factory.call_args.kwargs["base_url"], "https://api.x.ai/v1")
            request = factory.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["reasoning_effort"], "low")

    def test_anthropic_adapter_normalizes_text_blocks(self):
        response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=GOOD)],
            usage=SimpleNamespace(input_tokens=12, output_tokens=20),
            model="claude-3-5-sonnet",
        )
        with patch("anthropic.Anthropic") as factory:
            factory.return_value.messages.create.return_value = response
            factory.return_value.messages.count_tokens.return_value = SimpleNamespace(input_tokens=100)
            generator = AnthropicTopicGenerator(
                model="claude-3-5-sonnet", api_key="secret", context_window=200_000,
                response_format_mode="json",
            )
            result = generator.generate_initial(["rain today"])
            self.assertEqual(result.topics[0].topic_id, "T1")
            self.assertEqual(generator.generation_config["provider"], "anthropic")
            request = factory.return_value.messages.create.call_args.kwargs
            self.assertEqual(request["max_tokens"], 16_384)
            self.assertEqual(request["system"], "You produce valid compact JSON for open topic discovery.")

    def test_local_transformers_context_is_read_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 65_536}))
            tokenizer = SimpleNamespace(
                apply_chat_template=lambda messages, **kwargs: "prompt",
                __call__=lambda text, **kwargs: {"input_ids": [[1, 2]]},
                decode=lambda ids, **kwargs: GOOD,
            )
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=65_536))
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
            ):
                generator = TransformersTopicGenerator(str(model_dir))
            self.assertEqual(generator.budget.context_window, 65_536)
            self.assertEqual(generator.generation_config["provider"], "transformers")


if __name__ == "__main__":
    unittest.main()
