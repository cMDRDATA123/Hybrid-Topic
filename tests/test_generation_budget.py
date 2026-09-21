import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hybrid_topic.backends import OpenAITopicGenerator, MLXTopicGenerator, TopicGenerationError
from hybrid_topic.generation_budget import GenerationBudget, pack_documents, resolve_model_limits

GOOD = json.dumps({'topics': [{'topic_id': 'T1', 'name': 'Sports', 'definition': 'Sports events', 'core_keywords': ['tennis']}]})


def response(raw=GOOD, finish='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=raw, refusal=None))], usage=None, model='offline')


class BudgetTests(unittest.TestCase):
    def test_model_capabilities_are_loaded_from_replaceable_catalog(self):
        catalog = {
            'schema_version': 1,
            'snapshot_date': '2099-01-01',
            'providers': {
                'custom': {
                    'models': {
                        'demo-model': {
                            'context_window': 8192,
                            'max_output_tokens': 2048,
                            'tokenizer_encoding': 'cl100k_base',
                            'reasoning_efforts': ['low'],
                            'source': 'test_catalog',
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model_catalog.json'
            path.write_text(json.dumps(catalog))
            limits = resolve_model_limits('demo-model', provider='custom', catalog_path=path)
        self.assertEqual(limits.context_window, 8192)
        self.assertEqual(limits.max_output_tokens, 2048)
        self.assertEqual(limits.tokenizer_encoding, 'cl100k_base')
        self.assertEqual(limits.source, 'test_catalog')

    def test_catalog_model_without_verified_limits_requires_explicit_values(self):
        catalog = {
            'schema_version': 1,
            'providers': {'custom': {'models': {'listed-model': {'status': 'active'}}}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model_catalog.json'
            path.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'verified context_window'):
                resolve_model_limits('listed-model', provider='custom', catalog_path=path)
            limits = resolve_model_limits(
                'listed-model', provider='custom', catalog_path=path,
                context_window=4096, tokenizer_encoding='cl100k_base',
            )
        self.assertEqual(limits.context_window, 4096)
        self.assertEqual(limits.source, 'explicit_configuration')

    def test_partial_catalog_entry_uses_verified_context_and_requires_only_missing_tokenizer(self):
        catalog = {
            'schema_version': 1,
            'providers': {
                'custom': {
                    'models': {
                        'partial-model': {
                            'context_window': 131072,
                            'max_output_tokens': 8192,
                            'source': 'official_context_only',
                        }
                    }
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model_catalog.json'
            path.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'tokenizer_encoding'):
                resolve_model_limits('partial-model', provider='custom', catalog_path=path)
            limits = resolve_model_limits(
                'partial-model', provider='custom', catalog_path=path,
                tokenizer_encoding='cl100k_base',
            )
        self.assertEqual(limits.context_window, 131072)
        self.assertEqual(limits.max_output_tokens, 8192)

    def test_million_context_limits_complete_input_to_75_percent(self):
        budget = GenerationBudget(1_000_000)
        self.assertEqual(budget.input_limit, 750_000)
        self.assertEqual(budget.max_output_tokens, 16384)
        small = GenerationBudget(32768)
        self.assertLessEqual(small.input_limit + small.max_output_tokens + small.safety_margin_tokens, 32768)

    def test_all_documents_and_long_text_are_preserved_when_they_fit(self):
        docs = ['文' * 700 + str(i) for i in range(150)]
        prompt, stats = pack_documents(docs, render=lambda block: 'header\n' + block, count_tokens=len,
                                       budget=GenerationBudget(1_000_000))
        self.assertEqual(stats['selected_document_indices'], list(range(150)))
        self.assertIn(docs[-1], prompt)
        self.assertEqual(stats['sampling_fraction'], 1)

    def test_sampling_is_repeatable_and_complete_request_stays_in_budget(self):
        docs = ['x' * 100 + str(i) for i in range(100)]
        kwargs = dict(render=lambda block: 'topic definitions\n' + block, count_tokens=len,
                      budget=GenerationBudget(4096, max_output_tokens=1024), sampling_seed=23)
        prompt, stats = pack_documents(docs, **kwargs)
        self.assertEqual((prompt, stats), pack_documents(docs, **kwargs))
        self.assertGreater(stats['selected_document_count'], 0)
        self.assertLess(stats['selected_document_count'], len(docs))
        self.assertLessEqual(len(prompt), kwargs['budget'].input_limit)
        self.assertEqual(len(set(stats['selected_document_indices'])), stats['selected_document_count'])

    def test_oversized_fixed_prompt_and_document_fail_without_silent_truncation(self):
        budget = GenerationBudget(4096, max_output_tokens=1024)
        with self.assertRaisesRegex(ValueError, 'fixed|prompt'):
            pack_documents(['a'], render=lambda block: 'x' * 4000 + block, count_tokens=len, budget=budget)
        with self.assertRaisesRegex(ValueError, 'document'):
            pack_documents(['x' * 5000], render=lambda block: block, count_tokens=len, budget=budget)

    def test_model_limits_use_explicit_context_and_validate_known_upper_bound(self):
        with self.assertRaisesRegex(ValueError, 'context_window'):
            resolve_model_limits('unknown-model')
        known = resolve_model_limits('gpt-4.1-mini-2025-04-14')
        self.assertEqual(known.context_window, 1047576)
        self.assertEqual(known.source, 'provider_model_catalog')
        known = resolve_model_limits('gpt-4.1-mini-2025-04-14', context_window=1_000_000)
        self.assertEqual(known.context_window, 1_000_000)
        self.assertEqual(known.max_output_tokens, 32768)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            resolve_model_limits('gpt-4.1-mini', context_window=2_000_000)
        self.assertEqual(resolve_model_limits(
            'custom', context_window=100000, tokenizer_encoding='cl100k_base'
        ).context_window, 100000)

    def test_invalid_configuration_does_not_produce_negative_budget(self):
        for kwargs in ({'context_window': 0}, {'context_window': True},
                       {'context_window': 32000, 'input_context_ratio': 1},
                       {'context_window': 32000, 'max_output_tokens': -1},
                       {'context_window': 4096}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationBudget(**kwargs)


class GeneratorBudgetTests(unittest.TestCase):
    def test_context_window_is_required_at_model_configuration(self):
        auto = OpenAITopicGenerator(model='gpt-4.1-mini', api_key='offline')
        self.assertEqual(auto.budget.context_window, 1047576)
        with self.assertRaises(ValueError):
            MLXTopicGenerator('local-model')

    def test_openai_default_output_limit_and_bounded_format_repair(self):
        with patch('openai.OpenAI') as factory:
            create = factory.return_value.chat.completions.create
            create.side_effect = [response('{'), response()]
            gen = OpenAITopicGenerator(model='gpt-4.1-mini', api_key='offline-placeholder', context_window=1_000_000)
            result = gen.generate_initial(['x' * 900 + str(i) for i in range(150)])
            self.assertEqual(len(result.topics), 1)
            self.assertEqual(create.call_count, 2)
            first, second = [call.kwargs for call in create.call_args_list]
            self.assertEqual(first['max_completion_tokens'], 16384)
            self.assertIn('D149:', first['messages'][1]['content'])
            self.assertTrue(second['messages'][1]['content'].startswith(first['messages'][1]['content']))
            self.assertEqual(result.metadata['packing']['selected_document_count'], 150)
            self.assertEqual(len(result.metadata['attempts']), 2)
            self.assertEqual(factory.call_args.kwargs['max_retries'], 0)

    def test_retry_override_and_truncation_are_not_unbounded(self):
        for retries in [0, 2]:
            with self.subTest(retries=retries), patch('openai.OpenAI') as factory:
                create = factory.return_value.chat.completions.create
                create.return_value = response('{')
                gen = OpenAITopicGenerator(model='gpt-4.1-mini', api_key='offline', context_window=1_000_000, max_format_retries=retries)
                with self.assertRaises(TopicGenerationError):
                    gen.generate_initial(['document'])
                self.assertEqual(create.call_count, retries + 1)
                self.assertEqual(len(gen.last_generation_['attempts']), retries + 1)
        with patch('openai.OpenAI') as factory:
            create = factory.return_value.chat.completions.create
            create.return_value = response(GOOD, 'length')
            gen = OpenAITopicGenerator(model='gpt-4.1-mini', api_key='offline', context_window=1_000_000)
            with self.assertRaises(TopicGenerationError):
                gen.generate_initial(['document'])
            self.assertEqual(create.call_count, 1)

    def test_model_output_limit_caps_requested_tokens(self):
        with patch('openai.OpenAI'):
            gen = OpenAITopicGenerator(model='gpt-4o-mini', api_key='offline', context_window=128000, max_output_tokens=32768)
            self.assertEqual(gen.budget.max_output_tokens, 16384)

    def test_fenced_invalid_response_raises_typed_error(self):
        from hybrid_topic.backends import _parse_topics
        with self.assertRaises(TopicGenerationError):
            _parse_topics('```', prefix='T')

    @staticmethod
    def mlx_response(finish='stop'):
        return SimpleNamespace(text=GOOD, finish_reason=finish, prompt_tokens=180,
                               generation_tokens=60, peak_memory=1.0)

    def test_mlx_reports_truncation_without_format_retry(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda msgs, **kwargs: json.dumps(msgs),
                                    encode=lambda text, **kwargs: list(text))
        with patch('mlx_lm.load', return_value=(object(), tokenizer)), patch(
            'mlx_lm.stream_generate', return_value=iter([self.mlx_response('length')])
        ) as generate:
            gen = MLXTopicGenerator('offline', context_window=32768)
            with self.assertRaises(TopicGenerationError):
                gen.generate_initial(['document'])
            self.assertEqual(generate.call_count, 1)

    def test_saved_model_keeps_generation_audit_without_raw_text(self):
        from hybrid_topic import HybridTopic
        from tests.test_model import WordEmbedder
        with patch('openai.OpenAI') as factory, tempfile.TemporaryDirectory() as tmp:
            factory.return_value.chat.completions.create.return_value = response()
            gen = OpenAITopicGenerator(model='gpt-4.1-mini', api_key='offline', context_window=1_000_000)
            model = HybridTopic(generator=gen, embedding_model=WordEmbedder(), max_residual_rounds=0)
            weather_topic = json.dumps({'topics': [{'topic_id': 'T1', 'name': 'Weather',
                'definition': 'Weather and rain', 'core_keywords': ['rain']}]})
            factory.return_value.chat.completions.create.return_value = response(weather_topic)
            model.fit(['weather rain secret_document_a', 'rain today secret_document_b'])
            history = model.metadata_['generation_history']
            self.assertEqual(history[0]['packing']['selected_document_count'], 2)
            self.assertEqual(history[0]['configuration']['context_window'], 1_000_000)
            model.save(tmp)
            saved = Path(tmp, 'model.json').read_text()
            self.assertNotIn('secret_document_a', saved)
            self.assertNotIn('raw_response', saved)
            loaded = HybridTopic.load(tmp, embedding_model=WordEmbedder())
            self.assertEqual(loaded.metadata_['generation_history'], history)

    def test_mlx_uses_configured_context_and_current_sampler_api(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda msgs, **kwargs: json.dumps(msgs),
                                    encode=lambda text, **kwargs: list(text))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'config.json').write_text('{"max_position_embeddings":32768}')
            with patch('mlx_lm.load', return_value=(object(), tokenizer)), patch('mlx_lm.stream_generate', return_value=iter([self.mlx_response()])) as generate:
                gen = MLXTopicGenerator(tmp, context_window=32768)
                result = gen.generate_initial(['document'])
                self.assertEqual(gen.budget.context_window, 32768)
                self.assertEqual(generate.call_args.kwargs['max_tokens'], 16384)
                self.assertIn('sampler', generate.call_args.kwargs)
                self.assertNotIn('temp', generate.call_args.kwargs)
                self.assertEqual(len(result.topics), 1)


if __name__ == '__main__':
    unittest.main()
