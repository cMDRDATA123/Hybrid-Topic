"""Regression contracts for the September 7 audit; no model/API execution."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from hybrid_topic.backends import (AnthropicTopicGenerator, OpenAITopicGenerator,
    OpenAICompatibleTopicGenerator, TransformersTopicGenerator, TopicGenerationError,
    TOPIC_RESPONSE_FORMAT)

GOOD = '{"topics":[{"topic_id":"T1","name":"Rain","definition":"Weather","core_keywords":["rain"]}]}'

class ProviderAuditTests(unittest.TestCase):
    def catalog(self, tmp, provider, model, **entry):
        path = Path(tmp) / 'catalog.json'
        path.write_text(json.dumps({'snapshot_date':'2026-09-07','providers':{
            provider:{'models':{model:{'context_window':65536, **entry}}}}}))
        return path

    def test_claude_counts_actual_schema_request_and_maps_effort_without_tiktoken(self):
        client = Mock()
        client.messages.count_tokens.return_value = NS(input_tokens=100)
        client.messages.create.return_value = NS(stop_reason='end_turn', model='test-claude',
            content=[NS(type='text',text=GOOD)], usage=None)
        with tempfile.TemporaryDirectory() as tmp:
            catalog = self.catalog(tmp, 'anthropic','test-claude', reasoning_efforts=['low','high'])
            with patch.dict(sys.modules, {'tiktoken':None}):
                gen = AnthropicTopicGenerator(model='test-claude',api_key='offline', client=client,
                    catalog_path=catalog, reasoning_effort='low',request_options={'thinking':{'type':'adaptive'}})
                gen.generate_initial(['rain'])
        request = client.messages.create.call_args.kwargs
        counted = client.messages.count_tokens.call_args.kwargs
        self.assertNotIn('metadata', request)
        self.assertEqual(request['output_config']['effort'],'low')
        self.assertEqual(request['output_config']['format']['schema'], TOPIC_RESPONSE_FORMAT['json_schema']['schema'])
        self.assertIn({k:v for k,v in request.items() if k!='max_tokens'},
                      [call.kwargs for call in client.messages.count_tokens.call_args_list])
        self.assertEqual(gen.generation_config['token_count_method'],'anthropic.messages.count_tokens')

    def test_claude_rejects_options_missing_from_installed_sdk_before_counting(self):
        # Strict signatures model the SDK boundary, not permissive Mock kwargs.
        class Messages:
            def create(self, *, model, system, messages, max_tokens, output_config):
                raise AssertionError("generation must not start")
            def count_tokens(self, *, model, system, messages, output_config):
                raise AssertionError("counting must not start")
        for name in ['seed','frequency_penalty','temperature']:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'SDK'):
                AnthropicTopicGenerator(model='unknown',api_key='offline',context_window=65536,
                    client=NS(messages=Messages()),request_options={name:1})

    def test_claude_count_failure_does_not_fall_back_to_guessed_encoding(self):
        client = Mock()
        client.messages.count_tokens.side_effect=RuntimeError('count unavailable')
        gen = AnthropicTopicGenerator(model='unknown-claude',api_key='offline',client=client,context_window=65536)
        with self.assertRaisesRegex(RuntimeError,'count unavailable'):
            gen.generate_initial(['rain'])
        client.messages.create.assert_not_called()

    def test_deepseek_json_and_output_limit_match_provider_and_counter_sees_full_request(self):
        counter = Mock(return_value=100)
        with patch('openai.OpenAI') as factory:
            factory.return_value.chat.completions.create.return_value=NS(choices=[NS(
                message=NS(content=GOOD,refusal=None),finish_reason='stop')],usage=None,model='custom')
            gen = OpenAICompatibleTopicGenerator(provider='deepseek',model='custom',api_key='offline',
                context_window=65536, input_token_counter=counter, token_count_source='verified:test',
                request_options={'temperature':0.2,'extra_body':{'thinking':{'type':'disabled'}}})
            gen.generate_initial(['rain'])
            request = factory.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request['max_tokens'],16384)
        self.assertNotIn('max_completion_tokens',request)
        self.assertEqual(request['response_format'],{'type':'json_object'})
        self.assertEqual(request['extra_body']['thinking'],{'type':'disabled'})
        self.assertEqual(request['temperature'],0.2)
        self.assertEqual(counter.call_args.args[0],request)
        self.assertEqual(gen.response_format_version,'validated-json-fields-v1')

    def test_provider_only_thinking_and_top_k_are_mapped_or_rejected_before_sdk(self):
        with patch('openai.OpenAI'):
            gen = OpenAICompatibleTopicGenerator(provider='deepseek',model='custom',api_key='offline',
                context_window=65536,input_token_counter=lambda request:100,token_count_source='test',
                request_options={'thinking':{'type':'disabled'}})
            self.assertEqual(gen._request_kwargs('prompt')['extra_body']['thinking'],{'type':'disabled'})
            self.assertNotIn('thinking',gen._request_kwargs('prompt'))
            for option in ['thinking','top_k']:
                with self.assertRaises(ValueError):
                    OpenAITopicGenerator(model='gpt-4.1-mini',api_key='offline',request_options={option:1})

    def test_non_openai_encoding_name_alone_is_not_a_verified_counter(self):
        for provider in ['deepseek','xai','ollama','vllm']:
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError,'input_token_counter'):
                OpenAICompatibleTopicGenerator(provider=provider,model='custom',api_key='offline',
                    context_window=65536,tokenizer_encoding='o200k_base')

    def test_cannot_override_budget_or_contract_through_request_options(self):
        for options in [{'max_tokens':1},{'messages':[]},{'stream':True},
                        {'extra_body':{'max_tokens':1}},{'extra_body':{'messages':[]}}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                OpenAITopicGenerator(model='gpt-4.1-mini',api_key='offline',request_options=options)

    def test_known_unsupported_and_unknown_reasoning_are_not_silently_forwarded(self):
        for model in ['gpt-4.1-mini','unknown']:
            with self.subTest(model=model),self.assertRaisesRegex(ValueError,'reasoning_effort'):
                OpenAITopicGenerator(model=model,api_key='offline',context_window=65536,
                    tokenizer_encoding='o200k_base',reasoning_effort='banana')

    def test_catalog_metadata_contains_content_hash_entry_and_override_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.catalog(tmp,'openai','custom',tokenizer_encoding='o200k_base')
            gen=OpenAITopicGenerator(model='custom',api_key='offline',catalog_path=path,context_window=32768)
            cfg=gen.generation_config
            self.assertEqual(cfg['catalog_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(cfg['catalog_entry']['context_window'],65536)
            self.assertEqual(cfg['capability_sources']['context_window'],'explicit_configuration')
            self.assertEqual(cfg['capability_sources']['tokenizer_encoding'],'catalog')

    def test_transformers_only_eos_is_normal_stop_and_avoids_duplicate_special_tokens(self):
        import torch
        for tokens, expected in [([4,5,6],'length'),([4,5,2],'stop'),([4],'unknown')]:
            with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as tmp:
                Path(tmp,'config.json').write_text('{"max_position_embeddings":65536}')
                tokenizer=Mock()
                tokenizer.apply_chat_template.return_value='prompt'
                tokenizer.encode.return_value=[1,1]
                tokenizer.return_value={'input_ids':torch.tensor([[1,1]])}
                tokenizer.decode.return_value=GOOD
                tokenizer.eos_token_id=2
                model=Mock()
                model.device=torch.device('cpu')
                model.generation_config=NS(eos_token_id=2)
                model.generate.return_value=torch.tensor([[1,1]+tokens])
                with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),patch(
                    'transformers.AutoModelForCausalLM.from_pretrained',return_value=model):
                    gen=TransformersTopicGenerator(tmp,max_output_tokens=3)
                    if expected=='stop':
                        gen.generate_initial(['rain'])
                    else:
                        with self.assertRaises(TopicGenerationError):gen.generate_initial(['rain'])
                self.assertEqual(gen.last_generation_['attempts'][0]['finish_reason'],expected)
                self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
                self.assertEqual(model.generate.call_count,1)

if __name__=='__main__': unittest.main()
