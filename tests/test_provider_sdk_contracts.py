"""Serialize through real SDKs with HTTP replaced; no provider credentials."""
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from hybrid_topic.backends import AnthropicTopicGenerator, OpenAICompatibleTopicGenerator

GOOD = {'topics':[{'topic_id':'T1','name':'Rain','definition':'Weather','core_keywords':['rain']}]}

class SDKContractTests(unittest.TestCase):
    def test_anthropic_real_sdk_serializes_schema_for_generation_and_counting(self):
        import anthropic
        if 'output_config' not in inspect.signature(anthropic.resources.messages.Messages.count_tokens).parameters:
            self.skipTest('requires supported anthropic>=1.4.0; run clean extra environment')
        import httpx2 as httpx
        requests=[]
        def respond(request):
            requests.append((request.url.path,json.loads(request.content)))
            if request.url.path.endswith('/count_tokens'):
                return httpx.Response(200,json={'input_tokens':100})
            return httpx.Response(200,json={'id':'offline','type':'message','role':'assistant',
                'model':'claude-sonnet-4-6','content':[{'type':'text','text':json.dumps(GOOD)}],
                'stop_reason':'end_turn','stop_sequence':None,'usage':{'input_tokens':100,'output_tokens':40}})
        client=anthropic.Anthropic(api_key='offline',http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            generator=AnthropicTopicGenerator(model='claude-sonnet-4-6',api_key='offline',
                context_window=200000,client=client,reasoning_effort='low')
            result=generator.generate_initial(['rain'])
            self.assertEqual(result.topics[0].name,'Rain')
            create=[body for path,body in requests if not path.endswith('/count_tokens')]
            counts=[body for path,body in requests if path.endswith('/count_tokens')]
            self.assertEqual(len(create),1)
            self.assertEqual(create[0]['output_config']['effort'],'low')
            self.assertTrue(all(body['output_config']==create[0]['output_config'] for body in counts))
            self.assertNotIn('metadata',create[0])
        finally:client.close()

    @unittest.skipUnless(importlib.util.find_spec('openai'),'OpenAI extra not installed')
    def test_deepseek_real_openai_sdk_merges_thinking_into_wire_body(self):
        import openai
        import httpx
        requests=[]
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,
                'model':'deepseek-v4-flash','choices':[{'index':0,'finish_reason':'stop',
                    'message':{'role':'assistant','content':json.dumps(GOOD)}}]})
        gen=OpenAICompatibleTopicGenerator(provider='deepseek',model='deepseek-v4-flash',api_key='offline',
            input_token_counter=lambda request:100,token_count_source='offline-test',
            request_options={'extra_body':{'thinking':{'type':'disabled'}},'temperature':0.1})
        gen.client.close()
        gen.client=openai.OpenAI(api_key='offline',base_url='https://example.invalid/v1',
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        try:
            gen.generate_initial(['rain'])
            body=requests[0]
            self.assertEqual(body['thinking'],{'type':'disabled'})
            self.assertNotIn('extra_body',body)
            self.assertEqual(body['response_format'],{'type':'json_object'})
            self.assertEqual(body['max_tokens'],16384)
            self.assertNotIn('max_completion_tokens',body)
        finally:gen.client.close()

if __name__=='__main__':unittest.main()
