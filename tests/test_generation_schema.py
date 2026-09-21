import json,unittest
from types import SimpleNamespace
from unittest.mock import patch
from hybrid_topic.backends import _parse_topics,OpenAITopicGenerator

GOOD={'topic_id':'T1','name':'Sports','definition':'Competitive sports','core_keywords':['tennis']}
class GenerationSchemaTests(unittest.TestCase):
    def test_partial_malformed_output_must_not_silently_lose_topics(self):
        raw=json.dumps({'topics':[GOOD,{'topic_id':'T2','Technology and software':['software']}]})
        with self.assertRaises(ValueError) as caught:_parse_topics(raw,prefix='T')
        self.assertEqual(caught.exception.raw_response,raw)
    def test_missing_topics_and_invalid_field_types_are_errors(self):
        cases=[{},[],{'topics':[3]},{'topics':[{**GOOD,'name':3}]},{'topics':[{**GOOD,'core_keywords':[None]}]},{'topics':[{**GOOD,'core_keywords':[]}]},{'topics':[{**GOOD,'definition':' '}]}]
        for data in cases:
            with self.subTest(data=data),self.assertRaises(ValueError):_parse_topics(json.dumps(data),prefix='T')
    def test_empty_residual_and_well_formed_json_are_supported(self):
        self.assertEqual(_parse_topics('{"topics":[]}',prefix='R'),())
        self.assertEqual(len(_parse_topics(json.dumps({'topics':[GOOD]}),prefix='T')),1)
    def test_openai_requests_schema_and_rejects_truncation(self):
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content=json.dumps({'topics':[GOOD]}),refusal=None))])
        with patch('openai.OpenAI') as factory:
            factory.return_value.chat.completions.create.return_value=response
            gen=OpenAITopicGenerator(model='gpt-4.1-mini',api_key='offline-placeholder',context_window=1_000_000)
            with self.assertRaises(ValueError):gen.generate_initial(['document'])
            fmt=factory.return_value.chat.completions.create.call_args.kwargs['response_format']
            self.assertEqual(fmt['type'],'json_schema');self.assertTrue(fmt['json_schema']['strict'])

if __name__=='__main__':unittest.main()
