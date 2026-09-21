import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from hybrid_topic.types import GenerationResult, Topic
from experiments.release_benchmark import RecordingGenerator, validate_cached_embeddings

class FakeGenerator:
    generation_config = {'model':'offline','max_output_tokens':16384}
    last_generation_ = None
    def _count_input(self, prompt): return 10
    def _request(self, prompt):
        return 'response', {'usage':{'prompt_tokens':10,'completion_tokens':3},'finish_reason':'stop'}
    def generate_initial(self, documents):
        self._request('private prompt')
        return GenerationResult((Topic('T1','Topic','Definition',('word',)),),'private prompt','response')

class ReleaseRunnerTests(unittest.TestCase):
    def test_every_request_is_recorded_without_a_fee_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen=RecordingGenerator(FakeGenerator(),Path(tmp))
            result=gen.generate_initial(['text'])
            self.assertEqual(gen.history,[result])
            record=json.loads(next(Path(tmp).glob('request_*.json')).read_text())
            self.assertEqual(record['status'],'completed')
            self.assertEqual(record['usage']['prompt_tokens'],10)
            self.assertNotIn('max_usd',record)
    def test_transport_failure_is_preserved_and_never_retried_by_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeGenerator()
            backend._request=lambda prompt: (_ for _ in ()).throw(TimeoutError())
            gen=RecordingGenerator(backend,Path(tmp))
            with self.assertRaises(TimeoutError):gen.generate_initial(['text'])
            records=list(Path(tmp).glob('request_*.json'))
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads(records[0].read_text())['status'],'failed')
    def test_changed_cache_is_rejected_before_model_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bills.npz';p.write_bytes(b'changed')
            (Path(tmp)/'bills.json').write_text(json.dumps({'npz_sha256':'wrong'}))
            with self.assertRaises(ValueError):validate_cached_embeddings(Path(tmp),'bills',{'train':([],[]),'test':([],[])})

    def test_one_complete_fit_writes_auditable_predictions_and_reload_check(self):
        import csv
        from unittest.mock import patch
        from hybrid_topic.backends import OpenAITopicGenerator
        from tests.test_model import WordEmbedder
        from experiments.release_benchmark import run_one
        texts=['weather rain','rain today','music song','music today']
        data={'train':(['a','b','c','d'],texts),'test':(['x','y'],['rain tomorrow','music now'])}
        topics={'topics':[{'topic_id':'T1','name':'Weather','definition':'Weather and rain','core_keywords':['rain']},
                          {'topic_id':'T2','name':'Music','definition':'Music and songs','core_keywords':['music']}]}
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(topics),refusal=None))],usage=None,model='offline')
        with tempfile.TemporaryDirectory() as td, patch('openai.OpenAI') as factory:
            folder=Path(td);cache=folder/'cache/bills_seed11';cache.mkdir(parents=True)
            for split in ['train','test']:
                with (cache/f'bertopic_{split}.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(['doc_id','topic_id','score'])
                    for id in data[split][0]:writer.writerow([id,0,''])
            (cache/'bertopic.json').write_text('{}')
            factory.return_value.chat.completions.create.return_value=response
            gen=OpenAITopicGenerator(model='gpt-4.1-mini',api_key='offline',context_window=1000000)
            embedder=WordEmbedder()
            run_one(folder/'run','cloud','bills',11,data,embedder.encode(texts),embedder.encode(data['test'][1]),gen,embedder,folder/'cache')
            self.assertFalse((folder/'run/failure.json').exists())
            completed=json.loads((folder/'run/complete.json').read_text())
            self.assertTrue(completed['reload_all_queries_exact'])
            self.assertIn('hybrid_test.csv',completed['files'])
            self.assertIn('requests/request_0001.json',completed['files'])
