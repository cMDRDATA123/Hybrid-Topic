"""Current release benchmark: text-only execution, audited requests, cached baselines.

The user authorized 20 cloud + 20 local fits with no monetary ceiling.
Generation errors remain recorded; runs are not replaced according to quality.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import csv
import numpy as np
from hybrid_topic import HybridTopic
from hybrid_topic.backends import OpenAITopicGenerator, MLXTopicGenerator, SentenceTransformerEmbedder
from hybrid_topic.types import _unit_rows
from hybrid_topic.generation_budget import freeze_model_catalog, verify_frozen_model_catalog
from experiments.data_io import read_text_pack, sha256, write_json

SEEDS=[11,23,37,53,71]
DATASETS=['bills','sib200','massive','ag_news']
EMBEDDING='Qwen/Qwen3-Embedding-0.6B'
CLOUD_MODEL='gpt-4.1-mini-2025-04-14'
LOCAL_MODEL='mlx-community/Qwen2.5-7B-Instruct-4bit'
LOCAL_REVISION='c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'

def synchronize():
    import torch
    if torch.backends.mps.is_available():torch.mps.synchronize()


def prediction_file(path,ids,topics,scores=None):
    if len(ids)!=len(topics) or (scores is not None and len(ids)!=len(scores)):
        raise ValueError('prediction length differs from source IDs')
    if scores is not None and not np.isfinite(scores).all():raise ValueError('nonfinite scores')
    with path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['doc_id','topic_id','score'])
        for i,(id,t) in enumerate(zip(ids,topics)):
            if int(t)!=t or t < -1:raise ValueError('invalid topic ID')
            w.writerow([id,int(t),'' if scores is None else float(scores[i])])


class CachedEmbedder:
    model_name=EMBEDDING
    def __init__(self,backend,train_text,train_vectors,test_text,test_vectors):
        self.backend=backend;self.train_text=train_text;self.train_vectors=train_vectors
        self.queries=dict(zip(test_text,test_vectors));self.topics={}
    def encode(self,texts):
        if texts==self.train_text:return self.train_vectors.copy()
        if len(texts)==1 and texts[0] in self.queries:
            return np.asarray([self.queries[texts[0]]])
        # Cache the exact encoding call: changing an initial topic batch into
        # singleton calls can change floating-point embeddings and seed ties.
        key=tuple(texts)
        if key not in self.topics:self.topics[key]=self.backend.encode(texts)
        return self.topics[key].copy()


def input_data(inputs):
    manifest=json.loads((inputs/'manifest.json').read_text());result={}
    for name in DATASETS:
        result[name]={}
        for split in ['train','test']:
            info=manifest['datasets'][name][split];path=inputs/Path(info['input_file']).name
            if sha256(path)!=info['input_sha256']:raise ValueError('input hash mismatch')
            ids,texts=read_text_pack(path)
            if len(ids)!=info['rows']:raise ValueError('input count mismatch')
            result[name][split]=(ids,texts)
    return manifest,result



def atomic_json(path, value):
    temp=path.with_suffix('.writing')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)


class RecordingGenerator:
    """Journal physical requests while preserving the adapter's bounded retries."""
    def __init__(self, backend, folder):
        self.backend=backend;self.folder=folder;folder.mkdir(parents=True,exist_ok=True)
        self.history=[]
        original=backend._request
        def request(prompt):
            path=folder/f'request_{len(list(folder.glob("request_*.json")))+1:04d}.json'
            record={'status':'started','prompt':prompt,'configuration':backend.generation_config,
                    'estimated_input_tokens':backend._count_input(prompt),'started_at':time.time()}
            write_json(path,record)
            start=time.perf_counter()
            try:
                raw,details=original(prompt)
                record.update(details);record.update(status='completed',raw_response=raw)
                return raw,details
            except Exception as exc:
                record.update(status='failed',error_type=type(exc).__name__)
                raise
            finally:
                record['elapsed_seconds']=time.perf_counter()-start
                atomic_json(path,record)
        backend._request=request
    def __getattr__(self,name):return getattr(self.backend,name)
    def generate_initial(self,documents):
        result=self.backend.generate_initial(documents);self.history.append(result);return result
    def generate_residual(self,existing_codebook,residual_documents):
        result=self.backend.generate_residual(existing_codebook,residual_documents);self.history.append(result);return result


def validate_cached_embeddings(cache,name,data):
    path=cache/f'{name}.npz';meta=json.loads((cache/f'{name}.json').read_text())
    if sha256(path)!=meta['npz_sha256']:raise ValueError('embedding file hash mismatch')
    text_hash=hashlib.sha256(json.dumps([data['train'][1],data['test'][1]],ensure_ascii=False).encode()).hexdigest()
    if meta['input_text_sha256']!=text_hash or meta['model']!=EMBEDDING:
        raise ValueError('embedding inputs/model mismatch')
    if (meta['max_seq_length'],meta['train_encoding_batch_size'],meta['test_encoding_batch_size'])!=(512,16,1):
        raise ValueError('embedding preprocessing mismatch')
    with np.load(path) as a:tr,te=a['train'],a['test']
    if len(tr)!=len(data['train'][0]) or len(te)!=len(data['test'][0]) or not(np.isfinite(tr).all() and np.isfinite(te).all()):
        raise ValueError('invalid cached embedding shape/values')
    return tr,te


def prepare_reuse(data_root,cache_root,out):
    _,data=input_data(data_root/'inputs')
    prior=json.loads((cache_root/'preflight.json').read_text())
    if prior['input_manifest_sha256']!=sha256(data_root/'inputs/manifest.json'):
        raise ValueError('new inputs differ from baseline inputs')
    integrity=json.loads((cache_root/'artifact_integrity.json').read_text())
    entries={r['run']:r['files'] for r in integrity['runs']}
    reuse={}
    for name in DATASETS:
        validate_cached_embeddings(cache_root/'embeddings',name,data[name])
        for seed in SEEDS:
            key=f'{name}_seed{seed}';source=cache_root/key
            for file in ['bertopic.json','bertopic_train.csv','bertopic_test.csv']:
                if sha256(source/file)!=entries[key][file]:raise ValueError('baseline hash mismatch')
            reuse[key]={file:entries[key][file] for file in ['bertopic.json','bertopic_train.csv','bertopic_test.csv']}
    return data,reuse


def run_one(path,kind,name,seed,data,tr,te,backend,embedding,cache_root):
    path.mkdir(parents=True,exist_ok=False)
    write_json(path/'started.json',{'backend':kind,'dataset':name,'seed':seed,'started_at':time.time()})
    generator=RecordingGenerator(backend,path/'requests')
    encoder=CachedEmbedder(embedding,data['train'][1],tr,data['test'][1],te)
    started=time.perf_counter()
    try:
        model=HybridTopic(generator=generator,embedding_model=encoder).fit(data['train'][1])
        synchronize();fit_seconds=time.perf_counter()-started
        model.save(path/'model')
        initial=HybridTopic(embedding_model=encoder,max_residual_rounds=0).fit(
            data['train'][1],codebook=generator.history[0].topics)
        initial.save(path/'initial_model')
        for label,m in [('hybrid',model),('initial',initial)]:
            frame=m.get_document_info()
            prediction_file(path/f'{label}_train.csv',data['train'][0],frame.topic_id.to_numpy(),frame.score.to_numpy())
            ids,scores=m.transform(data['test'][1])
            prediction_file(path/f'{label}_test.csv',data['test'][0],ids,scores)
            if label=='hybrid':
                loaded=HybridTopic.load(path/'model',embedding_model=encoder)
                restored=loaded.transform(data['test'][1])
                if not all(np.array_equal(x,y) for x,y in zip(restored,(ids,scores))):
                    raise RuntimeError('full query reload mismatch')
                single=loaded.transform([data['test'][1][0]])
                if single[0][0]!=ids[0] or single[1][0]!=scores[0]:raise RuntimeError('query batch mismatch')
            for split,v in [('train',tr),('test',te)]:
                similarity=np.einsum('id,jd->ij',_unit_rows(v,name='documents'),m._topic_vectors,optimize=False)
                best=similarity.argmax(axis=1)
                prediction_file(path/f'direct_{label}_{split}.csv',data[split][0],best,similarity[np.arange(len(best)),best])
        from sklearn.cluster import KMeans
        km=KMeans(n_clusters=len(model._codebook),n_init=10,random_state=seed).fit(_unit_rows(tr,name='training'))
        prediction_file(path/'kmeans_train.csv',data['train'][0],km.labels_)
        prediction_file(path/'kmeans_test.csv',data['test'][0],km.predict(_unit_rows(te,name='queries')))
        for file in ['bertopic.json','bertopic_train.csv','bertopic_test.csv']:
            shutil.copy2(cache_root/f'{name}_seed{seed}'/file,path/file)
        write_json(path/'complete.json',{'backend':kind,'dataset':name,'seed':seed,
            'model_metadata':model.metadata_,'fit_seconds_excluding_document_embedding':fit_seconds,
            'total_seconds_excluding_document_embedding':time.perf_counter()-started,
            'reload_all_queries_exact':True,'batch_check':True,
            'files':{str(p.relative_to(path)):sha256(p) for p in path.rglob('*') if p.is_file()}})
        print('completed',kind,name,seed,'topics',len(model._codebook),'seconds',round(time.perf_counter()-started,1),flush=True)
    except Exception as exc:
        write_json(path/'failure.json',{'backend':kind,'dataset':name,'seed':seed,'error_type':type(exc).__name__,
                                      'error':str(exc),'seconds':time.perf_counter()-started,
                                      'generation_audit':backend.last_generation_})
        print('FAILED',kind,name,seed,type(exc).__name__,str(exc),flush=True)


def execute(data_root,cache_root,out,api_key_file,local_path):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'preflight.json').exists():raise RuntimeError('This frozen batch has already started; inspect it before resuming')
        data,reuse=prepare_reuse(data_root,cache_root,out)
        sources={str(p):sha256(p) for folder in ['hybrid_topic','experiments'] for p in Path(folder).glob('*.py')}
        model_catalog=freeze_model_catalog(out/'source_snapshot/model_catalog.json')
        preflight={'model_catalog':model_catalog,'protocol':'release-20260907-catalog-v2','seeds':SEEDS,'datasets':DATASETS,'backends':['cloud','local'],
                   'input_manifest_sha256':sha256(data_root/'inputs/manifest.json'),
                   'protocol_sha256':sha256('docs/EXPERIMENT_PROTOCOL_RELEASE.md'),
                   'data_root':str(data_root),'cache_root':str(cache_root), 'baseline_reuse':reuse,
                   'sources':sources,'cloud_model':CLOUD_MODEL,'cloud_context':1047576,
                   'local_model':LOCAL_MODEL,'local_revision':LOCAL_REVISION,'local_context':32768,
                   'no_fee_ceiling_user_authorized':True,'planned_fits':40,
                   'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','openai','tiktoken','mlx-lm','sentence-transformers']}}
        write_json(out/'preflight.json',preflight)
        for p in sources:
            dst=out/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        embedding=SentenceTransformerEmbedder(EMBEDDING,device='mps',local_files_only=True)
        local_backend=None
        for kind in ['cloud','local']:
            if kind=='local':
                local_backend=MLXTopicGenerator(str(local_path),context_window=32768)
            for name in DATASETS:
                d=data[name];tr,te=validate_cached_embeddings(cache_root/'embeddings',name,d)
                for seed in SEEDS:
                    for p,expected in sources.items():
                        if sha256(p)!=expected:raise RuntimeError('frozen execution source changed: '+p)
                    verify_frozen_model_catalog(model_catalog)
                    if kind=='cloud':
                        backend=OpenAITopicGenerator(model=CLOUD_MODEL,api_key=api_key_file.read_text().strip(),
                            context_window=1047576,sampling_seed=seed,catalog_path=model_catalog['path'])
                        if backend.generation_config['catalog_sha256']!=model_catalog['sha256']:
                            raise RuntimeError('frozen model catalog changed during generator configuration')
                        backend.client=backend.client.with_options(timeout=300,max_retries=0)
                    else:
                        import mlx.core as mx
                        mx.random.seed(seed)
                        backend=local_backend
                        backend.sampling_seed=seed;backend.generation_config['sampling_seed']=seed
                        backend.generation_config['mlx_random_seed']=seed
                    # A shared local backend must not accumulate request wrappers across runs.
                    original_request=backend._request
                    try:run_one(out/kind/f'{name}_seed{seed}',kind,name,seed,d,tr,te,backend,embedding,cache_root)
                    finally:backend._request=original_request
                    status={'completed':len(list(out.glob('*/*/complete.json'))),'failed':len(list(out.glob('*/*/failure.json'))),'planned':40}
                    atomic_json(out/'status.json',status)
        completed={str(p.parent.relative_to(out)):sha256(p) for p in out.glob('*/*/complete.json')}
        failures={str(p.parent.relative_to(out)):sha256(p) for p in out.glob('*/*/failure.json')}
        if len(completed)+len(failures)!=40:raise RuntimeError('planned run accounting mismatch')
        write_json(out/'predictions_frozen.json',{'preflight_sha256':sha256(out/'preflight.json'),
                                               'runs':completed,'failures':failures,'planned':40})
        requests=[json.loads(p.read_text()) for p in out.glob('*/*/requests/request_*.json')]
        cloud=[r for r in requests if r['configuration']['model']==CLOUD_MODEL]
        usage=[r['usage'] for r in cloud if r.get('usage')]
        write_json(out/'usage_summary.json',{'cloud_requests':len(cloud),'local_requests':len(requests)-len(cloud),
                   'cloud_prompt_tokens':sum(r['prompt_tokens'] for r in usage),
                   'cloud_completion_tokens':sum(r['completion_tokens'] for r in usage),
                   'cloud_requests_without_usage':len(cloud)-len(usage),
                   'cloud_usd_at_standard_no_discount_rates':sum((r['prompt_tokens']*.4+r['completion_tokens']*1.6)/1e6 for r in usage),
                   'rate_note':'GPT-4.1 mini standard uncached rates; estimate excludes cache discounts and unknown-usage requests'})
        print('FROZEN',len(completed),'completed',len(failures),'failed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--api-key-file',type=Path,default=Path('api.txt'));parser.add_argument('--local-path',type=Path,required=True)
    a=parser.parse_args();execute(a.data,a.cache,a.out,a.api_key_file,a.local_path)
