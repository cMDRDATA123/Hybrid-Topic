"""Supplemental fixed-final-codebook propagation on/off experiment, no API calls."""
from pathlib import Path
import argparse,json,time,hashlib,csv
import numpy as np
import pandas as pd
from hybrid_topic.types import _unit_rows
from hybrid_topic.diffusion_assignment import diffuse_scores
from experiments.data_io import sha256,write_json,read_text_pack
R=Path(__file__).resolve().parents[1]
O=R/'runs/propagation_ablation_20260915'

def seed_matrix(e,k):
    y=np.zeros_like(e)
    for j in range(e.shape[1]):
        idx=np.argsort(e[:,j])[::-1][:k];y[idx,j]=np.maximum(e[idx,j],0)
    return y

def query_assign(q,refs,f,tau,alpha):
    ids=[];values=[]
    for v in q:
        sim=np.einsum('id,d->i',refs,v,optimize=False)
        w=np.where((sim>=tau)&(sim>0),sim,0.);mass=float(w.sum())
        scores=np.einsum('i,ik->k',w,f,optimize=False)/mass if mass>0 else np.zeros(f.shape[1])
        scores=alpha*scores;best=int(scores.argmax());val=float(scores[best])
        ids.append(best if val>0 else -1);values.append(max(val,0.))
    return np.array(ids),np.array(values)

def generate():
    start=time.perf_counter();dest=O/'predictions';dest.mkdir(exist_ok=False);sources={};records=[]
    def checked(p,expected=None):
        h=sha256(p)
        if expected is not None:assert h==expected,str(p)
        sources[str(p.relative_to(R))]=h
    checked(O/'protocol.md');checked(Path(__file__))
    for generator,folder in [('GPT-4.1 mini','release_cloud_20260906'),('Luna','release_luna_20260906')]:
        root=R/'runs'/folder;frozen=json.loads((root/'predictions_frozen.json').read_text());pre=json.loads((root/'preflight.json').read_text());checked(root/'predictions_frozen.json');checked(root/'preflight.json',frozen['preflight_sha256'])
        data=Path(pre['data_root']);cache=Path(pre['cache_root'])/'embeddings';checked(data/'inputs/manifest.json',pre['input_manifest_sha256'])
        for key,digest in frozen['runs'].items():
            path=root/key;checked(path/'complete.json',digest);complete=json.loads((path/'complete.json').read_text())
            for name,h in complete['files'].items():checked(path/name,h)
            ds,seed=key.split('/')[1].rsplit('_seed',1);model=json.loads((path/'model/model.json').read_text());conf=model['configuration'];refs=np.array(model['reference_embeddings']);topics=np.array(model['topic_embeddings']);f=np.array(model['reference_scores']);tau=model['reference_threshold'];alpha=conf['diffusion_rate']
            meta=json.loads((cache/f'{ds}.json').read_text());checked(cache/f'{ds}.json');checked(cache/f'{ds}.npz',meta['npz_sha256'])
            ids,trtext=read_text_pack(data/f'inputs/{ds}_train.csv');testids,tetext=read_text_pack(data/f'inputs/{ds}_test.csv')
            assert hashlib.sha256(json.dumps([trtext,tetext],ensure_ascii=False).encode()).hexdigest()==meta['input_text_sha256']
            with np.load(cache/f'{ds}.npz') as a:tr=a['train'];te=a['test']
            assert np.array_equal(refs,_unit_rows(tr,name='train'))
            # Match historical CachedEncoder singleton dictionary lookup, including duplicates.
            mapping=dict(zip(tetext,te));q=np.array([_unit_rows(np.array([mapping[t]]),name='query')[0] for t in tetext])
            e=np.einsum('id,jd->ij',refs,topics,optimize=False);sim=np.einsum('id,jd->ij',refs,refs,optimize=False)
            y=seed_matrix(e,conf['seed_per_topic']);cfg={k:v for k,v in conf.items() if k!='max_residual_rounds'};replay=diffuse_scores(e,sim,**cfg)
            assert np.allclose(replay.scores,f,rtol=0,atol=1e-12);assert replay.threshold==tau
            original=pd.read_csv(path/'hybrid_test.csv');assert list(original.doc_id)==testids
            on,onscore=query_assign(q,refs,f,tau,alpha);assert np.array_equal(on,original.topic_id.to_numpy())
            off,offscore=query_assign(q,refs,y,tau,alpha)
            name=folder+'_'+ds+'_seed'+seed+'.csv';out=dest/name
            pd.DataFrame({'doc_id':testids,'propagation_off':off,'propagation_on':on,'off_score':offscore,'on_score':onscore}).to_csv(out,index=False)
            records.append({'generator':generator,'dataset':ds,'seed':int(seed),'prediction':str(out.relative_to(O)),'sha256':sha256(out),'gold_root':str(data),'on_predictions_exact':True,'replay_max_abs_error':float(np.abs(replay.scores-f).max()),'topics':len(topics)})
            print(generator,ds,seed,'replayed',flush=True)
    assert len(records)==40
    for p,h in sources.items():assert sha256(R/p)==h
    write_json(O/'predictions_frozen.json',{'protocol_sha256':sha256(O/'protocol.md'),'sources':sources,'runs':records,'seconds':time.perf_counter()-start})

def score():
    from sklearn.metrics.cluster import contingency_matrix
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    frozen=json.loads((O/'predictions_frozen.json').read_text());rows=[]
    for record in frozen['runs']:assert sha256(O/record['prediction'])==record['sha256']
    def metrics(true,pred):
        table=contingency_matrix(true,pred);gc=table.sum(1);pc=table.sum(0);h=float((gc*(2*table/(gc[:,None]+pc[None,:])).max(1)).sum()/table.sum())
        return {'hmp':h,'coverage':float(np.mean(pred>=0)),'ari':adjusted_rand_score(true,pred),'nmi':normalized_mutual_info_score(true,pred)}
    goldhash={}
    for r in frozen['runs']:
        data=Path(r['gold_root']);manifest=json.loads((data/'data_manifest.json').read_text());inputs=json.loads((data/'inputs/manifest.json').read_text());assert sha256(data/'data_manifest.json')==inputs['sealed_data_manifest_sha256']
        info=manifest['datasets'][r['dataset']]['splits']['test'];goldpath=data/info['gold_file'];assert sha256(goldpath)==info['gold_sha256'];goldhash[str(goldpath)]=sha256(goldpath)
        gold=pd.read_csv(goldpath);pred=pd.read_csv(O/r['prediction']);assert list(gold.doc_id)==list(pred.doc_id);eligible=gold.eligible.to_numpy()==1;true=gold.label.to_numpy();off=pred.propagation_off.to_numpy();on=pred.propagation_on.to_numpy();common=eligible&(off>=0)&(on>=0)
        for scope,mask in [('all_eligible',eligible),('common_assigned_diagnostic',common)]:
            if not mask.any():continue
            for method,values in [('off',off),('on',on)]:rows.append({**{k:r[k] for k in ['generator','dataset','seed']},'scope':scope,'method':method,'n':int(mask.sum()),**metrics(true[mask],values[mask])})
    out=O/'evaluation';out.mkdir(exist_ok=False);df=pd.DataFrame(rows);df.to_csv(out/'per_run.csv',index=False)
    summary=df.groupby(['generator','dataset','scope','method']).agg(runs=('seed','count'),hmp=('hmp','mean'),hmp_sd=('hmp','std'),coverage=('coverage','mean'),coverage_sd=('coverage','std'),ari=('ari','mean'),nmi=('nmi','mean')).reset_index();summary.to_csv(out/'summary.csv',index=False)
    pairs=df.pivot(index=['generator','dataset','scope','seed'],columns='method',values=['hmp','coverage','ari','nmi']);delta=pairs.xs('on',axis=1,level=1)-pairs.xs('off',axis=1,level=1);delta.to_csv(out/'paired_deltas.csv');delta.groupby(level=[0,1,2]).agg(['mean','std']).to_csv(out/'paired_summary.csv')
    # Confirm all new on metrics reproduce the current paper metrics, not just raw predictions.
    old=pd.read_csv(R/'paper/results/topicgpt_aligned_20260911/per_run_metrics.csv');old=old[old.method=='hybrid'].set_index(['generator','dataset','seed']);new=df[(df.scope=='all_eligible')&(df.method=='on')].set_index(['generator','dataset','seed'])
    for index,row in new.iterrows():
        assert abs(row.hmp-old.loc[index,'harmonic_purity'])<1e-12;assert abs(row.coverage-old.loc[index,'coverage'])<1e-12
    write_json(out/'verification.json',{'on_predictions_and_current_metrics_reproduced':40,'gold_hashes':goldhash,'prediction_freeze_sha256':sha256(O/'predictions_frozen.json'),'metric_version':'topicgpt-weighted-best-f1-v1'})
    print(summary[summary.scope=='all_eligible'].to_string(index=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['generate','score']);args=a.parse_args();generate() if args.phase=='generate' else score()
