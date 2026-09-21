"""Separate post-freeze scoring process. Never imported by model execution."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.data_io import sha256,write_json
from experiments.evaluation_report import build_evaluation_report

METHODS=['hybrid','initial','direct_hybrid','direct_initial','bertopic','kmeans']


def score(data,run,out):
    frozen=json.loads((run/'predictions_frozen.json').read_text())
    for key,expected in frozen.get('failures',{}).items():
        if sha256(run/key/'failure.json')!=expected:raise ValueError('failure record changed')
    manifest=json.loads((data/'data_manifest.json').read_text())
    input_manifest=json.loads((data/'inputs/manifest.json').read_text())
    if sha256(data/'data_manifest.json')!=input_manifest['sealed_data_manifest_sha256']:
        raise ValueError('sealed data/gold manifest changed')
    if sha256(run/'preflight.json')!=frozen['preflight_sha256']:raise ValueError('preflight changed')
    preflight=json.loads((run/'preflight.json').read_text())
    if sha256(data/'inputs/manifest.json')!=preflight['input_manifest_sha256']:raise ValueError('data input manifest changed')
    # Validate every output before reading labels, then score all methods together.
    for key,digest in frozen['runs'].items():
        p=run/key/'complete.json'
        if sha256(p)!=digest:raise ValueError('run completion changed')
        for name,expected in json.loads(p.read_text())['files'].items():
            if sha256(p.parent/name)!=expected:raise ValueError('prediction/model artifact changed')
    out.mkdir(parents=True,exist_ok=False);rows=[];reports={}
    for key in frozen['runs']:
        backend, run_key = key.split('/', 1)
        dataset,seed=run_key.rsplit('_seed',1)
        for split in ['train','test']:
            info=manifest['datasets'][dataset]['splits'][split];gold_path=data/info['gold_file']
            if sha256(gold_path)!=info['gold_sha256']:raise ValueError('gold sidecar changed')
            with gold_path.open() as f: gold=list(csv.DictReader(f))
            ids=[r['doc_id'] for r in gold]
            labels={label:i for i,label in enumerate(sorted({r['label'] for r in gold}))}
            true=np.array([labels[r['label']] for r in gold]);eligible=np.array([r['eligible']=='1' for r in gold])
            for method in METHODS:
                with (run/key/f'{method}_{split}.csv').open() as f:pred=list(csv.DictReader(f))
                if [r['doc_id'] for r in pred]!=ids:raise ValueError('prediction IDs/order differ from gold')
                predicted=np.array([int(r['topic_id']) for r in pred])
                scopes={'eligible':eligible}
                if not eligible.all():scopes['original_including_overlap']=np.ones(len(ids),dtype=bool)
                for scope,mask in scopes.items():
                    report=build_evaluation_report(true[mask],predicted[mask]);reports[f'{key}/{method}/{split}/{scope}']=report
                    row={'backend':backend,'dataset':dataset,'seed':int(seed),'method':method,'split':split,'sample_scope':scope,
                        'evidence_role':'release_revalidation_previously_seen',
                        'n_documents':report['n_documents'],'coverage':report['coverage'],'n_rejected':report['n_rejected'],'n_assigned_topics':report['n_assigned_topics'],**report['all_documents']['metrics']}
                    row.update({'assigned_'+k:v for k,v in (report['assigned_documents']['metrics'] or {}).items()});rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'all_metrics.csv',index=False)
    main=frame[(frame['split']=='test')&(frame['sample_scope']=='eligible')]
    summary=main.groupby(['backend','dataset','evidence_role','method'],sort=False).agg(
        runs=('seed','count'),hmp_mean=('harmonic_purity','mean'),hmp_std=('harmonic_purity','std'),
        coverage_mean=('coverage','mean'),coverage_std=('coverage','std'),ari_mean=('ari','mean'),nmi_mean=('nmi','mean'),
        topics_mean=('n_assigned_topics','mean')).reset_index()
    summary['planned_runs']=5
    summary['missing_or_failed_runs']=5-summary['runs']
    summary.to_csv(out/'test_summary.csv',index=False)
    write_json(out/'reports.json',reports)
    write_json(out/'scoring_manifest.json',{'prediction_freeze_sha256':sha256(run/'predictions_frozen.json'),
        'data_manifest_sha256':sha256(data/'data_manifest.json'),'report_version':'coverage-explicit-external-v1',
        'source_sha256':sha256(__file__),'rows':len(rows),'completed_runs':len(frozen['runs']),'failed_runs':len(frozen.get('failures',{})),'planned_runs':frozen['planned'],
        'metric_source_sha256':sha256('hybrid_topic/evaluation.py'),'report_source_sha256':sha256('experiments/evaluation_report.py')})
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();score(a.data,a.run,a.out)
