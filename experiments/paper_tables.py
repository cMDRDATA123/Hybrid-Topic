"""Build release-paper tables from audited metric CSVs, without generation.

Input directories contain evaluation/all_metrics.csv and its scoring manifest.
Never mixes earlier Hybrid runs, filters seeds, or selects the best backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

DATASETS=['bills','sib200','massive','ag_news']
METHODS=['hybrid','initial','direct_hybrid','direct_initial','bertopic','kmeans']
SEEDS=[11,23,37,53,71]
METRICS=['harmonic_purity','coverage','ari','nmi']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize(frame):
    expected={(d,m,s) for d in DATASETS for m in METHODS for s in SEEDS}
    keys=list(frame[['dataset','method','seed']].itertuples(index=False,name=None))
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('planned dataset/method/seed accounting mismatch')
    if frame[METRICS].isna().any().any():raise ValueError('missing metric value')
    summary=frame.groupby(['dataset','method'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index();summary['runs']=5
    return summary

def build(cloud,luna,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summaries=[];all_runs=[];manifest={'inputs':{},'metric_definitions':'HMP primary; coverage alongside; ARI/NMI secondary',
      'scope':'eligible test documents; five fixed seeds; release revalidation on previously seen data'}
    for label,root in [('GPT-4.1 mini',Path(cloud)),('Luna',Path(luna))]:
        source=root/'evaluation/all_metrics.csv'
        frame=pd.read_csv(source)
        frame=frame[(frame['split']=='test')&(frame['sample_scope']=='eligible')].copy()
        summary=summarize(frame);summary.insert(0,'generator',label)
        frame.insert(0,'generator',label)
        summaries.append(summary);all_runs.append(frame)
        manifest['inputs'][label]={'metrics_sha256':digest(source),
           'scoring_manifest_sha256':digest(root/'evaluation/scoring_manifest.json'),
           'prediction_freeze_sha256':digest(root/'predictions_frozen.json')}
    return write_tables(summaries,all_runs,manifest,out)


def build_from_per_run(source,out):
    """Public table reproduction using distributed numeric rows, no raw text."""
    frame=pd.read_csv(source)
    if set(frame.generator)!= {'GPT-4.1 mini','Luna'}:raise ValueError('both frozen generator batches are required')
    summaries=[];runs=[]
    for label in ['GPT-4.1 mini','Luna']:
        rows=frame[frame.generator==label].copy()
        summary=summarize(rows);summary.insert(0,'generator',label)
        summaries.append(summary);runs.append(rows)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    return write_tables(summaries,runs,{'numeric_input_sha256':digest(source),
        'scope':'table-only reproduction from published eligible test metric rows'},out)


def write_tables(summaries,all_runs,manifest,out):
    result=pd.concat(summaries,ignore_index=True)
    result.to_csv(out/'all_methods.csv',index=False)
    pd.concat(all_runs,ignore_index=True).to_csv(out/'per_run_metrics.csv',index=False)
    lines=['# Frozen release comparisons','',
      'Eligible test documents; each cell is mean ± sample SD across five fixed runs. '
      'BERTopic results are reused across the two batches, not independent repeats. '
      'KMeans is refitted with K equal to each run’s final topic count.','']
    for label in ['GPT-4.1 mini','Luna']:
        lines += ['## '+label,'','| Dataset | Method | HMP | Coverage | ARI | NMI |',
                  '|---|---|---:|---:|---:|---:|']
        for dataset in DATASETS:
            for method in METHODS:
                row=result[(result.generator==label)&(result.dataset==dataset)&(result.method==method)].iloc[0]
                values=[f"{row[m+'_mean']:.4f} ± {row[m+'_std']:.4f}" for m in METRICS]
                lines.append('| '+' | '.join([dataset,method,*values])+' |')
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    manifest['builder_sha256']=digest(__file__)
    manifest['outputs']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud',type=Path);p.add_argument('--luna',type=Path)
    p.add_argument('--per-run',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.per_run:
        if a.cloud or a.luna:p.error('choose per-run or cloud/luna inputs')
        build_from_per_run(a.per_run,a.out)
    elif a.cloud and a.luna:build(a.cloud,a.luna,a.out)
    else:p.error('provide --per-run or both --cloud and --luna')
