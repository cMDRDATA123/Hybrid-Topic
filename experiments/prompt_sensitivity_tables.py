"""Rebuild complete compact/neutral comparison tables from saved results and codebooks."""
from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1]
O=R/'paper/results/prompt_sensitivity_20260920';O.mkdir(parents=True,exist_ok=True)
old=pd.read_csv(R/'paper/results/topicgpt_aligned_20260911/per_run_metrics.csv'); old['condition']='compact'
new=pd.read_csv(R/'runs/neutral_prompt_20260919/evaluation/per_run_metrics.csv');new['generator']=new.backend.map({'cloud':'GPT-4.1 mini','luna':'GPT-5.6 Luna'});new['condition']='neutral'
# Match the generator spelling in the existing report.
print('old generators',old.generator.unique());new.loc[new.backend=='luna','generator']=next(x for x in old.generator.unique() if 'Luna' in x)
cols=['condition','generator','dataset','seed','method','n_documents','harmonic_purity','coverage','ari','nmi']
allrows=pd.concat([old[cols],new[cols]],ignore_index=True);assert len(allrows)==480
allrows.to_csv(O/'all_methods_per_run.csv',index=False)
runs=allrows[allrows.method=='hybrid'].copy();counts=[]
for row in runs.itertuples():
 backend='cloud' if row.generator=='GPT-4.1 mini' else 'luna'
 base=R/('runs/neutral_prompt_20260919' if row.condition=='neutral' else f'runs/release_{backend}_20260906')
 model=json.loads((base/backend/f'{row.dataset}_seed{row.seed}/model/model.json').read_text())
 counts.append(len(model['codebook']))
runs['final_topics']=counts;assert len(runs)==80
runs.to_csv(O/'per_run.csv',index=False)
summary=runs.groupby(['condition','generator','dataset']).agg(**{f'{m}_{stat}':(m,stat) for m in ['final_topics','harmonic_purity','coverage','ari','nmi'] for stat in ['mean','std']}).reset_index();summary.to_csv(O/'summary.csv',index=False)
full=allrows.groupby(['condition','generator','dataset','method']).agg(**{f'{m}_{stat}':(m,stat) for m in ['harmonic_purity','coverage','ari','nmi'] for stat in ['mean','std']}).reset_index();full.to_csv(O/'all_methods_summary.csv',index=False)
labels={'bills':'Bills','sib200':'SIB-200','massive':'MASSIVE','agnews':'AG News','ag_news':'AG News'}
table_en='| Generator | Collection | Topics: compact | Topics: neutral | HMP: compact | HMP: neutral |\n|---|---|---:|---:|---:|---:|\n';table_zh='| 生成器 | 语料 | 精简：主题数 | 中性：主题数 | 精简：HMP | 中性：HMP |\n|---|---|---:|---:|---:|---:|\n'
for gen in old.generator.unique():
 for ds in ['bills','sib200','massive',next(x for x in runs.dataset.unique() if x.startswith('ag'))]:
  c=summary[(summary.condition=='compact')&(summary.generator==gen)&(summary.dataset==ds)].iloc[0];n=summary[(summary.condition=='neutral')&(summary.generator==gen)&(summary.dataset==ds)].iloc[0]
  line=f'| {"4.1 mini" if gen=="GPT-4.1 mini" else "5.6 Luna"} | {labels[ds]} | {c.final_topics_mean:.1f} | {n.final_topics_mean:.1f} | {c.harmonic_purity_mean:.4f} | {n.harmonic_purity_mean:.4f} |\n'
  table_en+=line;table_zh+=line
(O/'README.md').write_text('''# Prompt sensitivity comparison, 20 September 2026

Compact remains the main experiment and release default. Neutral is a complete sensitivity comparison: four collections, two generators, five seeds each. Final topic counts include any residual additions and are counted from saved model codebooks.

- `per_run.csv`: 80 final Hybrid runs, topic counts and metrics.
- `summary.csv`: all 16 condition/generator/collection groups; means and sample SD (ddof=1).
- `all_methods_per_run.csv`: 480 configurations, all six methods under both conditions.
- `all_methods_summary.csv`: 96 groups with means and sample SD.

Compact scores come from `paper/results/topicgpt_aligned_20260911/per_run_metrics.csv`; neutral scores come from `runs/neutral_prompt_20260919/evaluation/per_run_metrics.csv`. Both use topicgpt-weighted-best-f1-v1. The baseline rows are reused under the shared protocol; repeated rows across conditions are not new baseline fits. Compact model codebooks are in release_cloud_20260906 and release_luna_20260906; neutral codebooks and the frozen source are in neutral_prompt_20260919. The comparison replaces the whole topic-discovery sentence group and repeats stochastic generation; differences include wording and generation variation. The propagation ablation and historical cost table refer to compact.

For neutral generation reproduction, use the frozen neutral source snapshot and the protocol in docs/EXPERIMENT_PROTOCOL_NEUTRAL_20260919.md. The active neutral runner now checks prompt identity before any paid work.
''')
