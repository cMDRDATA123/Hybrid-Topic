"""Plot audited dataset means; no fitting, selection or significance inference."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,default=Path('paper/results/topicgpt_aligned_20260911/all_methods.csv'))
p.add_argument('--out',type=Path,default=Path('paper/figures'))
a=p.parse_args();f=pd.read_csv(a.data);a.out.mkdir(parents=True,exist_ok=True)
manifest=json.loads(a.data.with_name('manifest.json').read_text())
if manifest.get('metric_version') != 'topicgpt-weighted-best-f1-v1':
 raise ValueError('figure requires TopicGPT-aligned metric manifest')
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
styles=[('GPT-4.1 mini','hybrid','Hybrid / GPT-4.1 mini','#0072B2','o'),
        ('Luna','hybrid','Hybrid / Luna','#D55E00','s'),
        ('GPT-4.1 mini','bertopic','BERTopic (shared)','#333333','^'),
        ('GPT-4.1 mini','kmeans','KMeans / GPT-4.1 mini K','#009E73','x'),
        ('Luna','kmeans','KMeans / Luna K','#CC79A7','+')]
for ax,dataset,title in zip(axes.flat,['bills','sib200','massive','ag_news'],['Bills','SIB-200','MASSIVE','AG News']):
 for generator,method,label,color,marker in styles:
  row=f[(f.generator==generator)&(f.method==method)&(f.dataset==dataset)].iloc[0]
  ax.scatter(row.coverage_mean,row.harmonic_purity_mean,label=label,color=color,marker=marker,s=62)
 ax.set_title(title);ax.grid(alpha=.22);ax.set_xlim(0,1.04);ax.set_ylim(0,1)
 ax.set_xlabel('Coverage');ax.set_ylabel('HMP (TopicGPT definition)')
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
fig.suptitle('Eligible test documents: means across five fixed runs',fontsize=13)
fig.tight_layout(rect=(0,.10,1,.95))
fig.savefig(a.out/'quality_coverage.png',dpi=180)
fig.savefig(a.out/'quality_coverage.pdf')
