"""Prepare text-only inputs and separate gold sidecars; never score predictions."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.data_io import normalized,select_news_rows,sha256,write_json


def prepare(root,source,out,revision):
    out.mkdir(parents=True,exist_ok=False)
    (out/'inputs').mkdir();(out/'gold').mkdir()
    manifest={'version':'v2-inputs-20260906','revision':revision,'datasets':{},'source_files':{}}
    raw={s:pd.read_parquet(source/f'{s}.parquet').to_dict('records') for s in ('train','test')}
    for s in raw: manifest['source_files'][str(source/f'{s}.parquet')]=sha256(source/f'{s}.parquet')
    train,test=select_news_rows(raw['train'],raw['test'])
    # Audit the exact predeclared selection before any model or metric runs.
    normed=[normalized(r['text']) for r in train+test]
    matrix=TfidfVectorizer(analyzer='char',ngram_range=(5,5)).fit_transform(normed)
    similarities=(matrix[1200:]@matrix[:1200].T).tocoo()
    pairs=[]
    def grams(text): return {text[i:i+5] for i in range(max(0,len(text)-4))}
    for i,j,cosine in zip(similarities.row,similarities.col,similarities.data):
        if cosine<0.70: continue
        a,b=grams(normed[1200+i]),grams(normed[j]);jac=len(a&b)/len(a|b) if a|b else 0
        if jac>=0.80: pairs.append({'test_index':int(i),'train_index':int(j),'jaccard':jac,'tfidf_cosine':float(cosine)})
    # Keep all exact-rule inputs. Pre-score eligibility mask is separate, fixed,
    # and disclosed; no replacements drawn for overlap exclusions.
    excluded={r['test_index'] for r in pairs}
    manifest['news_audit']={'selection':'NFKC/casefold/whitespace dedup; sha256 20260906 prefix; 1200/800',
        'train_raw':len(raw['train']),'test_raw':len(raw['test']),
        'cross_split_near_duplicate_rule':'char 5-gram Jaccard >= 0.80 among TF-IDF cosine >= 0.70 candidates',
        'cross_split_near_duplicate_pairs':pairs,'excluded_test_indices':sorted(excluded),
        'eligible_test_count':len(test)-len(excluded),'limitations':'Heuristic candidate screen, not exhaustive paraphrase detection. Original 800 rows retained; no replacement sampling.'}
    data={'ag_news':{'train':train,'test':test}}
    for name,stem in [('bills','topicgpt_bills'),('sib200','sib200_zh_en'),('massive','massive_zh_en')]:
        data[name]={}
        for split in ('train','test'):
            p=root/'data'/f'{stem}_{split}.csv'
            with p.open() as f: rows=list(csv.DictReader(f))
            data[name][split]=rows;manifest['source_files'][str(p)]=sha256(p)
    for name,splits in data.items():
        entry={'evidence_role':'prospective_candidate_pending_exposure_audit' if name=='ag_news' else 'exploratory_previously_exposed','splits':{}}
        for split,rows in splits.items():
            input_path=out/'inputs'/f'{name}_{split}.csv';gold_path=out/'gold'/f'{name}_{split}.csv'
            with input_path.open('x',newline='') as f, gold_path.open('x',newline='') as g:
                writer=csv.writer(f);writer.writerow(['doc_id','text'])
                gw=csv.writer(g);gw.writerow(['doc_id','label','eligible','language'])
                for i,row in enumerate(rows):
                    doc_id=f'{name}/{split}/{i:05d}'
                    writer.writerow([doc_id,row['text']]);gw.writerow([doc_id,row.get('label_text',row.get('label')),int(not(name=='ag_news' and split=='test' and i in excluded)),row.get('language','en')])
            entry['splits'][split]={'rows':len(rows),'input_file':str(input_path.relative_to(out)),'input_sha256':sha256(input_path),'gold_file':str(gold_path.relative_to(out)),'gold_sha256':sha256(gold_path)}
        manifest['datasets'][name]=entry
    write_json(out/'data_manifest.json',manifest)
    # A public/model manifest has no gold paths or hash to tempt accidental reads.
    model_manifest={'version':manifest['version'],'sealed_data_manifest_sha256':sha256(out/'data_manifest.json'),'datasets':{name:{split:{k:v for k,v in info.items() if not k.startswith('gold')} for split,info in e['splits'].items()} for name,e in manifest['datasets'].items()}}
    write_json(out/'inputs'/'manifest.json',model_manifest)
    print(json.dumps({'near_duplicate_pairs':len(pairs),'eligible_news_test':800-len(excluded),'datasets':{name:{split:len(rows) for split,rows in splits.items()} for name,splits in data.items()}},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--revision',required=True)
    args=parser.parse_args();prepare(Path.cwd(),args.source,args.out,args.revision)
