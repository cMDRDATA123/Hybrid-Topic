"""Frozen data preparation and integrity helpers; no generator or metric selection."""
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def write_json(path, value):
    """Exclusive artifact creation; never overwrite a previous run."""
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)



def read_text_pack(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['doc_id', 'text']:
            raise ValueError('model input must contain only doc_id,text')
        rows=list(reader)
    ids=[r['doc_id'] for r in rows]; texts=[r['text'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(not i or not t or not t.strip() for i,t in zip(ids,texts)):
        raise ValueError('model input needs unique nonempty IDs and nonempty texts')
    return ids,texts



def select_news_rows(train, test, *, train_n=1200, test_n=800):
    def unique(rows):
        result={}
        for row in rows:
            key=normalized(row['text'])
            if key: result.setdefault(key,row)
        return result
    def ordered(rows):
        return sorted(rows, key=lambda key:hashlib.sha256(('20260906:'+key).encode()).hexdigest())
    tr,te=unique(train),unique(test)
    keys=ordered(tr)[:train_n]
    train_out=[tr[k] for k in keys]
    heldout=[k for k in ordered(te) if k not in set(keys)][:test_n]
    if len(keys)!=train_n or len(heldout)!=test_n:
        raise ValueError('not enough distinct rows for predeclared sample')
    return train_out,[te[k] for k in heldout]

