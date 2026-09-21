"""Recompute paper metrics from the compact, text-free prediction bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix

from hybrid_topic.evaluation import evaluate
from experiments.paper_tables import DATASETS, METHODS, SEEDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_bundle(bundle, out):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest['schema'] != 'public-partition-evidence-v1':
        raise ValueError('unsupported bundle schema')
    source = bundle/'predictions.csv'
    if digest(source) != manifest['predictions_sha256']:
        raise ValueError('prediction hash mismatch')
    frame = pd.read_csv(source)
    columns = {'generator','dataset','seed','doc_index','gold','eligible',*METHODS}
    if set(frame.columns) != columns or frame.isna().any().any():
        raise ValueError('missing or unexpected prediction columns/values')
    for name in columns-{'generator','dataset'}:
        if frame[name].dtype.kind not in 'iu':
            raise ValueError(f'integer values required: {name}')
    if not frame.eligible.isin([0,1]).all() or (frame.gold < 0).any() or (frame[METHODS] < -1).any().any():
        raise ValueError('invalid label or eligibility value')
    expected = {(g,d,s) for g in ['GPT-4.1 mini','Luna'] for d in DATASETS for s in SEEDS}
    groups = frame.groupby(['generator','dataset','seed'],sort=False)
    if set(groups.groups) != expected:
        raise ValueError('planned run accounting mismatch')
    rows = []
    reference = {}
    for (gen,dataset,seed), group in groups:
        group = group.sort_values('doc_index')
        count = manifest['counts'][dataset]
        if list(group.doc_index) != list(range(count['original'])):
            raise ValueError('missing or duplicate document index')
        gold_mask = group[['doc_index','gold','eligible']].to_numpy()
        if dataset in reference and not np.array_equal(reference[dataset],gold_mask):
            raise ValueError('gold or eligibility changes between runs')
        reference[dataset] = gold_mask
        eligible = group[group.eligible == 1]
        if len(eligible) != count['eligible'] or not len(eligible):
            raise ValueError('eligible document count mismatch')
        true = eligible.gold.to_numpy()
        for method in METHODS:
            pred = eligible[method].to_numpy()
            values = evaluate(true,pred)
            table = contingency_matrix(true,pred)
            gold_sizes = table.sum(axis=1)
            pair_f1 = 2*table / (gold_sizes[:,None]+table.sum(axis=0)[None,:])
            weighted = float(np.sum(gold_sizes*pair_f1.max(axis=1))/table.sum())
            rows.append(dict(generator=gen,dataset=dataset,seed=seed,method=method,
                             n_documents=len(true),coverage=float(np.mean(pred != -1)),
                             harmonic_purity=values['harmonic_purity'],ari=values['ari'],nmi=values['nmi'],
                             topicgpt_weighted_best_f1=weighted))
    result = pd.DataFrame(rows)
    out.mkdir(parents=True)
    result.to_csv(out/'per_run_metrics.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({
        'bundle_manifest_sha256':digest(bundle/'manifest.json'),
        'predictions_sha256':digest(source),'scorer_sha256':digest(__file__),
        'legacy_metric_source_sha256':digest(Path(__file__).resolve().parents[1]/'hybrid_topic/evaluation.py'),
        'scope':'eligible test; -1 shared partition; both definitions preserved in separate columns',
        'current_primary_metric':'topicgpt_weighted_best_f1',
        'metric_version':'topicgpt-weighted-best-f1-v1',
        'legacy_column':'harmonic_purity remains the historical global HMP; aligned table builder maps the current primary explicitly',
        'rows':len(result),'output_sha256':digest(out/'per_run_metrics.csv'),
    },indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = score_bundle(args.bundle,args.out)
    print(f'Rescored {len(result)} configurations from document predictions.')
