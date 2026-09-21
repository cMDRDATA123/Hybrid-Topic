"""Export only coded labels, eligibility and predictions from verified local runs.

Original text, label names, source document IDs, embeddings, prompts and keys are
not exported. Original source-file hashes are retained for provenance.
"""
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('paper/results/public_predictions_20260911')
    if out.exists():
        raise FileExistsError(out)
    audit = json.loads(Path('paper/results/topicgpt_formula_20260910/manifest.json').read_text())
    for name, expected in audit['input_hashes'].items():
        if digest(name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    data = Path('data/benchmark_release_20260906')
    manifest = json.loads((data/'data_manifest.json').read_text())
    methods = ['hybrid','initial','direct_hybrid','direct_initial','bertopic','kmeans']
    records, counts, sources = [], {}, {}
    for gen, folder in [('GPT-4.1 mini','release_cloud_20260906'),('Luna','release_luna_20260906')]:
        run = Path('runs')/folder
        frozen = json.loads((run/'predictions_frozen.json').read_text())
        sources[str(run/'predictions_frozen.json')] = digest(run/'predictions_frozen.json')
        for key in frozen['runs']:
            dataset, seed = key.split('/',1)[1].rsplit('_seed',1)
            gold_path = data/manifest['datasets'][dataset]['splits']['test']['gold_file']
            with gold_path.open() as f:gold = list(csv.DictReader(f))
            sources[str(gold_path)] = digest(gold_path)
            labels = {name:i for i,name in enumerate(sorted({r['label'] for r in gold}))}
            counts[dataset] = {'original':len(gold),'eligible':sum(r['eligible']=='1' for r in gold)}
            predicted = {}
            for method in methods:
                path = run/key/f'{method}_test.csv'
                with path.open() as f:pred = list(csv.DictReader(f))
                if [r['doc_id'] for r in pred] != [r['doc_id'] for r in gold]:
                    raise ValueError('Prediction/gold alignment mismatch')
                predicted[method] = [int(r['topic_id']) for r in pred]
                sources[str(path)] = digest(path)
            for i,row in enumerate(gold):
                entry = dict(generator=gen,dataset=dataset,seed=int(seed),doc_index=i,
                             gold=labels[row['label']],eligible=int(row['eligible']))
                entry.update({method:predicted[method][i] for method in methods})
                records.append(entry)
    for name, expected in audit['input_hashes'].items():
        if digest(name) != expected:
            raise ValueError(f'Input changed during export: {name}')
    out.mkdir()
    with (out/'predictions.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (out/'manifest.json').write_text(json.dumps({
        'schema':'public-partition-evidence-v1','counts':counts,'rows':len(records),
        'predictions_sha256':digest(out/'predictions.csv'),'builder_sha256':digest(__file__),
        'scope':'two frozen cloud batches; all test rows retained with original eligibility',
        'gold_encoding':'lexically sorted original label strings mapped to integers per dataset test set',
        'doc_index':'zero-based row position in preserved test input; no source identifiers exported',
        'sources':sources,'limitations':'Supports score reproduction, not raw-text reconstruction or generation reproduction.',
    },indent=2))
    print(f'Exported {len(records)} rows, {len(sources)} source hashes; no source text fields.')


if __name__ == '__main__':
    main()
