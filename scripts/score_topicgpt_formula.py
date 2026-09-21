"""Supplementary, post-hoc formula comparison on frozen eligible test predictions.

Run from the repository root. Does not fit models or replace the primary metric.
The output directory is created once and never overwritten.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_topic.evaluation import harmonic_purity


METHODS = ['hybrid', 'initial', 'direct_hybrid', 'direct_initial', 'bertopic', 'kmeans']
DATASETS = ['bills', 'sib200', 'massive', 'ag_news']
SOURCE = Path('external/topicGPT/topicgpt_python/utils.py')
SOURCE_HASH = 'cea2c8a393449b85a1288f85df350ac3fa9f986982eb66b19d04c65821b0a52e'
OUT = Path('paper/results/topicgpt_formula_20260910')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected, records):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Changed evidence: {path}')
    records[str(path)] = actual


def independent_score(true, pred):
    # Pair F1 = 2 * intersection / (gold size + predicted size).
    # Validate the extracted author's function with an independent scalar form.
    terms = []
    for label in set(true):
        gold = true == label
        best = max(2 * np.count_nonzero(gold & (pred == topic)) /
                   (np.count_nonzero(gold) + np.count_nonzero(pred == topic))
                   for topic in set(pred))
        terms.append(np.count_nonzero(gold) * best)
    return sum(terms) / len(true)


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    checked_files = {}
    checked(SOURCE, SOURCE_HASH, checked_files)
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_purity')
    namespace = {'np': np, 'metrics': metrics}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    original = namespace['calculate_purity']

    def score(true, pred):
        with np.errstate(invalid='ignore', divide='ignore'):
            result = original('gold', 'pred', pd.DataFrame({'gold': true, 'pred': pred}))[2]
        if not np.isclose(result, independent_score(true, pred), atol=1e-12, rtol=0):
            raise ValueError('Author function disagrees with independent formula')
        return float(result)

    # Exact limiting cases, including rejection as an ordinary shared partition.
    for true, pred, expected in [([0, 0, 1, 1], [7, 7, 9, 9], 1.0),
                                 ([0, 0, 1, 1], [-1, -1, -1, -1], 2/3),
                                 ([0]*5+[1]*5, [0]*4+[1, 1]+[2]*4, 8/9)]:
        if not np.isclose(score(np.array(true), np.array(pred)), expected):
            raise ValueError('Formula fixture failed')

    primary = json.loads(Path('paper/results/manifest.json').read_text())
    for name, expected in primary['outputs'].items():
        checked(Path('paper/results')/name, expected, checked_files)
    data = Path('data/benchmark_release_20260906')
    input_manifest = json.loads((data/'inputs/manifest.json').read_text())
    checked(data/'data_manifest.json', input_manifest['sealed_data_manifest_sha256'], checked_files)
    manifest = json.loads((data/'data_manifest.json').read_text())
    rows = []
    for generator, directory in [('GPT-4.1 mini', 'release_cloud_20260906'), ('Luna', 'release_luna_20260906')]:
        run = Path('runs')/directory
        for path, key in [(run/'predictions_frozen.json', 'prediction_freeze_sha256'),
                          (run/'evaluation/all_metrics.csv', 'metrics_sha256'),
                          (run/'evaluation/scoring_manifest.json', 'scoring_manifest_sha256')]:
            checked(path, primary['inputs'][generator][key], checked_files)
        frozen = json.loads((run/'predictions_frozen.json').read_text())
        if len(frozen['runs']) != 20 or frozen.get('failures'):
            raise ValueError('Expected twenty completed cloud fits')
        checked(run/'preflight.json', frozen['preflight_sha256'], checked_files)
        preflight = json.loads((run/'preflight.json').read_text())
        checked(data/'inputs/manifest.json', preflight['input_manifest_sha256'], checked_files)
        old = pd.read_csv(run/'evaluation/all_metrics.csv')
        old = old[(old['split'] == 'test') & (old['sample_scope'] == 'eligible')]
        for key, expected in frozen['runs'].items():
            completion = run/key/'complete.json'
            checked(completion, expected, checked_files)
            for name, expected_hash in json.loads(completion.read_text())['files'].items():
                checked(completion.parent/name, expected_hash, checked_files)
            dataset, seed = key.split('/', 1)[1].rsplit('_seed', 1)
            info = manifest['datasets'][dataset]['splits']['test']
            gold_path = data/info['gold_file']
            checked(gold_path, info['gold_sha256'], checked_files)
            with gold_path.open() as handle:
                gold = list(csv.DictReader(handle))
            labels = {label: i for i, label in enumerate(sorted({r['label'] for r in gold}))}
            eligible = np.array([r['eligible'] == '1' for r in gold])
            true = np.array([labels[r['label']] for r in gold])[eligible]
            for method in METHODS:
                with (run/key/f'{method}_test.csv').open() as handle:
                    pred_rows = list(csv.DictReader(handle))
                if [r['doc_id'] for r in pred_rows] != [r['doc_id'] for r in gold]:
                    raise ValueError('Prediction/gold ID order mismatch')
                pred = np.array([int(r['topic_id']) for r in pred_rows])[eligible]
                hmp = harmonic_purity(true, pred)[2]
                coverage = float(np.mean(pred >= 0))
                previous = old[(old['dataset'] == dataset) & (old['seed'] == int(seed)) & (old['method'] == method)]
                if len(previous) != 1:
                    raise ValueError('Missing or duplicate original score')
                for metric, value in [('harmonic_purity', hmp), ('coverage', coverage), ('n_documents', len(true))]:
                    if not np.isclose(float(previous.iloc[0][metric]), value, atol=1e-12, rtol=0):
                        raise ValueError(f'Original score replay differs: {metric}')
                rows.append(dict(generator=generator, dataset=dataset, seed=int(seed), method=method,
                                 n_documents=len(true), coverage=coverage, frozen_hmp=hmp,
                                 topicgpt_weighted_best_f1=score(true, pred)))
    frame = pd.DataFrame(rows)
    if len(frame) != 240 or frame.duplicated(['generator','dataset','seed','method']).any():
        raise ValueError('Expected 240 unique eligible-test configurations')
    summary = frame.groupby(['generator','dataset','method'], sort=False).agg(
        runs=('seed','count'), frozen_hmp_mean=('frozen_hmp','mean'), frozen_hmp_sd=('frozen_hmp','std'),
        topicgpt_mean=('topicgpt_weighted_best_f1','mean'), topicgpt_sd=('topicgpt_weighted_best_f1','std'),
        coverage_mean=('coverage','mean')).reset_index()
    if len(summary) != 48 or not (summary['runs'] == 5).all():
        raise ValueError('Expected 48 complete five-run summaries')
    # The shared BERTopic series must remain exactly identical across generators.
    a = frame[(frame.generator == 'GPT-4.1 mini') & (frame.method == 'bertopic')].sort_values(['dataset','seed'])
    b = frame[(frame.generator == 'Luna') & (frame.method == 'bertopic')].sort_values(['dataset','seed'])
    if not np.array_equal(a.topicgpt_weighted_best_f1.to_numpy(), b.topicgpt_weighted_best_f1.to_numpy()):
        raise ValueError('Shared baseline differs')
    comparisons = []
    index = summary.set_index(['generator','dataset','method'])
    for gen in ['GPT-4.1 mini','Luna']:
        for dataset in DATASETS:
            h = index.loc[gen,dataset,'hybrid']
            for method in ['bertopic','kmeans','direct_hybrid','initial']:
                baseline = index.loc[gen,dataset,method]
                old_delta = h.frozen_hmp_mean - baseline.frozen_hmp_mean
                new_delta = h.topicgpt_mean - baseline.topicgpt_mean
                sign = lambda v: 0 if abs(v) < 1e-12 else int(np.sign(v))
                comparisons.append(dict(generator=gen,dataset=dataset,baseline=method,
                                        frozen_delta=old_delta,topicgpt_delta=new_delta,
                                        ordering_changed=sign(old_delta)!=sign(new_delta)))
    for path, expected in checked_files.items():
        if digest(path) != expected:
            raise ValueError(f'Evidence changed during scoring: {path}')
    OUT.mkdir(parents=True)
    frame.to_csv(OUT/'per_run.csv', index=False)
    summary.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(
        analysis_role='post_hoc_supplementary_metric_sensitivity_not_primary_replacement',
        formula='sum_c n_c/N * max_k 2*n_ck/(n_c+n_k)',
        scope='eligible test only; -1 retained as one predicted partition; no assigned-only substitution',
        source_url='https://github.com/chtmp223/topicGPT/blob/450564466abe72091797c728a90cfeec8ba3d651/topicgpt_python/utils.py',
        source_function='calculate_purity; extracted unchanged via AST',
        source_function_sha256=hashlib.sha256(ast.get_source_segment(SOURCE.read_text(),fn).encode()).hexdigest(),
        script_sha256=digest(__file__), input_hashes=checked_files,
        validation='3 exact fixtures; independent scalar agreement for all 240 rows; primary metric and coverage replay; shared baseline identity',
        rows=240, summary_rows=48,
        outputs={n:digest(OUT/n) for n in ['per_run.csv','summary.csv','comparisons.csv']}
    ),indent=2))
    print(summary.to_string(index=False))
    print('Ordering changes:')
    print(pd.DataFrame(comparisons).query('ordering_changed').to_string(index=False))


if __name__ == '__main__':
    main()
