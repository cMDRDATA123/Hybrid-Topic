"""One-shot paired cloud experiment; separate generation and scoring processes."""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.data_io import sha256, write_json
from experiments.release_benchmark import (
    DATASETS, SEEDS, EMBEDDING, CLOUD_MODEL, OpenAITopicGenerator,
    SentenceTransformerEmbedder, prepare_reuse, validate_cached_embeddings,
    run_one, atomic_json,
)


def execute(root, out):
    from hybrid_topic.backends import _initial_prompt
    if "Let the content of the documents guide the number and specificity" not in _initial_prompt([]):
        raise RuntimeError(
            "Neutral experiment requires its frozen source snapshot; the active release uses compact. "
            "Run from runs/neutral_prompt_20260919/source_snapshot with its PYTHONPATH."
        )
    start = time.time()
    def status(state, **fields):
        atomic_json(out / 'queue_status.json', dict(state=state, started_at=start,
                    updated_at=time.time(), elapsed_seconds=time.time()-start, planned=40, **fields))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'preflight.json').exists():
            raise RuntimeError('Batch already started; automatic paid rerun disabled')
        try:
            status('preparing')
            data_root = root / 'data/benchmark_release_20260906'
            cache = root / 'archive/release_consolidation_20260906/workspace/runs/v2_fixed_20260906'
            data, reuse = prepare_reuse(data_root, cache, out)
            configs = [('cloud', CLOUD_MODEL, 1047576, {}),
                       ('luna', 'gpt-5.6-luna', 1050000, {'reasoning_effort': 'low'})]
            write_json(out / 'preflight.json', dict(protocol='neutral-prompt-20260919-v1',
                prompt_version='content-guided-number-specificity-v1', models=configs,
                seeds=SEEDS, datasets=DATASETS, planned_fits=40,
                input_manifest_sha256=sha256(data_root / 'inputs/manifest.json'),
                baseline_reuse=reuse, source_snapshot=str(out/'source_snapshot'),
                max_format_retries=1, max_output_tokens=16384, input_context_ratio=0.75,
                no_fee_ceiling_user_authorized=True))
            embedding = SentenceTransformerEmbedder(EMBEDDING, device='mps', local_files_only=True)
            vectors = {name: validate_cached_embeddings(cache/'embeddings', name, data[name]) for name in DATASETS}
            key = (root / 'api.txt').read_text().strip()
            for kind, model, context, extra in configs:
                for name in DATASETS:
                    for seed in SEEDS:
                        completed = len(list(out.glob('*/*/complete.json')))
                        failed = len(list(out.glob('*/*/failure.json')))
                        status('running', current=f'{kind}/{name}_seed{seed}', completed=completed, failed=failed)
                        backend = OpenAITopicGenerator(model=model, api_key=key, context_window=context,
                            catalog_path=out/'source_snapshot/hybrid_topic/model_catalog.json',
                            sampling_seed=seed, tokenizer_encoding='o200k_base',
                            max_output_tokens=16384, input_context_ratio=0.75,
                            safety_margin_tokens=1024, max_format_retries=1, **extra)
                        backend.client = backend.client.with_options(timeout=300, max_retries=0)
                        tr, te = vectors[name]
                        run_one(out/kind/f'{name}_seed{seed}', kind, name, seed,
                                data[name], tr, te, backend, embedding, cache)
            completed = {str(p.parent.relative_to(out)): sha256(p) for p in out.glob('*/*/complete.json')}
            failures = {str(p.parent.relative_to(out)): sha256(p) for p in out.glob('*/*/failure.json')}
            if len(completed)+len(failures) != 40:
                raise RuntimeError('Run accounting mismatch')
            write_json(out/'predictions_frozen.json', dict(preflight_sha256=sha256(out/'preflight.json'),
                       runs=completed, failures=failures, planned=40))
            usage = {}
            for kind, model, _, _ in configs:
                reqs = [json.loads(p.read_text()) for p in out.glob(f'{kind}/*/requests/request_*.json')]
                usage[kind] = dict(model=model, requests=len(reqs),
                    input_tokens=sum((r.get('usage') or {}).get('prompt_tokens',0) for r in reqs),
                    output_tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in reqs),
                    missing_usage=sum(not r.get('usage') for r in reqs))
            write_json(out/'usage_summary.json', usage)
            status('scoring', completed=len(completed), failed=len(failures))
            subprocess.run([sys.executable, '-m', 'experiments.score_release', '--data',str(data_root),
                '--run',str(out),'--out',str(out/'evaluation_legacy')], check=True, cwd=out/'source_snapshot')
            subprocess.run([sys.executable, '-m', 'experiments.neutral_prompt_benchmark',
                '--root',str(root),'--out',str(out),'--score'], check=True, cwd=out/'source_snapshot')
            status('completed_with_failures' if failures else 'completed', completed=len(completed), failed=len(failures))
        except Exception as exc:
            status('failed', error_type=type(exc).__name__)
            raise


def score(root, out):
    import numpy as np
    import pandas as pd
    from scripts.score_topicgpt_formula import independent_score
    manifest = json.loads((root/'data/benchmark_release_20260906/data_manifest.json').read_text())
    frozen = json.loads((out/'predictions_frozen.json').read_text())
    frame = pd.read_csv(out/'evaluation_legacy/all_metrics.csv')
    frame = frame[(frame.split=='test') & (frame.sample_scope=='eligible')].copy()
    values = {}
    for key in frozen['runs']:
        kind, run = key.split('/')
        name, seed = run.rsplit('_seed',1)
        gold_file = manifest['datasets'][name]['splits']['test']['gold_file']
        with (root/'data/benchmark_release_20260906'/gold_file).open() as f:
            gold = list(csv.DictReader(f))
        eligible = np.array([r['eligible']=='1' for r in gold])
        true = np.array([r['label'] for r in gold])[eligible]
        for method in ['hybrid','initial','direct_hybrid','direct_initial','bertopic','kmeans']:
            pred = pd.read_csv(out/key/f'{method}_test.csv')
            if pred.doc_id.astype(str).tolist() != [r['doc_id'] for r in gold]:
                raise ValueError('Prediction/gold IDs differ')
            values[(kind,name,int(seed),method)] = independent_score(true,pred.topic_id.to_numpy()[eligible])
    frame['legacy_global_hmp'] = frame.harmonic_purity
    frame['harmonic_purity'] = [values[(r.backend,r.dataset,r.seed,r.method)] for r in frame.itertuples()]
    dest=out/'evaluation'; dest.mkdir(exist_ok=False)
    frame.to_csv(dest/'per_run_metrics.csv',index=False)
    summary=frame.groupby(['backend','dataset','method']).agg(runs=('seed','count'),
        hmp_mean=('harmonic_purity','mean'),hmp_std=('harmonic_purity','std'),
        coverage_mean=('coverage','mean'),coverage_std=('coverage','std'),
        ari_mean=('ari','mean'),nmi_mean=('nmi','mean')).reset_index()
    summary['planned_runs']=5
    summary['missing_or_failed_runs']=5-summary.runs
    summary.to_csv(dest/'summary.csv',index=False)
    write_json(dest/'manifest.json',dict(metric_version='topicgpt-weighted-best-f1-v1',
        scope='eligible test; -1 is a shared partition', rows=len(frame), expected_rows=240,
        completed=len(frozen['runs']), failed=len(frozen['failures'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--score',action='store_true')
    a=p.parse_args(); (score if a.score else execute)(a.root,a.out)
