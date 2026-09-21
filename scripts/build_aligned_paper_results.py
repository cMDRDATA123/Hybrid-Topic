"""Build current TopicGPT-aligned tables from scores containing both definitions.

The historical scorer's harmonic_purity column keeps its original meaning.
This explicitly versioned report maps the verified TopicGPT column to main HMP
and preserves the former score as legacy_global_hmp. No model is rerun.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paper_tables import digest, summarize, write_tables

METRIC_VERSION = 'topicgpt-weighted-best-f1-v1'


def build_aligned(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    frame = pd.read_csv(source)
    if 'topicgpt_weighted_best_f1' not in frame:
        raise ValueError('verified TopicGPT score column is required; legacy-only input rejected')
    if not set(['harmonic_purity', 'coverage', 'ari', 'nmi']).issubset(frame.columns):
        raise ValueError('missing score columns')
    metrics = frame[['topicgpt_weighted_best_f1', 'harmonic_purity', 'coverage', 'ari', 'nmi']]
    if not np.isfinite(metrics.to_numpy()).all():
        raise ValueError('finite scores required')
    if not frame.topicgpt_weighted_best_f1.between(0, 1).all():
        raise ValueError('TopicGPT scores must be in [0, 1]')
    if set(frame.generator) != {'GPT-4.1 mini', 'Luna'}:
        raise ValueError('both complete generator batches required')
    if 'legacy_global_hmp' not in frame:
        frame['legacy_global_hmp'] = frame.harmonic_purity
    frame['harmonic_purity'] = frame.topicgpt_weighted_best_f1
    summaries, runs = [], []
    for label in ['GPT-4.1 mini', 'Luna']:
        rows = frame[frame.generator == label].copy()
        summary = summarize(rows)
        summary.insert(0, 'generator', label)
        summaries.append(summary)
        runs.append(rows)
    manifest = {
        'metric_version': METRIC_VERSION,
        'primary_metric': 'gold-class-size-weighted best class-cluster F1 (TopicGPT implementation)',
        'formula': 'sum_c (n_c/N) * max_k (2*n_ck/(n_c+n_k))',
        'columns': {'harmonic_purity': 'current main HMP, equal to topicgpt_weighted_best_f1',
                    'legacy_global_hmp': 'previous global purity/inverse-purity harmonic mean'},
        'decision': '2026-09-11 author-directed alignment to prior research after both definitions were inspected',
        'numeric_input_sha256': digest(source),
        'alignment_builder_sha256': digest(__file__),
        'reference_revision': '450564466abe72091797c728a90cfeec8ba3d651',
        'scope': 'all 240 eligible test configurations; unchanged predictions, -1 partition, coverage, ARI and NMI',
    }
    out.mkdir(parents=True)
    result = write_tables(summaries, runs, manifest, out)
    table = out/'tables.md'
    table.write_text(table.read_text().replace('# Frozen release comparisons',
        '# Current comparisons: TopicGPT-aligned HMP\n\n'
        'Metric version: `topicgpt-weighted-best-f1-v1`. Main HMP is the '
        'gold-class-size-weighted best pair F1. The legacy global HMP is retained '
        'in the per-run file. This reporting revision reuses all frozen predictions.'))
    saved = json.loads((out/'manifest.json').read_text())
    saved['outputs']['tables.md'] = digest(table)
    (out/'manifest.json').write_text(json.dumps(saved, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build_aligned(args.per_run, args.out)
    print(f'Built {len(result)} summary rows using {METRIC_VERSION}.')
