# Reproducing the release evidence

The two result batches are `release_cloud_20260906` (GPT-4.1 mini) and
`release_luna_20260906` (Luna), with twenty completed fits each. Their predictions
and original source snapshots remain frozen. Later provider/configuration fixes
were checked against all 99 recorded OpenAI requests without changing their
payloads or input counts. Older Hybrid experiments are not combined with these.

The active reporting version is `topicgpt-weighted-best-f1-v1`. The original
`experiments.paper_tables` remains available for historical global-HMP tables.
Use the explicit alignment builder below for current manuscript results. In raw
scorer output, `harmonic_purity` retains its historical meaning; in the versioned
aligned report it aliases `topicgpt_weighted_best_f1`, with the old value retained
as `legacy_global_hmp`. Original files under `paper/results/` and the September 10
formula report are historical evidence, superseded as the main report by the
September 11 aligned directory.

## 1. Rebuild tables from the distributed numeric results

From the checkout after installing the base package:

```bash
python -m scripts.build_aligned_paper_results \
  --per-run paper/results/topicgpt_aligned_20260911/per_run_metrics.csv --out tmp/rebuilt_paper_tables
python -m pip install matplotlib
python scripts/plot_paper_results.py \
  --data tmp/rebuilt_paper_tables/all_methods.csv --out tmp/rebuilt_paper_figures
```

Choose a new table output directory. The 240 numeric rows represent two generators,
four collections, five runs and six methods. The builder rejects missing or duplicate
planned positions and reports every method. Outputs contain 48 collection/method
summary rows, means and sample SD, plus a manifest with input and builder hashes.
This reproduces aggregation and plots; it does not independently recompute scores
from document predictions. No API key, original text or model download is needed.

`hybrid` is the final model; `initial` retains the initial codebook and assignment;
`direct_hybrid` and `direct_initial` assign by nearest-topic cosine without
rejection. BERTopic is the same cached baseline in both batches. Each KMeans run
uses the final topic count of its own generator run, so the two KMeans series differ.

## 2. Recompute scores from the compact prediction bundle

The prepared [document-level bundle](../paper/results/public_predictions_20260911/README.md)
contains coded gold labels, eligibility flags and all six methods' predictions.
It omits source text, original document IDs, model weights and credentials.

```bash
python -m experiments.score_public_predictions \
  --bundle paper/results/public_predictions_20260911 --out tmp/rescored_public
python -m scripts.build_aligned_paper_results \
  --per-run tmp/rescored_public/per_run_metrics.csv --out tmp/rescored_public_tables
```

This path recomputes the current TopicGPT-aligned HMP and the historical global
HMP directly from document predictions, alongside coverage, ARI and NMI. It requires only the installed base
package and the CSV/manifest, not `runs/`, original texts, embeddings or APIs.
The same 799-document eligible AG News subset is retained. Validation matched
960 original metric values exactly and 240 TopicGPT-formula values to floating-point
precision, including an installed-wheel check in an isolated working directory.
The CSV and manifest are distributed in this repository.

### Full local audit path

The original research scorer remains available for the complete local artifacts:

```bash
python -m experiments.score_release \
  --data data/benchmark_release_20260906 \
  --run runs/release_cloud_20260906 --out tmp/rescored_cloud
python -m experiments.score_release \
  --data data/benchmark_release_20260906 \
  --run runs/release_luna_20260906 --out tmp/rescored_luna
```

This scorer additionally checks the full original data, model and completion
artifacts. The compact bundle is a separate export for score reproduction and
does not satisfy these full-bundle paths. Exact original source-record recovery
and generation/model-state packaging remain separate limitations.

## 3. Sources and frozen subset identities

| Collection | Source used by the preserved preparation code | Frozen induction/test |
|---|---|---:|
| Bills | [TopicGPT repository](https://github.com/chtmp223/topicGPT), Bills JSONL preparation | 272 / 376 |
| SIB-200 | [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200), `zho_Hans` and `eng_Latn` | 210 / 140 |
| MASSIVE scenarios | `SetFit/amazon_massive_scenario_zh-CN` and `SetFit/amazon_massive_scenario_en-US` | 540 / 360 |
| AG News | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), revision `eb185aade064a813bc0b7f42de02595523103ca4` | 1200 / 800, 799 eligible |

The SIB-200/MASSIVE historical preparation pooled the two languages before
label-balanced sampling: thirty train and twenty test records per category,
seed eleven. It did not enforce equal counts per language. Bills retained labels
with at least fifteen records and used the preserved per-label split rule.
AG News uses the deduplication and hashed selection in `experiments.data_io`, with
the overlap eligibility mask documented in the frozen protocol.

Current upstream downloads may differ from the saved subset. A source link and
sample size do not establish exact reproduction: use the source-record identity,
original revisions where recorded, and the frozen input hashes. Some earlier
upstream revisions were not pinned; no new revision should be invented for them.
Data acquisition and redistribution must retain the relevant source terms and
attribution. The software's eventual license does not replace dataset terms.

## 4. Rerunning generation

The current `experiments.luna_benchmark` command accepts `--data`, `--cache`,
`--out` and `--api-key-file`. It requires the verified document-vector and BERTopic
cache with its preflight, embedding index, and baseline files. Each new execution
freezes the selected model-catalog JSON. Use a new output directory and record a
new batch identity. New API responses are not expected to match historical
responses exactly. Do not overwrite old predictions or silently replace poorer runs.

The former mixed release runner can enter its stopped local phase and is not the
recommended cloud reproduction command. Its original artifacts remain preserved.
Use the public cloud example for new personal datasets; it is independent of the
research cache and requires no benchmark labels:

```bash
python -m examples.cloud_workflow --csv comments.csv --text-column text \
  --output my_cloud_run --allow-download
```

The reference parameters, limits and evidence boundaries are stated in the
[README](../README.md), [frozen protocol](EXPERIMENT_PROTOCOL_RELEASE.md), and
[current manuscript](../paper/manuscript.md). All four evaluation collections
had prior development exposure. The [11 September metric amendment](EVALUATION_PROTOCOL_20260911.md)
adopts TopicGPT’s weighted best-pair F1 as main HMP after both definitions were
inspected; coverage is reported alongside it. No composite or best-generator selection is used.

## 5. Prompt sensitivity

The complete compact/neutral comparison is in [the prompt sensitivity directory](../paper/results/prompt_sensitivity_20260920/README.md): 80 final Hybrid runs and 480 configurations across all methods. Summary CSVs include means and sample standard deviations. The neutral comparison changes the initial prompt sentence group and repeats generation. Raw generation requests, model files and original corpora remain in the research archive; the table builder for this comparison needs those saved models to recount topics. The public numerical results support reaggregation without generating new topics.
