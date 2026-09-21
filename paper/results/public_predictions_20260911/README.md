# Compact prediction evidence

This prepared release artifact contains 16,760 rows: two generator batches,
four fixed test collections and five runs. Six method predictions appear as
separate columns. Original eligibility flags are retained, including the excluded
AG News test row. The bundle is distributed with this repository.

No source text, label names, original document IDs, embeddings, model responses
or credentials are included. `doc_index` is the row position in each preserved
test input. `gold` numerically encodes the original label, consistently across
all runs of a dataset. `-1` remains a shared predicted partition for scoring.

After installing the base package from the checkout, run from its root:

```bash
python -m experiments.score_public_predictions \
  --bundle paper/results/public_predictions_20260911 --out tmp/rescored_public
python -m scripts.build_aligned_paper_results \
  --per-run tmp/rescored_public/per_run_metrics.csv --out tmp/rescored_public_tables
```

Use new output directories. Scoring needs only this CSV, its manifest and the
base package; it does not access `runs/`, `external/`, source texts, models or APIs.
It checks hashes, forty planned run positions, document indices, gold/eligibility
consistency and sample counts. The output includes the legacy global HMP, coverage, ARI, NMI and TopicGPT
weighted best-pair F1. The current aligned table builder uses the TopicGPT score
as main HMP, preserving the legacy score separately. See the
[11 September amendment](../../../docs/EVALUATION_PROTOCOL_20260911.md).

[The manifest](manifest.json) records source-file hashes and exporter identity.
Hashes support provenance and corruption checks, not independent verification
of the source annotations. Collection sources, attribution and historical
revision limitations are in [the reproduction guide](../../../docs/REPRODUCIBILITY.md).
This numeric export is distinct from the original text corpora; source terms and attribution are described in the reproduction guide and NOTICE.md.

On 11 September 2026, validation reproduced all 960 original metric values
exactly and 240 TopicGPT-formula scores within 2.23e-16. The installed wheel also
recomputed scores in an isolated working directory containing copies of only
the CSV and manifest. Full generation and saved-model reproduction materials
remain separate from this scoring bundle.
