# Prompt sensitivity comparison, 20 September 2026

Compact remains the main experiment and release default. Neutral is a complete sensitivity comparison: four collections, two generators, five seeds each. Final topic counts include any residual additions and are counted from saved model codebooks.

- `per_run.csv`: 80 final Hybrid runs, topic counts and metrics.
- `summary.csv`: all 16 condition/generator/collection groups; means and sample SD (ddof=1).
- `all_methods_per_run.csv`: 480 configurations, all six methods under both conditions.
- `all_methods_summary.csv`: 96 groups with means and sample SD.

Compact scores come from `paper/results/topicgpt_aligned_20260911/per_run_metrics.csv`; neutral scores come from `runs/neutral_prompt_20260919/evaluation/per_run_metrics.csv`. Both use topicgpt-weighted-best-f1-v1. The baseline rows are reused under the shared protocol; repeated rows across conditions are not new baseline fits. Compact model codebooks are in release_cloud_20260906 and release_luna_20260906; neutral codebooks and the frozen source are in neutral_prompt_20260919. The comparison replaces the whole topic-discovery sentence group and repeats stochastic generation; differences include wording and generation variation. The propagation ablation and historical cost table refer to compact.

For neutral generation reproduction, use the frozen neutral source snapshot and the protocol in docs/EXPERIMENT_PROTOCOL_NEUTRAL_20260919.md. The active neutral runner now checks prompt identity before any paid work.

Rebuild these tables with `python -m experiments.prompt_sensitivity_tables` in the project research environment, with the original saved runs available. This reads existing outputs and makes no generation requests.
