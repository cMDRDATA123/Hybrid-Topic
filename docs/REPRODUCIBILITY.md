# Running experiments and reading the paper

The [English manuscript](../paper/hybrid_topic_en.pdf), [Chinese manuscript](../paper/hybrid_topic_zh.pdf), and [supplementary material](../paper/supplementary_material.md) provide the published repository's research reading materials.

## Run the software

Follow the [README](../README.md) for installation, offline examples, and cloud generation. The cloud example requires your own credentials and model configuration:

```bash
python -m examples.cloud_workflow --output my_cloud_run --allow-download
```

## Research data and experiment scripts

The experiment and reporting scripts remain available as tools for preparing data, running models, and scoring saved predictions. Use your own input and output paths when running them. The original experiment archives, per-run exports, and paper source workspace are retained separately by the author. Recreating the reported runs requires their corresponding data, configurations, models, and saved outputs.

The paper reports two generators, five runs for each of four collection subsets, and the compact/neutral prompt comparison. The evaluation collections had prior development exposure. HMP uses the category-size-weighted best-match F measure, alongside coverage, ARI, and NMI. The supplementary material explains the metric history and comparison conditions.

## Dataset sources

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
