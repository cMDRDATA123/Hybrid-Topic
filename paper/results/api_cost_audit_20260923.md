# API generation cost audit — 23 September 2026

The main paper reports a **no-cache-pricing scenario**: every recorded input token is charged at the ordinary uncached input rate; every recorded output token is charged at the ordinary output rate. This is a common basis for comparing generators. It is an estimate, not the account invoice.

| Model | Fits | Requests | Input tokens | Output tokens | Standard input/output price per 1M | No-cache-pricing batch total | Per fit |
|---|---:|---:|---:|---:|---|---:|---:|
| GPT-4.1 mini | 20 | 55 | 782,176 | 19,476 | $0.40 / $1.60 | $0.344032 | $0.0172016 |
| GPT-5.6 Luna | 20 | 44 | 786,151 | 37,296 | $0.20 / $1.20 | $0.2019854 | $0.01009927 |

Calculations: `(782176 × 0.40 + 19476 × 1.60) / 1,000,000 = 0.344032`; `(786151 × 0.20 + 37296 × 1.20) / 1,000,000 = 0.2019854`. Each batch total is divided by twenty for the mean per fit. All 99 request records have usage. Every request has fewer than 272,000 input tokens, so Luna's long-context multiplier is inapplicable.

The usage records also include cache accounting. Mini has 573,440 cached-input tokens and no cache-write token field. Luna has 576,392 cached-input tokens and 209,627 cache-write tokens; these plus 132 other input tokens account for its 786,151 input tokens. The main table deliberately treats all of them as ordinary input tokens. If the recorded Luna cache-write tokens are retained, but every cache read is counterfactually charged at the ordinary input rate, its no-cache-hit estimate becomes `$0.2019854 + 209627 × ($0.20 × 0.25) / 1,000,000 = $0.21246675` or `$0.0106233375` per fit. A true alternative run with no cache hits could write a different number of tokens, so this latter estimate is conditional on the recorded write count.

For reference, applying the *recorded* cache categories and the published rates gives $0.1720 for mini and $0.10871619 for Luna. These are rate-card estimates and have not been reconciled to the account invoice. The paper's main table uses neither of these discounted totals. Local encoding, graph computation, data preparation and other development requests are outside the reported generation cost.

Official rate sources: [GPT-4.1 mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini); [GPT-5.6 Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna). The Luna page states that cache writes cost 1.25 times the uncached input rate, and inputs above 272K carry a larger request-wide rate. No recorded request reaches that threshold. [OpenAI's July 2026 price update](https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/) confirms that the $0.20/$1.20 Luna rates preceded these September experiments.

Research evidence: `runs/release_cloud_20260906/usage_summary.json`, `runs/release_luna_20260906/usage_summary.json`, and their request-level usage records. The public repository distributes these aggregate calculations and the paper; raw requests remain in the local research archive.
