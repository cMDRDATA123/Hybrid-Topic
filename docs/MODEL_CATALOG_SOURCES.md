# Official model catalog and G-08 provenance

This project does not treat a hand-written list of model names as permanent
truth. The snapshot in
[`MODEL_CATALOG_SNAPSHOT_20260906.json`](MODEL_CATALOG_SNAPSHOT_20260906.json)
records the September 6 inventory review. The audit found incomplete capability
coverage; it is not a complete or permanently current list of supported models.
The runtime JSON now separates field provenance, lifecycle and account visibility.

| Provider | Official inventory | Runtime list | Capability/lifecycle source | Current adapter status |
|---|---|---|---|---|
| OpenAI | [All models](https://developers.openai.com/api/docs/models/all) | `/v1/models` | Individual model pages under `developers.openai.com/api/docs/models/` | OpenAI and Luna capability entries verified; remaining entries require individual capability verification |
| Anthropic | [Model deprecations and status](https://platform.claude.com/docs/en/about-claude/model-deprecations) | `/v1/models` | Individual Claude model pages and lifecycle table | Claude adapter implemented; capability entries remain explicit until model-page limits are recorded |
| xAI | [Models](https://docs.x.ai/developers/models) | `/v1/language-models` | Individual Grok model pages | OpenAI-compatible adapter implemented; account-scoped inventory must be refreshed with credentials |
| DeepSeek | [List models](https://api-docs.deepseek.com/api/list-models/) | `/models` | [Models and pricing](https://api-docs.deepseek.com/quick_start/pricing) | OpenAI-compatible adapter implemented; V4 limits recorded; legacy aliases retained only for diagnostics |

The provider pages show why the catalog needs lifecycle and provenance fields.
OpenAI's official page combines current, deprecated, multimodal and embedding
families; Anthropic publishes active, deprecated and retired states; xAI exposes
account-scoped language-model listings; and DeepSeek separates its model list
from model limits and legacy-alias notices.

G-08 therefore has three safe behaviors:

1. Use a verified catalog entry when its context, output, tokenizer and reasoning
   fields are known.
2. Refresh model IDs from the provider's official list endpoint when credentials
   are available, while preserving the returned snapshot and date.
3. Refuse to infer missing limits from a model name. The user must provide an
   explicit value before the model can participate in whole-document packing.

The catalog is an input to capability resolution, not a claim that every listed
model supports structured topic JSON or is available on every account. Those
properties remain adapter- and provider-specific and must be tested separately.

## Runtime configuration

The checked-in runtime file is
[`../hybrid_topic/model_catalog.json`](../hybrid_topic/model_catalog.json). It is
package data, so an installed copy can resolve the same verified entries as a
checkout. A generator can use a refreshed file explicitly with
`catalog_path="path/to/model_catalog.json"`, or all generators can use the
`HYBRID_TOPIC_MODEL_CATALOG` environment variable. The resolver reads the JSON
when a generator is constructed and records its byte hash, selected entry,
path/date and explicit-versus-catalog field sources in generation metadata.

Updating the file is therefore a data/configuration update, not a code edit. A
refresh must preserve the official source URL and date, keep capability fields
empty until the provider documents them, and never infer a tokenizer or limit
from a model-name suffix. New runs copy the selected catalog to `source_snapshot/model_catalog.json` before
fitting. The preflight records its source and hash; every fit uses the frozen
copy and verifies the hash. External configuration edits cannot change that
batch. September 6 completed runs predate JSON externalization and preserve the
original capability code in their Python snapshots.

With the corresponding provider key configured, the inventory portion can be
refreshed without changing capability fields:

```bash
python scripts/refresh_model_catalog.py --provider openai --dry-run
python scripts/refresh_model_catalog.py --provider openai
```

Use `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`, or
`--api-key` for the other supported providers. The command records which IDs
were visible to that account; it does not mark an unverified model safe for
automatic context budgeting. Review the official capability page and then add
the capability fields to the configuration as a separate, dated change.


## September 7 repair contract

The runtime inventory currently contains 47 IDs. Ten OpenAI entries include
context, output ceiling and a tokenizer. Other entries are partial or unverified;
no account inventory was refreshed during repair. Existing capability values
retain their original recording date and are labelled inherited unless this
repair explicitly rechecked the field. A model-list refresh changes only
account visibility and `last_inventory_refresh`, not `snapshot_date` or a
capability's verification date. New IDs have unknown modality and unverified
capabilities until reviewed; they are not automatically text-generation defaults.

Claude effort values were rechecked against the [official effort compatibility
table](https://platform.claude.com/docs/en/build-with-claude/effort).
The adapter uses [Messages output configuration](https://platform.claude.com/docs/en/api/messages/create)
and [model-aware input counting](https://platform.claude.com/docs/en/api/messages/count_tokens).
DeepSeek request mapping and effort values follow the [official Chat Completions
reference](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/).
Legacy DeepSeek aliases are retained as `legacy_unverified`, without the previous
conflicting `active` claim. This is not a claim that legacy aliases remain usable.

For compatible endpoints, pass `input_token_counter(request_kwargs) -> int` and
`token_count_source="provider/model/tokenizer revision and framing policy"`.
The callback receives model, messages, format, output cap and any extra body.
It must account for the deployed model's tokenizer and template, plus schema
and transport framing. A provider counting endpoint or a pinned matching local
tokenizer can implement it. `tiktoken` selected by name is not automatically a
valid counter for xAI, DeepSeek, Ollama or vLLM. The counter's accuracy is the
integrator's responsibility; its source is retained with the generation audit.
Claude's counter is built in and does not need this callback.

The adapter enforces the budget against that count and records post-call usage.
The 75% ceiling is a count-based limit, not a universal promise that an arbitrary
custom estimate equals a provider's internal tokenizer. Prompt text, schema,
packing seed and assignment/metric definitions used by the existing OpenAI
experiments are unchanged by these repairs.
