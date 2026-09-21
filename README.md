# Hybrid Topic

Hybrid Topic is a Python/Jupyter topic modeling prototype. A configured language
model discovers named topic definitions; local embeddings and graph diffusion
assign documents. It accepts a supplied topic codebook or an optional analysis
direction, and supports result inspection, export, model saving, and new texts.

[English paper](paper/hybrid_topic_en.pdf) · [中文论文](paper/hybrid_topic_zh.pdf) · [Quickstart notebook](examples/quickstart.ipynb) · [Reproduce the results](docs/REPRODUCIBILITY.md)

Developed by **Jiashuo Ren, University of Stirling**. Code is available under the [MIT license](LICENSE); article and dataset terms are described in [NOTICE.md](NOTICE.md).

## Install

Python 3.10+ is declared; development validation uses Python 3.12 on macOS.
Install from a checkout:

```bash
git clone https://github.com/cMDRDATA123/Hybrid-Topic.git
cd Hybrid-Topic
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m examples.basic_workflow --output tmp/offline_example
```

The base install supports the offline example using a supplied codebook. For
real discovery, install the backends you plan to use:

```bash
python -m pip install -e '.[embeddings,openai]'
# For the offline tutorial:
python -m pip install -e '.[notebook]'
python -m jupyterlab examples/quickstart.ipynb
```

Installing an extra does not itself start generation. Loading uncached weights
may download them; cloud generation sends text to the API, including candidate
texts sent for counting where the adapter uses a remote count endpoint.
The [quickstart notebook](examples/quickstart.ipynb) executes offline; its optional
real-backend example is clearly marked.

## Run the cloud example

With `OPENAI_API_KEY` set in your environment and the embedding/OpenAI extras
installed, run twelve synthetic product comments end to end:

```bash
python -m examples.cloud_workflow --output my_cloud_run --allow-download
```

Omit `--allow-download` when the embedding model is already cached. The default
generator is `gpt-4.1-mini-2025-04-14`; use `--model` and `--context-window` for a
verified alternate deployment. The example discovers topics, exports tables,
saves and reloads the model, and checks individual versus batch query results.
Choose a new output directory for each run. It sends the selected comments to
OpenAI and incurs API usage; embeddings run on the configured local device.
Use `--csv comments.csv --text-column text` for your own collection, or
`--direction "Identify recurring product issues"` to supply optional guidance.

The public [reproduction guide](docs/REPRODUCIBILITY.md) distinguishes rebuilding
paper tables from numeric data, scoring frozen predictions, and rerunning models.
The [current result tables](paper/results/topicgpt_aligned_20260911/tables.md) include all methods and runs.

## Model workflow

```python
import os
from hybrid_topic import HybridTopic, Topic, read_documents
from hybrid_topic.backends import OpenAITopicGenerator, SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    "Qwen/Qwen3-Embedding-0.6B", device="cpu", local_files_only=False,
)
generator = OpenAITopicGenerator(
    model=os.environ["OPENAI_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
    # Optional: use a refreshed copy of hybrid_topic/model_catalog.json.
    # The default package catalog is used when this is omitted.
    # catalog_path="config/model_catalog.json",
    # Known OpenAI models resolve their verified context automatically.
    # Supply context_window for an unlisted deployment.
    # Optional; omit for general discovery:
    topic_instruction="Identify recurring product issues",
)
model = HybridTopic(generator=generator, embedding_model=embedder)
documents = read_documents("comments.csv", text_column="text")
topics, scores = model.fit_transform(documents)
model.get_topic_info()
model.get_document_info()
model.export("my_results")
model.save("my_model")

loaded = HybridTopic.load("my_model", embedding_model=embedder)
new_result = loaded.transform_result(["The battery drains quickly"])
new_result.get_document_info()
```

For supplied topics, use `model.fit(documents, codebook=[Topic(...)])`; this skips
all generation and keeps the codebook fixed. Topic fields are `topic_id`, `name`,
`definition`, and `core_keywords`. Model output IDs are zero-based positions;
`-1` means unassigned. Scores are uncalibrated assignment scores, not probabilities.

| Model setting | Default |
|---|---:|
| `p`, graph-similarity quantile | `0.95` |
| `seed_per_topic` | `5` |
| `diffusion_rate` | `0.70` |
| `rounds`, maximum diffusion iterations | `10` |
| `max_residual_rounds` | `2` |

These are fixed empirical starting values with explicit overrides. The model
does not select parameters against labels or run the retired Pareto grid.
Generator-backed fit tries up to two residual rounds, using only currently
unassigned texts and the full current codebook. It stops early if there are too
few residual texts or no valid new topics. Set `max_residual_rounds=0` or `1` to
reduce expansion. A failed fit preserves the previously fitted state.

New-document `transform` encodes each query independently and uses frozen training
references. It does not create new topics or construct a graph between query
texts. With an unchanged deterministic embedding backend, a text's assignment
is independent of the other texts submitted alongside it. Training diffusion and
new-query inference are different operations, so `transform(training_texts)`
need not match `fit_transform` labels.

The embedding adapter currently truncates at 512 tokens; the generator's context
budget is separate. Training uses a dense document graph, so large collections
need memory checks. Cross-platform, long-document and broad multilingual quality
claims require evidence beyond the current engineering checks.

## Generation context and retries

Known model IDs resolve their checked context, output limit, tokenizer and
reasoning options from the replaceable `hybrid_topic/model_catalog.json` file.
Set `catalog_path=...` on a generator or the `HYBRID_TOPIC_MODEL_CATALOG`
environment variable to use an updated copy. Unknown deployments, and models
listed in the official inventory whose limits have not yet been verified, must
provide their actual `context_window`. OpenAI also needs a verified
`tokenizer_encoding` when the catalog does not identify one. Claude uses its
model-aware `messages.count_tokens` endpoint. Other compatible endpoints require
an `input_token_counter` callable and a descriptive `token_count_source`; a
`tiktoken` encoding name alone cannot establish another model's input budget.

| Setting | Default | Meaning |
|---|---:|---|
| `context_window` | auto for known models | Actual configured context capacity in tokens |
| `input_context_ratio` | `0.75` | Complete input ceiling; configurable in `(0, 0.75]`, including `0.70` |
| `max_output_tokens` | `16384` | Output ceiling, capped with a warning at a known provider maximum |
| `safety_margin_tokens` | `1024` | Additional space reserved outside the input/output allocation |
| `max_format_retries` | `1` | Additional attempts for invalid JSON/topic fields; `0` disables repair |
| `sampling_seed` | `42` | Reproducible document selection when the input exceeds the budget |

The complete input budget is:

```text
min(floor(context_window * input_context_ratio),
    context_window - effective_max_output_tokens - safety_margin_tokens)
```

For a declared 1,000,000-token context, the default input ceiling is 750,000.
For a 32,768-token context with 16,384 output tokens and a 1,024-token safety
margin, it is 15,360. Output tokens are a maximum, not a required response length.
If the output reservation leaves no input space, configuration fails with an
explanation rather than silently lowering the requested output limit.

Packing counts instructions, the existing codebook, document IDs, text, and
reserved repair instructions. OpenAI uses `tiktoken` plus schema and estimated
message overhead; actual provider usage is recorded after the call. Claude counts
messages, system text, thinking settings and output schema through its official
counting endpoint. These additional counting requests also send candidate text
to Claude; they may include documents that are later excluded from generation.
Counting errors stop generation. Large sampled collections may require many
count requests. Compatible-endpoint counters receive complete SDK request kwargs,
including `extra_body`, and must count the deployed tokenizer, chat template and
schema/framing overhead. The library validates the returned count's type, but
cannot certify a user-supplied counter's accuracy.
All candidate documents are included unchanged when they fit. Otherwise a fixed
random order is used to fit whole documents into the remaining token budget.
This can favor shorter documents and is not label-stratified sampling. Empty
budgets and cases where no whole document fits produce an explicit error;
there is no implicit 500-character truncation. All input documents still take
part in embedding-based topic assignment even if generation sees a sample.

Format repair uses the same selected documents plus a fixed repair instruction.
Invalid JSON or invalid topic fields can trigger repair; provider refusal,
output truncation, and transport errors are surfaced without automatic retry.
SDK transport retries are disabled so they cannot multiply format attempts.
The default initial discovery plus two residual rounds remains at most three
logical generation calls, with up to six physical attempts when format repair
is needed. Experiment requests and usage are recorded separately.

`GenerationResult.metadata` and fitted `model.metadata_["generation_history"]`
record configuration, sampling indices/counts, token-count method, attempts and
available provider usage. Residual indices refer to that residual input; its
`source_document_indices` map them to the original training documents. Saved
models retain this audit without raw responses or training text. For failure
diagnosis, `generator.last_generation_` holds the latest call's attempt responses
in memory; these may contain input-derived text and are not automatically saved.

The provider/model capability registry was checked on 2026-09-06 against the
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1),
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[GPT-4o](https://developers.openai.com/api/docs/models/gpt-4o), and
[GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini) model pages.
The GPT-5.6 Luna entry records its 1,050,000-token context and supported
reasoning efforts for the sensitivity experiment. A different deployment may
have different limits; unknown models require explicit values.
The dated inventory snapshot and its official source URLs are kept in
[`docs/MODEL_CATALOG_SNAPSHOT_20260906.json`](docs/MODEL_CATALOG_SNAPSHOT_20260906.json);
the runtime copy is shipped as [`hybrid_topic/model_catalog.json`](hybrid_topic/model_catalog.json).
Refresh that file only from official provider inventories and retain its
inventory date. Capability provenance and inventory refresh dates are separate.
New experiment runs copy the actual selected JSON, hash it, use that frozen copy
for every fit, and reject changes. The completed September 6 batches retain their
original Python capability snapshots and results.

### Provider adapters and validation scope

| Path | Current evidence |
|---|---|
| OpenAI GPT-4.1 mini and GPT-5.6 Luna | Two completed 20-fit cloud batches; audited predictions and metrics |
| Claude | Request/count protocol tests and real SDK serialization with offline HTTP; no account-backed generation validation |
| DeepSeek and xAI | Provider field mapping and offline request tests; user supplies a deployment-specific input counter; no account-backed generation validation |

Install Claude support with `python -m pip install '.[anthropic]'` (tested with
`anthropic==1.4.0`; no `tiktoken` dependency). For example, with your deployment's
actual context configured:

```python
from hybrid_topic.backends import AnthropicTopicGenerator

generator = AnthropicTopicGenerator(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    context_window=200_000,
    reasoning_effort="low",
)
```

Claude uses `output_config.format` for schema output and `output_config.effort`
for reasoning. For a deployment without native schema support, explicitly set
`response_format_mode="json"`; local field validation and bounded repair still
apply. `request_options={"thinking": {"type": "adaptive"}}` can pass a supported
native thinking setting. Supported options depend on the exact model; listing a
model in the catalog does not mean every option is available.

`OpenAICompatibleTopicGenerator(provider="deepseek", ...)` selects `json_object`
and `max_tokens`. Its `request_options` supports sampling settings, for example
`{"temperature": 0.2, "extra_body": {"thinking": {"type": "disabled"}}}`.
`provider="xai"` selects the xAI endpoint. Both require `input_token_counter` and
`token_count_source`; see [the counter contract](docs/MODEL_CATALOG_SOURCES.md).
Reasoning effort must appear in the selected catalog entry's `reasoning_efforts`:
`[]` means unsupported, while a missing/null field means unverified. Add verified
capabilities to your JSON rather than sending guessed reasoning options.

`request_options` accepts sampling/provider settings, and rejects attempts to
override model, messages, output limits, JSON contract, streaming, or tools. It
does not promise to accept every field from every provider API. Responses are
still checked for refusal, truncation and topic fields.

The September release and paper make cloud-path claims only. Experimental local
adapters remain in source for future validation; local experiment artifacts are
excluded from the release result tables.

This is `context-budget-whole-documents-v1`, a new generation-input contract.
Earlier local experiments used a 120-document /
500-character pack and original budgets. They are not results for this new
configuration, and their retired runner is not the current experiment launcher. Current comparison
conditions are specified in [the release protocol](docs/EXPERIMENT_PROTOCOL_RELEASE.md).

New documents are encoded **one at a time** and query **frozen training references**.
This prevents query batch size, padding, order, or composition from changing the
reference calculation. Use a deterministic embedding backend with unchanged
weights and preprocessing; a stochastic custom encoder cannot offer repeatable
predictions. Singleton encoding trades some batching throughput for consistent
query behavior. `transform` never generates
topics or changes the training results. Documents with no positive evidence
through the fixed reference threshold remain `-1` (unassigned).

The current training workflow is `bounded-residual-reference-diffusion-v2`;
the frozen-reference inference rule remains `frozen_reference_one_step_v1`.
Earlier `initial-codebook-reference-diffusion-v1` saves remain loadable and keep
their original workflow identity when saved again.
Training diffusion and reference-based inference are different operations, so
calling `transform` on training text need not reproduce `fit_transform` labels.
Its held-out quality must be evaluated independently of archived results.

Model saves contain topic/reference vectors, reference topic scores, configuration
and topic counts in JSON. They exclude training text, API clients and credentials.
Load with the same embedding model/version and preprocessing; an identifier
check catches declared mismatches but cannot verify model weights. Custom backends
must supply a stable `embedding_model_id` if they do not expose `model_name`.
Loaded models can show topic summaries and assign new documents; original
training text is only available in separately exported results. No N-by-N graph
is retained in the saved model, although fitting still constructs a dense graph.


## Development and repository layout

```bash
python -m pip install -e '.[openai,anthropic]'
python -m unittest discover -s tests
```

Provider tests use offline responses. Optional SDK checks require their corresponding extras. Current modules are:

- `hybrid_topic/`: public lifecycle, backend adapters, shared topic types, diffusion,
  residual stopping, result export and evaluation definitions.
- `examples/`: current offline workflow and notebook.
- `experiments/`: current release runner, frozen data preparation and scoring.
- `docs/EXPERIMENT_PROTOCOL_RELEASE.md`: frozen release comparison conditions.
- `paper/`: the article, figures, numerical results and coded prediction evidence.

The two deliverables are a usable GitHub release and a corresponding arXiv paper.
Experiment results are associated with their recorded data, configuration and source version. Historical results are not merged
into the new Hybrid Topic averages.


### Topic-generation default

The release uses a compact set of recurring topics as its default discovery instruction. Topic count remains open. The manuscript reports this configuration as the main experiment and a complete neutral-prompt sensitivity comparison in Section 5.4. See [the comparison data](paper/results/prompt_sensitivity_20260920/README.md) for both conditions.

## Citation

Use the repository citation menu or [CITATION.cff](CITATION.cff) to cite the software. The manuscript is included in `paper/`; its arXiv link will be added after publication.
