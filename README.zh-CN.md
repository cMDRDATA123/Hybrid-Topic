# Hybrid Topic

[English](README.md) | 简体中文

Hybrid Topic 是一个可以在 Python 和 Jupyter 中使用的主题建模工具原型。它先用大语言模型发现主题，生成主题名称、定义和关键词，再通过语义嵌入与图传播为文档分配主题。你可以直接输入文本进行探索，也可以指定分析方向，或提供已有的主题方案。

系统支持查看主题与文档结果、导出表格、保存模型，以及使用同一套主题分析后续文本。

[中文论文](paper/hybrid_topic_zh.pdf) · [英文论文](paper/hybrid_topic_en.pdf) · [入门 Notebook](examples/quickstart.ipynb) · [实验复现说明](docs/REPRODUCIBILITY.md) · [下载安装包](https://github.com/cMDRDATA123/Hybrid-Topic/releases/tag/v0.1.0)

**作者：Jiashuo Ren，University of Stirling。** 代码采用 [MIT 许可证](LICENSE)；论文和数据的许可说明见 [NOTICE.md](NOTICE.md)。

## 可以用它做什么

- **自动发现主题**：输入新闻、评论或其他文本，获得带名称、定义和关键词的主题集合。
- **指定分析方向**：例如要求模型重点关注消费者遇到的问题。
- **使用已有主题**：提供自己的主题名称、定义和关键词，将文本分配到既定主题。
- **查看和导出结果**：检查主题规模、文档分配与分数，导出 CSV 和 JSON。
- **保存并继续使用**：保存模型后，对新文本进行分配，沿用已有主题。

默认提示词要求生成一组精简、反复出现的主题（a compact set of recurring topics），具体数量由模型根据文本决定。论文将这一配置作为主实验，并在第5.4节完整比较中性提示词下的结果，详见[提示词比较数据](paper/results/prompt_sensitivity_20260920/README.md)。

## 安装与快速开始

项目声明支持 Python 3.10 及以上版本，当前发布验证使用 macOS 和 Python 3.12。下面的环境激活命令适用于 macOS/Linux。

```bash
git clone https://github.com/cMDRDATA123/Hybrid-Topic.git
cd Hybrid-Topic
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m examples.basic_workflow --output tmp/offline_example
```

最后一条命令会运行一个完全离线的示例，使用预先给定的主题和简易文本特征，演示建模、导出、保存、重载和新文档分配。运行结束后，结果保存在 `tmp/offline_example`。

如需使用大语言模型自动发现主题，并使用真实语义编码器，安装相应依赖：

```bash
python -m pip install -e '.[embeddings,openai]'
```

如需在 Jupyter 中跟随入门教程：

```bash
python -m pip install -e '.[notebook]'
python -m jupyterlab examples/quickstart.ipynb
```

入门 Notebook 的默认流程可离线执行，其中使用真实模型的可选示例已单独标明。安装依赖本身不会调用生成服务；首次加载未缓存的编码模型可能需要下载权重。使用云端生成时，选中的文本会发送到相应服务商，并产生 API 用量。

## 运行云端示例

安装 `embeddings` 和 `openai` 依赖，并在环境变量中设置 `OPENAI_API_KEY` 后，运行：

```bash
python -m examples.cloud_workflow --output my_cloud_run --allow-download
```

该示例使用十二条合成产品评论，完成主题发现、结果导出、模型保存与重载，并检查单篇和批量查询的结果一致性。编码模型已经缓存时，可以省略 `--allow-download`。每次运行请使用新的输出目录。

默认生成模型为 `gpt-4.1-mini-2025-04-14`。切换部署时，可用 `--model` 和 `--context-window` 指定模型及实际上下文容量。主题生成通过云端 API 完成，文本编码在配置的本地设备上运行。

分析自己的 CSV 文件：

```bash
python -m examples.cloud_workflow \
  --csv comments.csv --text-column text \
  --output my_comments_run --allow-download
```

通过 `--direction` 指定可选分析方向，例如：

```bash
python -m examples.cloud_workflow \
  --csv comments.csv --text-column text \
  --direction "重点识别消费者遇到的问题" \
  --output my_focused_run --allow-download
```

## 在 Python 中使用

以下示例从 `comments.csv` 的 `text` 列读取文本。先将 `OPENAI_MODEL` 和 `OPENAI_API_KEY` 设置为你使用的模型与密钥。

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
    # 已核实的模型可以从目录读取上下文容量。
    # 未收录的部署请填写实际 context_window。
    # 可选：catalog_path="config/model_catalog.json",
    # 可选分析方向；通用主题发现时可省略：
    topic_instruction="重点识别消费者遇到的问题",
)
model = HybridTopic(generator=generator, embedding_model=embedder)
documents = read_documents("comments.csv", text_column="text")
topics, scores = model.fit_transform(documents)

# 查看并导出结果
model.get_topic_info()
model.get_document_info()
model.export("my_results")
model.save("my_model")

# 重载后分析新文本
loaded = HybridTopic.load("my_model", embedding_model=embedder)
new_result = loaded.transform_result(["电池耗电太快了"])
new_result.get_document_info()
```

如果已有主题方案，使用 `model.fit(documents, codebook=[Topic(...)])`。这一路径直接使用给定主题，跳过主题生成和补充。每个 `Topic` 包含 `topic_id`、`name`、`definition` 和 `core_keywords`。

结果中的主题编号从 `0` 开始，`-1` 表示未分配。分数用于描述模型的分配依据，尚未校准为概率。

### 默认建模参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `p` | `0.95` | 按文档相似度的分位数筛选图中的连接 |
| `seed_per_topic` | `5` | 每个主题选择的种子文档数 |
| `diffusion_rate` | `0.70` | 每次更新中邻居分数的权重 |
| `rounds` | `10` | 图传播的最大更新次数 |
| `max_residual_rounds` | `2` | 未分配文本触发主题补充的最大轮数 |

这些是固定的经验默认值，用户可以显式调整。生成模式下，首次发现主题后，系统最多再对未分配文本进行两轮主题补充。剩余文本太少或没有有效新主题时，会提前结束。可将 `max_residual_rounds` 设为 `0` 或 `1`，减少补充轮数。一次拟合失败时，先前已拟合的模型状态会保留。

### 新文本分配与保存复用

`transform` 对新文本逐篇编码，再参照拟合时保存的信息完成分配，主题集合保持固定。在编码器、权重和预处理保持一致且具有确定性的条件下，同一文本的分配结果不受同批其他文本影响。没有获得正分数依据的文本保留为 `-1`。

建模阶段会在训练文档之间传播分数；新文本查询使用固定参考信息。因此，对训练文本再次调用 `transform`，其结果可能与原先的 `fit_transform` 不同。

保存文件以 JSON 记录主题和参考向量、参考主题分数、配置及主题数量。API 客户端、密钥和训练原文不写入模型文件。重载模型可查看主题汇总并分配新文档；需要保留训练原文与分配关系时，请另外导出结果。

重载时应使用相同的编码模型、版本和预处理方式。系统检查声明的编码器标识；自定义编码器如果没有 `model_name`，需要提供稳定的 `embedding_model_id`。模型保存后保留参考信息，拟合时构建的完整文档图无需继续保存。

## 上下文预算与格式重试

系统从可替换的 [`hybrid_topic/model_catalog.json`](hybrid_topic/model_catalog.json) 读取已核实的模型能力，包括上下文容量、输出上限、分词器和推理选项。可通过生成器的 `catalog_path` 参数或 `HYBRID_TOPIC_MODEL_CATALOG` 环境变量指定更新后的目录。

对于未收录或容量尚未核实的部署，需要填写实际 `context_window`。OpenAI 模型的分词器未收录时，还需指定已核实的 `tokenizer_encoding`。Claude 使用服务商的 `messages.count_tokens` 接口。其他兼容接口需要提供 `input_token_counter` 函数及说明计数方法的 `token_count_source`。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `context_window` | 已知模型自动读取 | 所用部署的实际上下文容量 |
| `input_context_ratio` | `0.75` | 完整输入占上下文的比例上限，可在 `(0, 0.75]` 内设置 |
| `max_output_tokens` | `16384` | 最大输出 token 数；超过已知模型上限时会提示并限制到该上限 |
| `safety_margin_tokens` | `1024` | 输入与输出之外的额外预留空间 |
| `max_format_retries` | `1` | JSON 或主题字段不符合格式时，额外修复尝试的次数；`0` 表示关闭 |
| `sampling_seed` | `42` | 输入超出预算时，用于复现文档抽样顺序的随机种子 |

完整输入预算取以下两者的较小值：

```text
floor(context_window * input_context_ratio)
context_window - effective_max_output_tokens - safety_margin_tokens
```

例如，上下文为 1,000,000 token 时，默认输入上限为 750,000 token。上下文为 32,768 token、预留输出为 16,384 token、安全余量为 1,024 token 时，输入上限为 15,360 token。输出设置代表最大允许长度。如果配置没有留下输入空间，系统会明确报错。

输入计数包含提示指令、已有主题、文档编号、文档正文和预留的格式修复说明。OpenAI 使用 `tiktoken`，加上结构化输出及消息开销估计，并记录请求返回的实际用量。Claude 通过官方接口计算消息、系统文本、推理设置和输出格式的 token 用量；计数请求也会发送候选文本，其中可能包括最终未进入生成请求的文档。计数失败会停止生成，大语料抽样可能产生多次计数请求。

兼容接口的自定义计数函数接收完整请求参数，包括 `extra_body`。计数应覆盖所用模型的分词器、聊天模板和输出格式开销；系统检查返回值类型，计数方法的准确性由具体实现保证。

候选文本在预算内时全部送入；超出预算时，按固定随机顺序选取能够放下的完整文档。这种选择可能更容易容纳短文档，抽样过程不读取类别标签。没有文档能够完整放入时会报错。所有输入文档都会参与后续编码与分配，包括未被选入生成请求的文档。

格式修复沿用相同的已选文档，并附加固定修复说明。无效 JSON 或主题字段可以触发修复；服务商拒绝、输出截断和传输错误会直接返回给调用方。SDK 的传输自动重试已关闭。按默认设置，首次发现加两轮主题补充最多产生三次逻辑生成调用；每次都触发一次格式修复时，最多六次实际生成请求。

`GenerationResult.metadata` 和拟合后的 `model.metadata_["generation_history"]` 记录配置、抽样位置与数量、计数方法、尝试次数和可用的服务商用量。补充阶段的 `source_document_indices` 可映射回原始训练文档。模型保存时保留这些记录，并去除原始返回与训练正文。排查失败时，可查看内存中的 `generator.last_generation_`；其中可能包含由输入衍生的文本，不会自动保存。

### 更新模型能力目录

目录中的能力记录与核查日期应一起维护。当前目录保留了 2026年9月6日的核查记录；Luna 实验条目记录的上下文容量为 1,050,000 token。不同部署可能具有不同限制，使用时应与实际服务配置一致。

- [带日期与来源的目录快照](docs/MODEL_CATALOG_SNAPSHOT_20260906.json)
- [模型来源与计数接口说明](docs/MODEL_CATALOG_SOURCES.md)
- [运行时目录](hybrid_topic/model_catalog.json)

更新能力信息时应使用服务商官方资料。新实验会保存所选目录的副本，在同一批运行中固定使用。既有实验继续保留各自的原始配置与结果。

## 模型服务适配与验证范围

| 路径 | 当前验证情况 |
|---|---|
| OpenAI GPT-4.1 mini、GPT-5.6 Luna | 两批各20次主实验，已核对预测与指标；另有完整中性提示词比较 |
| Claude | 已进行请求、计数协议与离线 HTTP 下的实际 SDK 序列化测试；真实账号生成待验证 |
| DeepSeek、xAI | 已进行字段映射与离线请求测试；需提供部署专用输入计数函数，真实账号生成待验证 |
| 本地模型 | 源码保留实验性适配器，端到端验证属于后续工作 |

安装 Claude 依赖：

```bash
python -m pip install '.[anthropic]'
```

Claude 适配器的协议验证使用 `anthropic==1.4.0`，不依赖 `tiktoken`。以下示例需与所用部署的实际上下文及支持选项对应：

```python
import os
from hybrid_topic.backends import AnthropicTopicGenerator

generator = AnthropicTopicGenerator(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    context_window=200_000,
    reasoning_effort="low",
)
```

Claude 通过 `output_config.format` 设置结构化输出，通过 `output_config.effort` 设置推理强度。部署未支持原生结构化输出时，可显式设置 `response_format_mode="json"`，继续使用本地字段检查和有限次数的修复。`request_options={"thinking": {"type": "adaptive"}}` 可传入部署支持的原生推理设置。具体可用选项取决于模型。

`OpenAICompatibleTopicGenerator(provider="deepseek", ...)` 使用 `json_object` 和 `max_tokens`；其 `request_options` 可传入部署支持的采样等设置，例如 `{"temperature": 0.2, "extra_body": {"thinking": {"type": "disabled"}}}`。`provider="xai"` 则选择 xAI 接口。两者均需提供 `input_token_counter` 和 `token_count_source`，详见[计数接口说明](docs/MODEL_CATALOG_SOURCES.md)。

推理强度应包含在模型目录条目的 `reasoning_efforts` 中。空列表 `[]` 表示不支持；缺失或 `null` 表示尚未核实。`request_options` 接受支持的采样和服务商参数，模型、消息、输出上限、JSON 合同、流式输出及工具调用等受系统管理的字段会受到保护。

## 当前使用范围

- 语义编码器当前按 **512 token** 截断输入，这与生成模型的上下文预算分别设置。
- 拟合使用稠密文档图，文档数较大时需评估内存需求。
- 发布验证基于 macOS/Python 3.12；论文实验使用固定的中英文语料子集。长文档、大规模运行、其他语言和本地生成的质量需要进一步验证。
- 当前训练流程为 `bounded-residual-reference-diffusion-v2`，新文本查询规则为 `frozen_reference_one_step_v1`。旧版 `initial-codebook-reference-diffusion-v1` 保存文件仍可加载，并保留其原有版本身份。

## 论文与实验复现

论文主实验使用精简提示词，补充比较完整报告中性提示词下的主题数量与分组表现。各批结果与其数据、配置和代码版本对应。

- [主实验完整结果](paper/results/topicgpt_aligned_20260911/tables.md)：包括所有方法和运行。
- [精简与中性提示词比较](paper/results/prompt_sensitivity_20260920/README.md)：包括80次最终拟合及全部方法配置。
- [公开预测数据](paper/results/public_predictions_20260911/README.md)：包含编码后的参考标签与预测，保留评分所需信息，省去原始语料正文。
- [复现指南](docs/REPRODUCIBILITY.md)：分别介绍汇总表重建、逐篇预测评分，以及重新生成主题所需的条件。
- [发布验证记录](docs/RELEASE_VALIDATION.md)：安装、示例、测试与评分核对结果。

例如，安装基础依赖后，可以从仓库根目录重新评分：

```bash
python -m experiments.score_public_predictions \
  --bundle paper/results/public_predictions_20260911 --out tmp/rescored_public
python -m scripts.build_aligned_paper_results \
  --per-run tmp/rescored_public/per_run_metrics.csv --out tmp/rescored_public_tables
```

这一路径使用公开数值文件，无需 API、原始文本或模型下载。请使用新的输出目录。完整重新生成实验需要复现指南所列的原始研究材料。

## 开发与目录结构

```bash
python -m pip install -e '.[openai,anthropic]'
python -m unittest discover -s tests
```

服务商请求测试使用离线响应，部分 SDK 检查需要安装相应依赖。

- `hybrid_topic/`：模型生命周期、生成适配器、图传播、主题补充、结果导出与指标。
- `examples/`：离线示例、云端示例和入门 Notebook。
- `experiments/`：实验运行、数据准备与评分程序。
- `scripts/`：结果汇总、绘图及目录更新等辅助程序。
- `docs/`：实验协议、模型能力来源与复现说明。
- `paper/`：论文、图表、数值结果和公开预测数据。

## 引用与许可

使用 GitHub 仓库的引用入口或 [CITATION.cff](CITATION.cff) 引用本软件。论文位于 `paper/`，arXiv 发布后将补充链接。

代码采用 [MIT 许可证](LICENSE)。论文与数据遵循 [NOTICE.md](NOTICE.md) 中的说明。
