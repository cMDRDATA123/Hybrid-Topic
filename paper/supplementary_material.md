# Supplementary methods and analyses / 补充方法与分析

This companion retains complete additional comparisons and implementation details. The main paper presents the system, primary results and fixed-codebook propagation ablation.

## Generation cost accounting / 生成费用核算

Table 5 uses the request records for the twenty fitted runs in each cloud batch. Summing all 55 GPT-4.1 mini request records gives 782,176 input tokens and 19,476 output tokens; all 44 Luna request records give 786,151 input tokens and 37,296 output tokens. Every request has recorded usage. Initial discovery and topic-supplementation requests are both included. Mean cost per fit is the batch estimate divided by twenty, averaging over the four collection sizes in Table 1.

| Generator | Fits | Requests | Input tokens | Output tokens | Historical API estimate (USD) |
|---|---:|---:|---:|---:|---:|
| GPT-4.1 mini | 20 | 55 | 782,176 | 19,476 | 0.344032 |
| Luna | 20 | 44 | 786,151 | 37,296 | 0.2019854 |

The amounts come from the preserved GPT-4.1 mini usage summary and Luna usage summary, calculated at the historical standard uncached rates recorded for the experiments. They are estimates rather than invoices and exclude cache discounts, local embedding, graph computation, and other development runs. They describe the collection sizes and configurations used here. Full cost comparisons across systems would require the same data, hardware, and pricing assumptions.

表5统计两批各二十次拟合中的初始发现与主题补充请求。全部99份请求均有用量记录，累计token与两批汇总一致。每次拟合均值为整批估算除以二十，涵盖表1中不同大小的四份语料。金额采用实验保存的历史标准非缓存费率，属于生成API估算费用；不含缓存折扣、本地编码、图计算和其他开发运行。原始请求与汇总记录保留上述统计范围，跨系统总成本比较需要统一数据、硬件和计价条件。


## English

### S1. Direct assignment and topic supplementation

Table S1 compares assignment and topic supplementation. With GPT-4.1 mini's final codebooks, Hybrid HMP exceeds direct cosine assignment on MASSIVE (0.4896 versus 0.4481) and AG News (0.7177 versus 0.6886). With Luna, the improvement occurs only on AG News (0.6262 versus 0.5883). Direct assignment has higher HMP on Bills and SIB-200 in both batches and on MASSIVE with Luna. It has complete coverage by construction. The ordering of the complete assignment strategies varies across collections.

*Table S1. Component comparison in HMP, mean ± sample SD. Initial and final denote codebooks before and after bounded residual discovery. The direct columns use nearest-topic cosine assignment with the corresponding codebook. All columns use the same eligible test sets.*

<!-- component-table:start -->
| Generator | Collection | Hybrid initial | Hybrid final | Direct initial | Direct final |
|---|---|---:|---:|---:|---:|
| GPT-4.1 mini | Bills | 0.3527 ± 0.0331 | 0.3647 ± 0.0301 | 0.3731 ± 0.0345 | 0.3808 ± 0.0304 |
| GPT-4.1 mini | SIB-200 | 0.5009 ± 0.0442 | 0.5009 ± 0.0442 | 0.5794 ± 0.0447 | 0.5801 ± 0.0454 |
| GPT-4.1 mini | MASSIVE | 0.4877 ± 0.0329 | 0.4896 ± 0.0310 | 0.4363 ± 0.0492 | 0.4481 ± 0.0517 |
| GPT-4.1 mini | AG News | 0.7230 ± 0.0475 | 0.7177 ± 0.0454 | 0.6918 ± 0.0708 | 0.6886 ± 0.0697 |
| Luna | Bills | 0.4160 ± 0.0143 | 0.4181 ± 0.0154 | 0.4524 ± 0.0181 | 0.4581 ± 0.0278 |
| Luna | SIB-200 | 0.4759 ± 0.0145 | 0.4759 ± 0.0145 | 0.5655 ± 0.0255 | 0.5655 ± 0.0255 |
| Luna | MASSIVE | 0.6608 ± 0.0236 | 0.6608 ± 0.0236 | 0.6671 ± 0.0183 | 0.6671 ± 0.0183 |
| Luna | AG News | 0.6262 ± 0.0513 | 0.6262 ± 0.0513 | 0.5883 ± 0.0461 | 0.5883 ± 0.0461 |
<!-- component-table:end -->

Residual expansion produces modest changes. GPT-4.1 mini Hybrid moves from 0.3527 to 0.3647 on Bills and from 0.4877 to 0.4896 on MASSIVE, but from 0.7230 to 0.7177 on AG News. SIB-200 is unchanged. Luna moves from 0.4160 to 0.4181 on Bills, with unchanged HMP on the other collections. Test coverage is unchanged by expansion in both batches. Higher coverage than BERTopic therefore cannot be attributed to the residual stage in these runs.

The comparison separates the observed effects of retained initial versus final codebooks, but does not make every component independent. A changed codebook can change the selected seeds and subsequent propagation. In addition, Hybrid and direct assignment differ in their rejection behavior. Their HMP difference reflects the complete assignment procedures, including treatment of unassigned texts, rather than a pure effect at matched coverage.




### S2. Baseline configuration details

The reused BERTopic baseline uses UMAP `n_neighbors=15`, `n_components=5`, `metric="cosine"`, `min_dist=0.0`, `n_jobs=1`, `random_state=seed` and `transform_seed=42`. HDBSCAN uses `min_cluster_size=10`, `min_samples=None`, `metric="euclidean"`, `cluster_selection_method="eom"`, `allow_single_cluster=False`, `core_dist_n_jobs=4` and `prediction_data=True`. KMeans uses `n_init=10`, the recorded seed and the corresponding generated final topic count. Frozen prediction manifests establish the reused baseline identities.

The generation records retain model identifiers, request configuration, selected documents, token accounting, attempts and outputs. The executed GPT-4.1 mini and Luna runs use the registered `o200k_base` token encoding. 

### S3. Full numeric evidence

[Full result tables](results/topicgpt_aligned_20260911/tables.md) include HMP, coverage, ARI and NMI for all six methods, both generators and every collection. [Per-run metrics](results/topicgpt_aligned_20260911/per_run_metrics.csv) contain 240 rows. [The aggregation manifest](results/topicgpt_aligned_20260911/manifest.json) records source and builder identities. Repeated BERTopic rows identify reused results, not additional baseline evidence. Main-text tables are generated from the same aggregate file using `scripts/update_manuscript_tables.py`.

For sensitivity and provenance, the [legacy global-HMP tables](results/tables.md) retain the earlier definition: the harmonic mean of global purity and inverse purity. The [complete two-formula comparison](results/topicgpt_formula_20260910/README.md) records all 240 configurations. The earlier reporting definition and the amendment are documented in the reproduction materials. Three of the 32 recorded final-Hybrid comparator orderings change: Luna versus BERTopic on AG News, and Luna versus matched KMeans on Bills and MASSIVE. These differences are retained in full; no favorable subset determines the reporting definition.


### S4. Software implementation

The saved model contains topic descriptions, topic and reference embeddings, reference scores, configuration and metadata. It omits raw induction texts, API credentials and encoder weights. Reloading requires the same embedding backend. Document tables preserve texts separately when exported. New-query assignment requires the semantic encoder but no topic-generation call. Its work still grows with the number of retained references, and singleton encoding trades some batching throughput for stable query behavior.

A separate acceptance run on twelve synthetic comments produced three topics and completed discovery, export, saving, reloading and batch-consistency checks with one generation request. An isolated installation also executed the eight code cells of the quickstart notebook without errors. These checks support the runnable workflow. They are excluded from the quality tables and do not constitute a user study.

### S5. Generation configuration

The generator receives induction documents without benchmark labels, category names or a target topic count. It returns topic IDs, names, definitions and keywords. Each field is validated before assignment. Invalid JSON or missing required fields permits one additional format attempt by default, configurable by the user. Repair retains the same selected documents. Refusal, truncation and transport errors do not trigger this format retry.

The complete input is limited to 75% of the configured context capacity, rounded down, or the capacity remaining after reserving the output ceiling and safety allowance, whichever is smaller. The default output ceiling is 16,384 tokens and the safety allowance is 1,024 tokens. Packing accounts for instructions, document identifiers, an existing codebook when applicable, and repair overhead. Every candidate document is included when the complete request fits. Otherwise, whole documents are selected in a fixed random order under the budget. This procedure can favor shorter documents and does not enforce language or label proportions. Documents omitted from generation still participate in local embedding and assignment. All logical generation calls in the reported batches included every candidate document, so the experiments do not evaluate constrained-context sampling.


## 中文

### S1. 直接分配与主题补充

表S1比较分配方式与主题补充，逐次配对差值保留在原分析产物中。使用 GPT-4.1 mini 最终编码表时，Hybrid 的 HMP 在 MASSIVE（0.4896 对 0.4481）和 AG News（0.7177 对 0.6886）上高于余弦直接分配。使用 Luna 时，仅 AG News 较高（0.6262 对 0.5883）。直接分配在两批的 Bills、SIB-200，以及 Luna 的 MASSIVE 上取得更高 HMP，且按设计覆盖全部文档。两种完整分配策略的排序随语料变化。

*表S1. 组件对照的 HMP，数值为均值±样本标准差。“初始”和“最终”分别指有限主题补充前后的编码表。“直接分配”列使用对应编码表按最近主题余弦相似度分配。所有列采用相同的有效测试集。*

| 生成模型 | 语料 | Hybrid 初始 | Hybrid 最终 | 直接分配初始 | 直接分配最终 |
|---|---|---:|---:|---:|---:|
| GPT-4.1 mini | Bills | 0.3527 ± 0.0331 | 0.3647 ± 0.0301 | 0.3731 ± 0.0345 | 0.3808 ± 0.0304 |
| GPT-4.1 mini | SIB-200 | 0.5009 ± 0.0442 | 0.5009 ± 0.0442 | 0.5794 ± 0.0447 | 0.5801 ± 0.0454 |
| GPT-4.1 mini | MASSIVE | 0.4877 ± 0.0329 | 0.4896 ± 0.0310 | 0.4363 ± 0.0492 | 0.4481 ± 0.0517 |
| GPT-4.1 mini | AG News | 0.7230 ± 0.0475 | 0.7177 ± 0.0454 | 0.6918 ± 0.0708 | 0.6886 ± 0.0697 |
| Luna | Bills | 0.4160 ± 0.0143 | 0.4181 ± 0.0154 | 0.4524 ± 0.0181 | 0.4581 ± 0.0278 |
| Luna | SIB-200 | 0.4759 ± 0.0145 | 0.4759 ± 0.0145 | 0.5655 ± 0.0255 | 0.5655 ± 0.0255 |
| Luna | MASSIVE | 0.6608 ± 0.0236 | 0.6608 ± 0.0236 | 0.6671 ± 0.0183 | 0.6671 ± 0.0183 |
| Luna | AG News | 0.6262 ± 0.0513 | 0.6262 ± 0.0513 | 0.5883 ± 0.0461 | 0.5883 ± 0.0461 |

主题补充带来的变化较小。GPT-4.1 mini Hybrid 在 Bills 上由 0.3527 变为 0.3647，在 MASSIVE 上由 0.4877 变为 0.4896，在 AG News 上则由 0.7230 变为 0.7177；SIB-200 不变。Luna 在 Bills 上由 0.4160 变为 0.4181，其他语料 HMP 不变。两批测试覆盖率均没有因补充而改变。因此，不能把这些运行中高于 BERTopic 的覆盖率归因于主题补充阶段。

这一比较区分了保留初始编码表与最终编码表的实际效果，但没有使各组件完全独立。编码表改变可能影响种子选择和后续传播。此外，Hybrid 与直接分配的拒绝行为不同，其 HMP 差异反映完整分配过程及对未分配文本的处理，不能解释为相同覆盖率下的纯组件效应。




### S2. 基线配置细节

复用的 BERTopic 基线，其 UMAP 参数为 `n_neighbors=15`、`n_components=5`、`metric="cosine"`、`min_dist=0.0`、`n_jobs=1`、`random_state=seed`、`transform_seed=42`。HDBSCAN 使用 `min_cluster_size=10`、`min_samples=None`、`metric="euclidean"`、`cluster_selection_method="eom"`、`allow_single_cluster=False`、`core_dist_n_jobs=4`、`prediction_data=True`。KMeans 使用 `n_init=10`、记录的种子及对应运行的最终生成主题数。冻结预测清单确认了复用基线的身份。

生成记录保留模型标识、请求配置、所选文档、token 核算、尝试次数及输出。GPT-4.1 mini 和 Luna 的实际运行均使用已登记的 `o200k_base` 编码。

### S3. 完整数值证据

[完整结果表](results/topicgpt_aligned_20260911/tables.md)包含全部六种方法、两种生成模型及每份语料的 HMP、覆盖率、ARI 和 NMI。[逐次指标](results/topicgpt_aligned_20260911/per_run_metrics.csv)共240行。[汇总清单](results/topicgpt_aligned_20260911/manifest.json)记录来源与构建程序身份。重复的 BERTopic 行表示结果复用，不代表额外基线证据。英文主文表格通过 `scripts/update_manuscript_tables.py` 从同一汇总文件生成，中文表格保留相同数字。

作为敏感性分析和来源记录，[旧全局 HMP 表](results/tables.md)保留原定义，即全局纯度与逆纯度的调和平均。[完整两公式比较](results/topicgpt_formula_20260910/README.md)包含全部240个配置。指标修订过程另见复现材料。已记录的32项最终 Hybrid 与对照的排序中，有三项方向改变：Luna 在 AG News 上相对 BERTopic，以及 Luna 在 Bills、MASSIVE 上相对匹配 KMeans。这些差异完整保留，报告定义不由某个有利子集决定。

### S4. 软件实现

保存的模型包含主题描述、主题及参考文档向量、参考分数、配置和元数据，不包含原始建模文本、API 凭证或编码器权重。重载需要相同的编码后端。导出的文档表另行保留原文。新文档分配需要语义编码器，但无需主题生成调用；计算量仍随参考文档数量增加。逐篇编码用部分批量处理效率换取稳定的新文档分配行为。

另一次独立验收在十二条合成评论上生成三个主题，通过一次生成请求完成主题发现、导出、保存、重载及批次一致性检查。隔离安装环境也无错误执行了快速入门 notebook 的八个代码单元。这些检查支持工作流程可运行；它们不进入质量表格，也不构成用户研究。

### S5. 生成配置

生成器接收建模文本，不接收基准标签、类别名称或目标主题数。输出包含主题编号、名称、定义与关键词。分配前逐项检查这些字段。若 JSON 格式错误或缺少必要字段，默认允许再尝试一次格式修复，用户可修改重试次数。修复请求保留相同的已选文档。模型拒绝、输出截断和传输错误不会触发这种格式重试。

完整输入上限取以下两者中较小的值：配置上下文容量的75%（向下取整），或扣除输出上限与安全预留后的剩余容量。默认输出上限为16,384个token，安全预留为1,024个token。输入打包计算指令、文档标识、适用时已有的主题编码表，以及格式修复所需的额外空间。完整请求能够容纳所有候选文档时，全部纳入；否则按固定随机顺序，在预算内选择完整文档。这可能偏向较短文档，也不强制维持语言或标签比例。没有送入生成器的文档仍参与本地编码与分配。本文批次中的全部逻辑生成调用都纳入了所有候选文档，因此实验没有评价上下文受限时的抽样效果。


## Evaluation provenance / 评价定义与版本

### Exact implementation settings / 精确实现设置

Topic descriptions are serialized as `[Topic] {name} [Definition] {definition} [Keywords] {comma-separated keywords}`. The configured generation context capacities were 1,047,576 tokens for GPT-4.1 mini and 1,050,000 for Luna. Run identifiers and document-packing seeds were 11, 23, 37, 53, and 71; these are not API generation seeds. All candidates fitted the input budget in the reported calls.

Residual discovery skips a candidate when its normalized name already occurs or its topic-vector cosine similarity to an existing topic exceeds 0.95. It stops when fewer than two induction documents remain unassigned, no valid topic is accepted, the round limit is reached, or two successive rounds do not reduce the unassigned count. Earlier topics are retained. User-supplied topic sets skip both generation stages. Model weights, assignment rules and experiment outputs were not changed during manuscript revision.

The new-document query uses a weighted lookup over fitted reference scores rather than rerunning the induction update. Re-entering an induction document as a query therefore need not reproduce its induction assignment. The calculation is described in Section 3.4 of the paper.

主题描述使用上述固定模板组合名称、定义与关键词。运行编号及打包种子不代表生成API的随机种子。主题补充的名称去重、余弦阈值及停止条件如上，初始主题始终保留。新文档查询与建模阶段的传播计算不同，因此将建模文档再次输入查询接口不保证重现其建模分配。

The current HMP is the category-size-weighted best-match F measure described in Amigó et al. (2009), Section 4.1, also used under the HMP name by TopicGPT. The implementation was checked against TopicGPT commit 450564466abe72091797c728a90cfeec8ba3d651. The reporting definition was changed after both formulas had been inspected; all methods were rescored without changing predictions or eligible samples. See the [evaluation amendment](../docs/EVALUATION_PROTOCOL_20260911.md). No claim of prospective metric registration is made.

[传播开关完整结果](results/propagation_ablation_20260915/summary.csv) · [配对差值](results/propagation_ablation_20260915/paired_summary.csv) · [实验协议](results/propagation_ablation_20260915/protocol.md)。共同已分配子集作为选择条件下的辅助诊断，用于观察两种设置都可分配的文本，其数值不替代全体测试集指标。


## Generator topic example / 生成器主题案例

MASSIVE has the largest absolute mean HMP difference between the two generators in the main table. We inspect the first scheduled paired run (identifier 11) to describe topic scope and report final topic counts for all five runs, in identifier order 11, 23, 37, 53, 71. GPT-4.1 mini: 11, 11, 11, 11, 14. Luna: 24, 28, 22, 28, 24. English main-text names for Luna topics are translations of the Chinese output.

选取MASSIVE以具体查看主表中生成器平均HMP差异最大的语料；主题描述取既定顺序的首组运行，主题数量则覆盖全部五次运行。来源路径、文件哈希与所引主题字段保存于[案例证据](review_comments_20260915/wp7_case_evidence.json)。

### GPT-4.1 mini

- **Alarms and Reminders**: Setting, checking, and managing alarms, reminders, calendar events, and notifications.
- **News and Social Media**: Requests for news updates, social media posts, status changes, and online information tracking.

### Luna

- **闹钟与定时提醒**: 创建、查看、修改、取消或静音基于时间或周期的闹钟与定时提醒。
- **日历、会议与事件安排**: 在日历中创建、查看、修改或删除会议、约会、生日、聚会及其他日程事件。
- **新闻与时事资讯**: 获取本地、国际、政治、财经、科技、体育或专题新闻的最新报道与摘要。
- **社交媒体操作与动态**: 查看社交平台动态，发布状态、微博、推文或帖子，并发送社交媒体投诉或消息。

## Generation prompts / 主题生成提示词

### Initial discovery / 初始主题发现

The experiments use the following instruction before the document list. “Open-K” means that the number of topics is determined from the documents. The system appends “Induction documents:” followed by numbered documents.

> Create an Open-K topic codebook from the induction documents below. Use no gold labels, benchmark category names, target K, or external taxonomy. Discover a compact set of recurring topics yourself. Each topic must provide a name, definition, and all useful core_keywords. Return JSON only as {"topics":[{"topic_id":"T1","name":"...","definition":"...","core_keywords":["..."]}]}.

初始请求要求模型根据文本自行确定主题，返回主题编号、名称、定义和关键词。系统随后附上带编号的建模文档。正文展示其中关于主题内容的要求，上方保留完整指令及返回格式。

### Topic supplementation / 主题补充

The next instruction is followed by the full current topic list under “Current codebook:” and the unassigned documents under “Residual documents:”.

> Extend the current Open-K codebook using only the residual documents. The full current codebook is shown below. Return an empty topics list when the residual does not contain a recurring theme genuinely absent from that codebook. Do not rename, duplicate, or split an existing topic. Return JSON only as {"topics":[{"topic_id":"R1","name":"...","definition":"...","core_keywords":["..."]}]}.

补充请求同时提供现有主题表与未分配文档，要求只增加尚未涵盖的重复主题。没有这样的主题时返回空列表，已有主题保持原有名称和范围。

The public interface can insert a user-supplied analytical direction before the document or topic-list blocks. The experimental requests use the general discovery instructions above. The two cloud batches contain 99 recorded request files; all use the same initial or supplementary instruction prefix. File identities and checked prefixes are retained in the [prompt evidence record](review_comments_20260915/wp8_prompt_evidence.json).


## Prompt sensitivity: compact and neutral / 提示词敏感性比较

The main experiments, propagation ablation, and historical generation-cost table use the compact prompt. The neutral comparison repeats forty fits, with all results retained. Only the initial discovery sentence group changes; the residual prompt, data, seeds, budget, and assignment settings remain the same. The generator can vary between calls, so this comparison measures the complete prompt condition together with that variation. Baseline scores are reused under the common protocol.

主实验、传播消融及历史生成费用表采用精简提示词。中性比较重新完成四十次拟合，完整保留所有结果。初始主题发现的一组语句发生变化，残差提示词、数据、随机种子、预算和分配设置保持一致。生成模型在不同调用间存在波动，因此比较反映整组提示要求及其生成波动。基线在共同协议下复用。

Compact / 精简：
> Discover a compact set of recurring topics yourself. Each topic must provide a name, definition, and all useful core_keywords.

Neutral / 中性：
> Identify the recurring topics in the documents. Let the content of the documents guide the number and specificity of the topics. For each topic, provide a name, a definition, and relevant core keywords.

| Generator | Collection | Topics: compact | Topics: neutral | HMP: compact | HMP: neutral |
|---|---|---:|---:|---:|---:|
| 4.1 mini | Bills | 13.6 | 16.8 | 0.3647 | 0.3938 |
| 4.1 mini | SIB-200 | 9.8 | 10.2 | 0.5009 | 0.4889 |
| 4.1 mini | MASSIVE | 11.6 | 21.0 | 0.4896 | 0.5981 |
| 4.1 mini | AG News | 10.6 | 11.6 | 0.7177 | 0.6825 |
| 5.6 Luna | Bills | 21.8 | 25.0 | 0.4181 | 0.4316 |
| 5.6 Luna | SIB-200 | 16.0 | 19.0 | 0.4759 | 0.4707 |
| 5.6 Luna | MASSIVE | 25.2 | 28.0 | 0.6608 | 0.6676 |
| 5.6 Luna | AG News | 16.6 | 22.6 | 0.6262 | 0.4932 |

[All 80 final runs](results/prompt_sensitivity_20260920/per_run.csv), [means and sample SD](results/prompt_sensitivity_20260920/summary.csv), [all 480 method configurations](results/prompt_sensitivity_20260920/all_methods_per_run.csv), and [all 96 method summaries](results/prompt_sensitivity_20260920/all_methods_summary.csv) provide topic counts, HMP, coverage, ARI, and NMI. Topic counts refer to the final saved codebook, including residual additions. SD uses the sample definition across five runs. Full provenance and frozen-source reproduction instructions are in the [comparison README](results/prompt_sensitivity_20260920/README.md).

逐次数据及汇总覆盖全部模型与语料。最终主题数由保存的主题集合直接计数，包含残差补充；标准差按五次运行计算样本标准差。六种方法均完整报告，复用基线行保留原分数。

### Percentile threshold / 分位阈值

P95 concentrates graph connections on the strongest approximately 5% of document-pair similarities within each collection. The fixed value follows the development default (D-011). The development record describes 45 engineering checks using three previously seen training collections, five historical topic sets, and the 92.5th, 95th, and 97.5th percentiles. Those topic sets had previously been generated from the collections. This record documents the origin of the default; the reported main experiments hold it fixed across collections. The historical record is archived in `archive/release_consolidation_20260906/workspace/docs/PROJECT_MEMORY.md`.

P95按每份语料内部的相似度分布保留关系最强的约5%文档对。具体数值沿用开发默认D-011。开发记录包含三份已有训练语料、五组历史主题和P92.5、P95、P97.5共45次工程检查；这些历史主题此前由相关语料生成。该记录说明默认设置的来源，本文主实验在各语料中统一固定使用P95。
