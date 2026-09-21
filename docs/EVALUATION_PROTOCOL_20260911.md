# 2026-09-11 评估定义修订：与 TopicGPT 对齐

## 当前决定与适用范围

用户明确要求“指标和现有研究对齐即可”。论文主指标从此前的全局 purity / inverse purity 调和平均，统一改为 TopicGPT 已公开实现中的按真实类别规模加权的最佳类别—预测分组 F1。版本为 `topicgpt-weighted-best-f1-v1`，不另造综合评分。覆盖率并列，ARI、NMI 保留为辅助指标。

本文件修订 [原发布实验合同](EXPERIMENT_PROTOCOL_RELEASE.md) 的**报告指标定义**。原合同、冻结评分器、预测、模型和输入文件不改写。两种公式的结果已被查看后才作出此次决定，因此不把新定义追述为实验开始前已冻结的选择。

## 公式与来源

设 n_ck 为真实类别 c 与预测分组 k 的交集数量，n_c、n_k 为各自大小，N 为总文档数：

`HMP = sum_c (n_c / N) * max_k [2*n_ck / (n_c + n_k)]`

采用 [TopicGPT 论文](https://aclanthology.org/2024.naacl-long.164/) Appendix I 和 [calculate_purity 固定代码版本](https://github.com/chtmp223/topicGPT/blob/450564466abe72091797c728a90cfeec8ba3d651/topicgpt_python/utils.py)。原函数已直接执行，并与独立公式在全部 240 个评分位置核对，来源及验证见 [公式评分记录](../paper/results/topicgpt_formula_20260910/manifest.json)。对齐具体实现不等于完成与 TopicGPT 模型的同条件实验比较。

旧公式 `2PR/(P+R)`（P、R 分别为全局 purity 与 inverse purity）保留为历史/敏感性结果。两者不是相同数值公式。

## 不变的实验语义

- 两批云端各 20 次拟合，四语料、五 seed、六方法，全部 240 个位置同时采用新主指标。
- eligible test、AG News 的 799 条有效记录和 `-1` 共享分区规则不变。未分配分区可获得分区质量分数，因此必须同时阅读覆盖率。
- 主题、预测、模型参数、数据、coverage、ARI、NMI 不变，不调用生成 API、不重新拟合、不恢复本地实验。
- BERTopic 仍是跨两批复用的同一组基线；KMeans 仍匹配各次最终主题数。
- 均值和样本标准差按五次运行汇总；不选择最佳 seed、不逐数据集拼接最佳生成器、不声明统计显著或综合最优。

## 当前文件与复现

- [当前主结果](../paper/results/topicgpt_aligned_20260911/tables.md)、[逐次指标](../paper/results/topicgpt_aligned_20260911/per_run_metrics.csv)、[版本清单](../paper/results/topicgpt_aligned_20260911/manifest.json)。
- [旧主结果](../paper/results/tables.md) 与 [完整两公式比较](../paper/results/topicgpt_formula_20260910/README.md) 保留。后者关于“仅作补充”的建议是历史记录，已由本次用户决定取代。
- 修订前稿件与图表快照。
- [公开评分复现入口](REPRODUCIBILITY.md)。评分程序同时输出旧 `harmonic_purity` 与 `topicgpt_weighted_best_f1`，不偷偷改变旧列的意思。当前报告构建器明确把后者映射为主表 HMP，旧值另存 `legacy_global_hmp`；当前结果 manifest 标明列含义和版本。

此次修订只需重算和核验报告，不产生新的质量实验。后续任务仍在 工作手册 维护。
