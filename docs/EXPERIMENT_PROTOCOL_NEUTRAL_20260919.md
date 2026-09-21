# 中性主题提示词实验：neutral-prompt-20260919-v1

用户于2026-09-19授权立即重跑GPT-4.1 mini与GPT-5.6 Luna。采用完整新措辞：

> Identify the recurring topics in the documents. Let the content of the documents guide the number and specificity of the topics. For each topic, provide a name, a definition, and relevant core keywords.

初始提示词其余要求、JSON返回字段和补充主题提示保持原样。这次是整段生成指令的修订，比较解释为中性提示版本的效果；不单独归因于删除compact一词。

沿用四份语料、原建模/测试拆分、五次编号11/23/37/53/71，共40次拟合。模型为gpt-4.1-mini-2025-04-14与gpt-5.6-luna，Luna reasoning low。两者上下文、16384输出上限、75%输入比例、一次格式重试、最多两轮主题补充以及图参数均沿用前批。使用原文档向量及BERTopic基线；主题向量、Hybrid及匹配主题数KMeans重新计算。费用无上限，不依据分数补跑择优。

运行使用独立源码副本，目录runs/neutral_prompt_20260919。由launchctl托管，mini之后排队Luna，共用本地编码器。预测完成后单独进程评分；当前主HMP为加权最佳配对F1，覆盖率并列、ARI/NMI辅助。旧评分器产物单独标为evaluation_legacy，新版主评分在evaluation目录。AG News主要评分保留799篇规则。失败完整记录。

旧批次、论文及其结果表保留。本批完成后比较主题数量、主题范围、HMP、覆盖率、用量与耗时，再更新文章和补充材料；这仍是既有开发材料上的版本复核。常规运行复用既有完整性机制，不另设反复手工哈希检查。
