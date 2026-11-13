# General Algorithm SuperAgent：知识与论文笔记

本文面向团队成员，解释本项目涉及的 Agent、AutoML、实验评估和成本控制，并列出可核对的论文及官方技术资料。**论文中的方法是参考，不表示本项目已经实现或复现论文结果**。代码的实际能力和运行证据以 `STATUS.md`、`AI_WORK.md` 和测试为准。课程约束以根目录 Full Assignment Requirements PDF 为准。

## 1. 问题定义与整体结构

给定未知的分类任务，系统获得数据路径、目标列、模态、训练/验证划分、指标和资源限制，输出可加载的分类模型与预测结果。任务可能是表格、文本或图像，标签可能是二分类或多分类。相同的 Agent 应依据任务的**可观察特征和实验反馈**选择动作，而不能通过数据集名称命中预制方案。

一轮有反馈的实验可以抽象为：

```text
任务规格 + 训练数据摘要 + 历史实验结果
                    ↓
             决策：实验 / 停止
                    ↓
        校验动作与参数 → 执行工具
                    ↓
        验证指标、运行时间或错误信息
                    ↓
          记录结果 → 进入下一轮
```

建议把模型的文本或结构化输出当作**待验证的提议**：控制器负责检查允许的动作、算法和参数范围，再调用工具；工具输出真实结果；下一轮决策读取这些反馈。`TaskSpec`、`ExperimentConfig`、`ExperimentResult` 和 `AgentDecision` 提供跨模块的数据契约。失败也应返回明确错误，不能把失败分数假装为有效结果。

## 2. Agent、工具调用与反馈

### 2.1 ReAct：交替决策、行动和观察

Yao 等人的 ReAct 把语言模型的推理与对环境的行动交替进行；环境反馈可用于调整后续计划。论文在问答、事实核验及交互任务上研究这一模式，不能直接推断它会提升本项目的分类分数。[Yao et al., ReAct, 2022/2023](https://arxiv.org/abs/2210.03629)

对本项目的启发：Agent 可提出候选实验，控制器执行训练工具后把验证分数、耗时和错误返回，再让 Agent 决定下一步。**工程上的关键是校验与记录**：只允许预先定义的工具和参数，设置实验上限，记录决策理由与实际工具输出。无需把模型内部推理过程当作可信证据；应记录可核查的动作、依据和结果。

### 2.2 Toolformer：何时调用工具、参数和结果如何进入上下文

Toolformer 探索通过自监督方式训练模型决定使用哪些 API、何时调用、传什么参数及如何使用返回结果。[Schick et al., Toolformer, 2023](https://arxiv.org/abs/2302.04761)

本项目可借鉴的是工具接口设计，而不是宣称复现 Toolformer 训练：定义工具名称、输入字段、可接受的类型和范围；解析模型输出；拒绝未知工具与非法参数；把错误反馈给后续决策。工具名称、配置和调用结果需要关联实验 ID，防止把某次结果错归到另一配置。

### 2.3 Reflexion：把失败经验带到下一轮

Reflexion 研究不更新模型权重、而以语言反馈和记忆影响后续尝试的方案。[Shinn et al., Reflexion, 2023](https://arxiv.org/abs/2303.11366)

这里可记录失败原因，例如类别数不符、缺失目标、资源不足或参数不合法，并反馈给下一轮；如果反馈不能改变后续决策，那么反思只是一段日志，不构成反馈驱动搜索。测试应使用可控的模拟 LLM 输出和真实工具反馈，检查第二轮决策确实能接收前一轮结果，同时保留明确的停止上限。

## 3. AutoML 与实验选择

AutoML 将算法选择、特征处理和超参数选择作为可自动执行的搜索过程。auto-sklearn 论文把多个分类器与预处理方式组织成搜索空间，并研究利用相似数据集上的历史表现和候选模型集成。[Feurer et al., Efficient and Robust Automated Machine Learning, NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html)

本项目的课程约束要求通用性：可以根据列类型、样本数、类别统计、训练时间和验证反馈选择可用方法；不能依据目录或任务名称调用手工制作的特解。当前可用的模型和参数范围应由代码明确声明，不能让 Agent 提议未实现算法后假装训练成功。表格基线使用缺失值填补、类别编码和逻辑回归；文本与图像处理需要对应的独立通用实现。

Snoek 等人研究用贝叶斯优化利用此前实验结果提出下一个超参数配置，并讨论实验耗时不同的情况。[Snoek et al., Practical Bayesian Optimization of Machine Learning Algorithms, 2012](https://arxiv.org/abs/1206.2944) Hyperband 则研究向候选配置分配不同资源并提前停止部分候选，以提高搜索效率。[Li et al., Hyperband, JMLR](https://jmlr.org/papers/v18/16-558.html) 这些方法是后续可选方向；仅设置 `max_experiments` 并不等于已经实现贝叶斯优化或 Hyperband。

适用于本项目的选择原则：

1. 实验开始前确定任务指定指标和验证方法；只有可比较的成功实验才参与选择。
2. 为每次实验记录算法、参数、随机种子、训练资源、耗时、指标和错误；比较时使用相同指标和验证划分。
3. 明确训练时间和模型调用预算；耗尽上限时停止，并从已成功的验证实验中按明确规则选择模型。如果没有成功实验，应报告失败，不能伪造结果。
4. `ExperimentConfig.max_seconds` 是资源限制的契约字段；是否真正强制超时必须以具体运行实现和测试为准。

## 4. 数据划分、数据泄漏与指标

**Train** 用来拟合参数和预处理；**validation** 用来比较配置与选择模型；课程指定的 **test** 仅供最终评估。若任务未给验证集，内部划分策略需要明确配置，并尊重任务可能存在的时间、用户或分组边界；不能因为分层抽样方便就忽视这些边界。

数据泄漏的一种常见形式是先用全量数据学习缺失值统计、缩放参数或特征选择，再切分训练/验证。scikit-learn 官方文档建议在训练数据上拟合变换，并通过 pipeline 保持拟合与预测步骤一致。[scikit-learn：Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html) 本项目的 tabular 预处理放在 sklearn pipeline 中；后续交叉验证也必须确保每折的预处理只拟合该折训练部分。

反复在同一个验证集上比较大量实验会对验证集本身过拟合，得到的最佳验证分数可能偏乐观；Cawley 和 Talbot 分析了模型选择标准的方差与选择偏差。[Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html) 可以在不违反任务规则的前提下使用内部交叉验证或更稳健的选择过程；指定 test 仍须保持隔离。

分类 **accuracy** 是正确预测数除以样本总数；macro F1 先分别计算每个类别的 F1，再对类别取不加权平均。类别不平衡时两个数可能差异明显。课程报告应以每个任务规定的指标为准，辅助指标不能替代规定指标。对未知任务，目标列、类别含义、划分和评价指标若不明确，应停下并标记待确认，不能猜测。

## 5. LLM 用量与预算控制

账户的总额度和课程**每个任务**的计费上限是不同约束。课程限定每任务人民币 20 元，按 2026-10-21 官方标准输入、输出 Token 价计费，不采用缓存价、促销、免费额度或平台实付价；本地部署同样要按该规则记录。模型必须是截止日期前正式发布的 Qwen 或 DeepSeek。官方价格和发布证据未核验时，`configs/model_prices.yaml` 必须留空，正式模式失败关闭。

价格单位为人民币/百万 Token 时，单次调用成本为：

```text
费用 = (输入 Token × 输入单价 + 输出 Token × 输出单价) / 1,000,000
剩余预算 = 任务预算上限 - 所有已确认调用费用
```

请求前需根据可靠的输入 Token 上界和限定的最大输出 Token 计算最坏费用，确保不会超过剩余额度；请求后按服务返回的实际输入/输出 Token 记账并写入审计日志。若用量缺失、返回中断或重试可能已经计费，应保留不确定状态并阻止后续正式调用，不能将未知消耗写成零。并发请求需要统一预留机制，不能让多个请求分别看到相同余额。单次请求可能受到服务端 Token 计数和输出上限语义影响，接入具体供应商时必须查其官方协议并验证预检假设。

用户提供的 API Key 应通过环境变量传入；不写入源码、配置样例、测试夹具、日志或文档。实验日志至少关联模型名、实际 Token、标准价、费用、调用状态和任务 ID。**上面是目标与约束，不代表所有真实供应商调用链已经具备上述保障。**

## 6. 模型持久化与复现

训练和预测必须使用同一套预处理：例如类别编码的映射和特征列顺序应随模型保存。保存实验配置、随机种子、数据划分、软件依赖和日志有助于复现，但随机种子不保证所有硬件和并行运算都逐位一致。`joblib` 和 NPY 的对象反序列化都应只加载可信文件；`allow_pickle=True` 是本地优惠券数据格式的读取要求，不应无条件加载未知来源文件。

验证需要同时覆盖：训练集拟合、独立验证、模型保存与重载、预测输出、test 未被训练流程读取、实际调用使用量缺失与预算不足时失败。测试数据应由测试代码独立合成；真实任务的开发验证只用 train/val。课程正式结果需由最终冻结版本的完整自主运行生成，Mock 或人工挑选的结果不能代替。

## 7. 参考资料与建议阅读顺序

1. Yao et al. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)，2022/2023：理解决策—行动—观察循环。
2. Schick et al. [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)，2023：理解工具选择、参数和返回值。
3. Shinn et al. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)，2023：理解跨尝试的语言反馈。
4. Feurer et al. [Efficient and Robust Automated Machine Learning](https://proceedings.neurips.cc/paper/2015/hash/11d0e6287202fced83f79975ec59a3a6-Abstract.html)，NeurIPS 2015：理解 AutoML 搜索空间与系统设计。
5. Snoek et al. [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)，2012；Li et al. [Hyperband](https://jmlr.org/papers/v18/16-558.html)：了解实验搜索与资源分配的不同思路。
6. Cawley & Talbot. [On Over-fitting in Model Selection and Subsequent Selection Bias](https://www.jmlr.org/papers/v11/cawley10a.html)，2010；[scikit-learn 数据泄漏说明](https://scikit-learn.org/stable/common_pitfalls.html)：理解验证与模型选择的边界。
