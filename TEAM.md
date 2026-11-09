# 团队分工与集成约定

| 成员 | 比例 | 主责任与目录所有权 | 验收标准 |
|---|---:|---|---|
| Member A — Agent Core | 35% | `agent/`：LLM Client、Tool Calling、Controller、Planner、Reflection、实验选择/停止 | 根据任务元数据和**真实**实验反馈自主决定下一轮实验，能调用 ML 工具；记录理由、Token 和费用；Mock 不算正式验收 |
| Member B — ML Engine | 35% | `tools/`、`models/`：数据/模态识别，Tabular/Text/Image 训练、预处理、验证、持久化、预测 | 不依赖 LLM，通过共享接口完成各模态训练、验证、保存、重载与预测；test 不参与选择 |
| Member C — Runtime & Integration | 30% | `runtime/`、`tests/`、`main.py`、`configs/`、`requirements.txt`：CLI、Runner、预算、日志、异常恢复、复现和交付协调 | 统一入口可完整运行；正式价缺失/用量缺失/超预算时失败关闭；自动测试和日志可复现 |

## 模块边界

`runtime/schemas.py` 是唯一的跨包契约。A 输出 `AgentDecision`/`ExperimentConfig`，B 接收 `TaskSpec` 与 `ExperimentConfig` 并返回 `ExperimentResult`，C 串接、记录并执行资源策略。A/B 不互相导入，避免循环依赖。改动契约时由 C 提交兼容性检查，A/B 同步评审。

## 并行开发规则

1. 各成员优先修改自己目录；跨目录变更先在 `TASKS.md` 写清接口及依赖，避免同时改同一文件。
2. 每个功能附最小可运行测试和失败路径；未完成部分显式 TODO 或 `NotImplementedError`，不返回假结果。
3. 原始数据和课程 PDF 不移动、不修改、不删除。任何任务逻辑只能依赖声明的模态、字段、统计和反馈，不根据任务名或目录名分支。
4. A 的 LLM 调用必须经 C 的预算预检与使用量日志；B 的训练不得读取 test；C 在集成测试中检查二者。
5. Member C 负责合并、复现、交付清单与冻结版本核对；A/B 分别提供 Agent 设计、模型实验与局限材料。

## 文档与展示

报告和 Presentation 由三人共同撰写、审核和讲解。A 负责自主决策与资源约束说明，B 负责方法/实验/指标，C 负责日志、复现、提交一致性和排期组织。AI 工具使用声明共同审核。课程正式结果只能来自同一冻结版本的完整自主运行。

## 集成验收

- 单一入口接受未知任务配置，并在无法确定目标/划分/指标时停止。
- Tabular、Text、Image 都走相同 Agent 契约；没有数据集身份分支。
- 指定 test 仅最终评估一次；候选实验只看训练/验证。
- 每次真实 LLM 调用有模型、Token、官方标准价和费用；按任务累计不超人民币 20 元。
- 训练、预处理、模型加载、预测和结果文件可复现；日志能解释每轮选择、失败和终止。
