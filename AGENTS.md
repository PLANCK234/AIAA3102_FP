# 后续 Codex 会话须知

## 目标与硬性约束

构建同一个通用分类 SuperAgent，覆盖未知 tabular、text、image 二/多分类。以根目录 Full Assignment Requirements PDF 为准。正式评估必须完整自主运行，不能由人改代码、指定下一实验或选择模型。不得按数据集名称写特殊分支。正式 test 只能用于最终评估，不得用于训练、调参或模型选择。

初始化过程和当前代码库逐文件说明见 `AI_WORK.md`；接手开发前先读该文件、`TASKS.md` 与 `STATUS.md`。

Agent LLM 仅限 2026-10-21 及之前正式发布的 Qwen / DeepSeek。每任务按 2026-10-21 官方标准输入/输出 Token 价计费，20 元人民币封顶，不使用缓存优惠。所有调用、Token、价格、费用、候选实验、验证结果和关键决定必须真实记录。禁止伪造实验结果或 Token 消耗；用量缺失时正式模式应失败。价格尚未核验，`configs/model_prices.yaml` 留空，严禁编造。

## 代码规范与边界

- Python 3.11+；优先标准库、NumPy、Pandas、scikit-learn、PyTorch；使用 pytest，避免过度抽象。
- 使用共享 `runtime.schemas` dataclass 接口。`agent/` 与 `tools/`/`models/` 不互相导入；`runtime/runner.py` 负责组合。
- `agent/` 属 Member A；`tools/`、`models/` 属 B；`runtime/`、`tests/`、`main.py` 属 C。跨模块改动同步更新接口测试和 `STATUS.md`。
- 不硬编码绝对路径；API Key 仅取环境变量；不提交 `.env`、密钥、缓存或大型模型。
- 不修改、移动或删除原始数据目录或课程 PDF。`outputs/` 是生成产物目录。
- 未完成模块要明确 TODO/抛出 `NotImplementedError`，不假装已经自主规划、完成正式评估或使用了 LLM。

## 测试与交付

在用户的 `coursework` Conda 环境运行 `python -m pytest -q` 和实际数据开发模式验证。单元测试使用合成数据，核对模型重载预测、test 隔离、预算和失败关闭。开发 Mock 运行不可用于课程正式结果。正式版本须增加真实 Agent、官方价格与用量审计、文本/图像、完整自动运行和最终评估流程，再冻结提交。
