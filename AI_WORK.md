# AI Work：项目初始化与代码库开发记录

> 记录范围：2026-11-09 的初始化工作。本文说明本次加入的文件、设计取舍、实际验证结果和下一步接手方式。它是开发记录，不是正式课程实验报告。实时进度以 `STATUS.md` 和 `TASKS.md` 为准。

## 1. 初始化前的状态与依据

开始时，工作目录只有两份课程 PDF、`headlines_ctr_clean/`、`in_vehicle_coupon_recommendation/` 和少量系统/编辑器文件，没有 Agent 代码、统一入口、测试或 Git 仓库。初始化过程中没有移动、删除或修改原始数据和 PDF，也没有运行付费 LLM API。

先完整读取 `AIAA3102 - Final Project - Full Assignment Requirements.pdf` 与 `AIAA3102_Final_Project_Preview.pdf`。两份文件有差异时，以 Full Assignment Requirements 为准。由完整要求落实到工程中的关键点是：

1. 同一个 SuperAgent 处理未知的 tabular、text、image 二分类或多分类任务；不能依据数据集身份选择预制特解。
2. 正式结果必须由冻结版本完整自主运行产生，不能人工指定后续实验或最终模型；指定 test 仅用于最终评估。
3. Agent 的 LLM 限定为 2026-10-21 及之前正式发布的 Qwen/DeepSeek；按该日官方标准输入/输出 Token 价计算，每任务不超过人民币 20 元。
4. 模型调用、Token、成本、实验、决策都要留痕；最终交付还包括可运行预测流程、日志、报告、展示与 AI 工具使用声明。

课程未在总要求中为所有未来任务指定统一的字段、格式或指标；这些由任务发布时提供。因此 `TaskSpec` 要求显式写明目标、任务类型、验证划分和指标，未确认信息不应猜测。

## 2. 原始数据探查结果

探查时读取两个数据集自带的 README、优惠券 `info.json`、各文件结构、字段类型和 train/val 标签统计。对 test 只检查文件存在、结构/形状/字段，不使用 test 标签值训练、调参或选择模型。

| 数据 | 已确认的输入、标签、划分 | 已确认的最终指标 | MVP 状态 |
|---|---|---|---|
| 新闻标题 CTR | `headline_A`、`headline_B` 两个文本字段；`label_pairwise=1` 表示 A 的观测 CTR 更高，`0` 表示 B 更高。CSV train/val/test 分别为 8,301/940/3,232 行。 | test accuracy | 元数据写入配置；Text 训练尚未实现。 |
| 车载优惠券 | `C_*.npy` 为 21 列 `object` 特征，`y_*.npy` 为同序整数标签；`1` 接受、`0` 拒绝。train/val/test 分别为 8,117/2,030/2,537 行。特征名由数据 README 提供。 | test accuracy | 已通过通用“特征 NPY + 标签 NPY”表格加载接口运行 train/val 基线。 |

优惠券 train 标签计数为 0:3,503 / 1:4,614，val 为 0:876 / 1:1,154。标题 train 为 0:4,126 / 1:4,175，val 为 0:459 / 1:481。优惠券特征文件按其 README 的要求使用 `np.load(..., allow_pickle=True)`；这只适用于可信本地任务文件。

## 3. 本次创建的代码和文档

### 3.1 跨模块契约与运行层

| 文件 | 加入的内容 |
|---|---|
| `runtime/schemas.py` | 五个 dataclass：`TaskSpec`、`ExperimentConfig`、`ExperimentResult`、`AgentDecision`、`UsageRecord`，以及 JSON 日志序列化函数。`TaskSpec` 必须显式提供分类类型、目标列、指标和验证路径或验证比例；NPY 特征必须配标签路径。 |
| `runtime/runner.py` | 读取任务/运行设置 YAML，按配置文件所在目录解析相对路径；创建唯一运行目录；串联分析、Mock 计划、训练、验证、记录和停止。训练失败会形成失败的 `ExperimentResult` 并写日志。开发 Runner 不读取 `test_path` 或 `test_labels_path`。 |
| `runtime/budget.py` | `Decimal` 成本累计、剩余额度、调用前按输入和最大输出 Token 做预算预检、调用后按实际 Token 记账。拒绝负数、缺失用量、未知模型、超预算以及超过 20 元的任务预算。正式模式要求有模型价格配置。 |
| `runtime/logger.py` | 追加写入 JSONL，记录 UTC 时间和结构化事件。 |
| `main.py` | `run`、`smoke`、`predict` 三个 CLI 子命令；错误以非零退出码报告。`smoke` 在 `outputs/` 下生成独立合成 CSV，不碰原始数据。 |

`runtime.schemas` 是 Agent Core、ML Engine 与 Runtime 的共同接口。Agent 包和 ML 包不互相导入，由 Runner 组合，减少循环依赖和三人并行开发时的接口冲突。

### 3.2 Agent Core

| 文件 | 加入的内容与实际边界 |
|---|---|
| `agent/llm_client.py` | `LLMClient` 协议、响应结构、无 API Key 的 `MockLLMClient`。`RealLLMClient` 明确抛 `NotImplementedError`，目前不会发付费请求，也不会伪造 Token。 |
| `agent/planner.py` | `MockPlanner` 只给 tabular 任务生成固定逻辑回归基线；不调用 LLM，也不根据数据集名称分支。 |
| `agent/controller.py` | `initial → analyzed → planned → trained → complete/failed` 状态机与实验次数上限。 |
| `agent/reflection.py` | 目前是停止钩子；会标明 Mock 基线结束或失败，不具备基于反馈提出新实验的能力。 |

因此“计划”在目前只是固定开发基线，不能声称为课程要求的自主规划。正式模式在 `runtime/runner.py` 中主动拒绝运行，即使日后价格表填好，在真实适配器、反馈搜索和审计接线完成前仍需保持该限制。

### 3.3 ML Engine

| 文件 | 加入的内容 |
|---|---|
| `tools/data_analyzer.py` | CSV 或配对 NPY 读取；检查显式目标列、数据/标签形状、空数据、缺失标签；输出行列数、字段类型、缺失计数和类别计数。对 NPY 中可完整转为数值的列做通用类型转换。 |
| `tools/preprocessing.py` | 按训练特征类型构造 `ColumnTransformer`：数值列中位数填补；类别列众数填补与未知类别可容忍的 one-hot。预处理在训练 pipeline 中拟合。 |
| `tools/training.py` | 只在 train/val 上训练与验证；支持任务提供独立 val，或明确配置 `validation_fraction` 后按标签分层切分。检查类别数与验证字段。 |
| `tools/evaluation.py` | 计算 accuracy 与 macro F1；任务指定哪个指标由 `TaskSpec.metric` 决定。 |
| `models/tabular.py` | scikit-learn `LogisticRegression` 基线；用 joblib 保存 pipeline、特征列顺序与目标列；重新加载后对 CSV/NPY 特征预测并输出一列 `prediction`。 |
| `models/text.py`、`models/image.py` | 明确 TODO/`NotImplementedError` 接口；没有伪造文本或图像训练结果。 |

模型文件包含已经拟合的预处理和分类器，预测时沿用训练时的列顺序；代码不会在预测阶段重新拟合。joblib 基于 pickle，模型文件应只从可信运行产物加载。

### 3.4 配置、测试和协作文档

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | 开发模式、1 次 Mock 实验、20 元任务预算、输出目录和价格表位置。 |
| `configs/model_prices.yaml` | 故意保留 `models: {}`；官方价格及发布日期尚未核验，不能编造。注释列出以后需要的字段。 |
| `configs/coupon_development.yaml` | 优惠券任务的通用 NPY 文件路径、21 个公开字段名、标签和 accuracy；这些是**任务数据配置**，不是代码里的名称特判。 |
| `configs/headlines_development.yaml` | 标题任务的文本字段所在文件、标签和 accuracy；运行时会明确提示 Text 未实现。 |
| `tests/test_schemas.py` | 契约最小示例和无效输入检查。 |
| `tests/test_budget.py` | 空价格表拒绝、成本计算、剩余预算、超额预检、缺失用量拒绝。价格测试值仅为虚构测试夹具，不是官方报价。 |
| `tests/test_pipeline.py` | 合成 CSV/NPY 端到端、模型重载预测、结构化日志，以及不存在的 test 路径不会被开发运行读取。 |
| `requirements.txt`、`.env.example`、`.gitignore` | 依赖清单、未来真实适配器的环境变量示例、密钥/缓存/模型/输出忽略规则。 |
| `README.md`、`TEAM.md`、`AGENTS.md`、`TASKS.md`、`STATUS.md` | 使用指南、三人模块所有权、后续 Codex 约束、带 ID 的任务和进度快照。 |

初始化时没有执行 `git init`；`.gitignore` 已备好，项目是否建立 Git 仓库由团队工作流决定。

## 4. 当前端到端流程与输出

```text
任务 YAML → TaskSpec 校验 → 读取并分析 train
         → MockPlanner 生成 ExperimentConfig
         → 读取 train/val → 拟合预处理与逻辑回归
         → 验证指标 → 保存 model.joblib
         → AgentDecision 停止 → result.json + events.jsonl
```

一个开发运行目录为 `outputs/<UTC时间>-<随机后缀>/`，包含：

- `events.jsonl`：`run_started`、`data_analyzed`、`agent_decision`、`experiment_result`、停止决定、`run_completed` 或 `run_failed`。
- `result.json`：`ExperimentResult` 的状态、验证指标、模型路径、耗时和错误信息。
- `model.joblib`：已经拟合的预处理及模型，供 `predict` 加载。生成的 `outputs/` 文件被 Git 忽略。

Mock 运行的日志中 `llm_calls=0`、`spent_rmb=0`，只表示**确实没有发生 LLM 调用**。这不是正式 Agent 的成本证明。预算模块虽已实现计算与拦截，但尚未接入真实 LLM 请求。

## 5. 环境与复现命令

用户指定使用已有的 Conda **`coursework`** 环境。实际检查为 Python 3.11.16，NumPy、Pandas、scikit-learn、joblib、PyYAML、pytest 均已安装，所以本次没有下载安装依赖，也没有创建 `.venv`。若以后缺包，在该环境内执行 `python -m pip install -r requirements.txt`。

```powershell
conda activate coursework
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m pytest -q
python main.py smoke
python main.py run --task configs/coupon_development.yaml
```

设置线程数为 1 是因为初始化验证时并行启动两个 Python 进程曾触发 `MemoryError`；改为顺序运行并限制 BLAS/OMP 线程后，测试和训练正常通过。若当前 PowerShell 无法 `conda activate`，可用 `conda run -n coursework python ...`。

保存模型的预测示例（替换实际 run ID）：

```powershell
python main.py predict --model outputs/<run-id>/model.joblib --data in_vehicle_coupon_recommendation/C_val.npy --output outputs/validation_predictions.csv
```

这里故意用 **val** 演示。开发阶段不要用 test 进行模型选择或调参。

## 6. 本次验证证据

| 验证 | 观察结果 | 含义 |
|---|---|---|
| `python -m pytest -q` | **8 passed** | 契约、预算、CSV/NPY 基线、重载预测和 test 路径隔离通过。 |
| `python main.py smoke` | 验证 accuracy **0.888889** | 合成 CSV 的开发链路可完整运行。 |
| `python main.py run --task configs/coupon_development.yaml` | 验证 accuracy **0.661576**，macro F1 **0.646667** | 现有优惠券 train/val 表格基线可运行。 |
| 保存模型后 `predict` 读取 `C_val.npy` | 生成 **2,030** 条预测 | 持久化后的预测流程可重新加载。 |
| `--mode formal` | `Formal mode needs verified official model prices`，退出码非零 | 未配置官方价格时失败关闭。 |

这些是**开发验证**，不是完整自主运行的课程成绩，未报告 test accuracy。测试时没有自动调用付费 API。

## 7. 明确未完成的工作

1. 核验允许使用的 Qwen/DeepSeek 模型发布日期及 **2026-10-21 官方标准**输入/输出价格，填写来源；价格未核验前正式模式不得启动。
2. 实现真实 LLM Client、工具调用、逐次调用 Token/费用审计，并把调用前预算预检接到网络请求。
3. 把固定 Mock Planner/停止钩子换成基于任务元数据和实验反馈的多轮自主实验选择、终止与最终模型选择。
4. 实现 Text/Image 训练与预测、更多通用候选算法，以及 `ExperimentConfig.max_seconds` 的实际超时执行。
5. 实现严格隔离的最终 test 评估、异常恢复、冻结版本复现和课程提交产物。未来任务的标签、划分、指标须待发布后确认。

具体责任、依赖与验收标准见 `TEAM.md` 和 `TASKS.md`。接手者修改代码后，应同步更新 `STATUS.md` 与本文相关章节；不要把 Mock 的验证记录描述成正式成绩。
