# Member A 开发记录

本记录覆盖 2026-11-13 的 Agent Core 开发和必要的 Runtime 集成。课程要求、历史初始化记录分别见 `AGENTS.md` 和 `AI_WORK.md`；相关概念及论文见 `KNOWLEDGE.md`。当前结果是开发版本，正式模式仍关闭。

## 实现范围

### A-03：LLM 客户端与审计

`agent/llm_client.py` 的 `RealLLMClient` 实现了可配置的 OpenAI 兼容 `chat/completions` HTTP 客户端。构造时不发请求，密钥仅从 `SUPERAGENT_API_KEY` 环境变量读取；课程平台地址和模型名称由外部配置。客户端支持注入 transport，因此测试可返回模拟响应而无需网络或付费调用。[校方 API 文档](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-usage/)确认 V2 使用 OpenAI 协议、Bearer 认证和下述端点。

一次调用依次执行：

1. 检查密钥、模型价格以及输入 Token 上界和输出上限对应的最坏成本是否在剩余额度内。
2. 发送前写 `llm_call_started`；收到有效用量后写 `llm_usage_observed` 和 `llm_call_usage`，包括输入/输出 Token、配置的标准价格及成本。
3. 通信异常、响应缺少用量、超出预检边界等情况写 `llm_call_unresolved`，阻断该客户端后续调用。未知费用不记为零，也不自动重试可能已经计费的请求。

客户端只支持顺序调用。输入 Token 上界必须由使用者依据供应商协议独立核实，涵盖平台附加内容；代码无法自行证明上界可靠。校方说明了基础协议，但具体模型的 `max_tokens`、响应 `usage` 和 `model` 值、日期价格及输入上界仍需核实。价格文件里的 HTTPS 来源仅作格式检查，**不等于代码已核验其官方真实性**。无法安全恢复未知成本的跨进程重启/多客户端共享预算，仍需后续完成。

用户提供的是账户凭据和账户额度；账户上限不能替代课程的**每任务人民币 20 元**上限。凭据未写入本仓库、测试、配置或日志，也没有在本次开发中发起真实请求。

### A-04/A-05：多轮决策与工具执行

`agent/planner.py` 的 `LLMPlanner` 将任务模态、类别、指标、训练数据摘要及已完成实验的配置、验证指标、耗时和错误反馈放入请求，让模型输出受限 JSON 动作。任务路径、数据集名称和 test 内容不会作为任务身份发送给模型。`parse_decision` 只接受 `experiment` / `stop`，并验证理由、逻辑回归参数范围、随机种子及停止条件；不合法输出会记录 `planner_rejected` 并附错误反馈有限次数重试。

`agent/controller.py` 保存每轮配置和结果，监督状态转移和实验次数。成功结果按 `TaskSpec.metric` 比较，同分保留较早结果；达到上限后自动停止。`runtime/runner.py` 执行经校验的实验动作，使用各自的模型文件，反馈失败或验证结果，并保存所选结果。工具接口是**结构化 JSON 动作 + Runner 分派**，目前没有使用供应商原生 function calling。

目前训练工具只支持 `logistic_regression` 的 `C`、`max_iter`，多轮搜索范围因此有限。`max_seconds` 只记录、不强制执行；Text/Image、更多候选算法、完整工具恢复及课程最终 test 评估仍待实现。Mock Planner 仍为默认值，保持无 LLM 调用的开发基线。

### 必要的 B/C 改动

- `runtime/runner.py` 支持注入 Planner、循环执行、每轮独立保存模型和最佳验证结果选择。`main.py` 新增显式 `--planner llm` 和 `--max-experiments`；默认仍为 Mock。
- `TaskSpec.validation_seed` 固定内部划分种子，避免不同候选实验因模型随机种子不同而使用不同验证划分；显式阻止 test 文件路径同时作为训练/验证文件。
- 正式模式仍在价格校验后主动阻断；当前路径检查只能发现相同解析路径，不能证明不同文件中没有重复或泄漏数据。

## 配置与开发使用

平台 Base URL 是 `https://aigc-api.hkust-gz.edu.cn/api/v2`，完整对话端点是 `https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions`，已写入 `.env.example` 供参考。[校方模型列表](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-model-list/)列出以下当前 API 名称及后端；该列表本身不证明这些后端在截止日前的发布状态或其历史模型官方价格。

| API 模型名及别名 | 列表中的后端 | 发布证据 |
|---|---|---|
| `Qwen3.8-27B` / `Qwen` | `Qwen3.8-27B-w8a8` | [Qwen 官方仓库](https://github.com/QwenLM/Qwen3.8)记载基础模型于 2026-08-14 发布；校方量化后端的版本和发布时间仍需核对。 |
| `DeepSeek-V4-Pro` / `DSP` | `DeepSeek-V4-Pro` | [DeepSeek 官方更新记录](https://api-docs.deepseek.com/zh-cn/updates/)列出 2026-08-13 正式版上线。 |
| `DeepSeek-V4.1-Flash` / `DSF` | `DeepSeek-V4.1-Flash-1017` | 同一官方记录列出 2026-09-10 发布及 2026-10-17 版本更新。 |

优先使用完整 API 模型名；目前预算模块要求模型名以 `Qwen` 或 `DeepSeek` 开头，`DSP` 和 `DSF` 别名会被拒绝。平台可能返回后端版本而非请求名，客户端当前会对 `model` 不一致失败关闭；其实际行为未验证。上述上下文窗口不能直接当作覆盖平台额外开销的计费输入上界。

`configs/model_prices.yaml` 因 2026-10-21 模型官方标准价格尚未核验继续为空；校方列表的当前平台价格不能直接替代课程规定的价格。填写时需要模型发布日期、`2026-10-21` 价格日期、官方来源 URL、每百万输入和输出 Token 的人民币标准价格。不要用单元测试中的虚构数值。

真实客户端的配置入口：

```text
SUPERAGENT_API_KEY                 由使用者在当前环境安全设置
SUPERAGENT_BASE_URL                https://aigc-api.hkust-gz.edu.cn/api/v2
SUPERAGENT_MODEL                   已核验、满足课程截止日的完整 API 模型名
SUPERAGENT_INPUT_TOKEN_UPPER_BOUND  已核验的单次输入 Token 上界
```

客户端会在 Base URL 后追加 `/chat/completions`，所以 Base URL 的路径需要与供应商协议匹配。**此适配器没有读取 `.env` 文件的功能**；`.env.example` 只是变量名示例，应先在实际进程环境中设置。配置并核验完成后，显式选择以下命令会向供应商发请求并可能产生费用：

```powershell
conda activate coursework
python main.py run --task configs/coupon_development.yaml --planner llm --max-experiments 3
```

这仍是 development 模式，真实自主课程评估必须等待正式运行条件满足。默认 `python main.py run --task configs/coupon_development.yaml` 使用 Mock，不调用 API。未知成本时要保留日志并与供应商核对，不能直接重启、假设零费用继续运行。

## 验证记录

在 `coursework` Python 3.11 环境限制 BLAS/OMP 线程后，运行全部 pytest：**20 passed**。其中新增模拟供应商测试覆盖调用预检、账单使用量、未知使用量阻断及密钥不写日志；集成测试覆盖两轮实验、反馈、独立模型、验证选模和 test 路径隔离。开发环境沙箱无法使用默认系统临时目录时，使用项目 `outputs/` 下的 pytest 临时目录。

校方地址确认后增加了离线端点构造测试，再次运行全部测试为 **21 passed**；同时重跑优惠券 train/val Mock 流程，验证 accuracy 为 **0.661576**。端点测试使用模拟 transport，没有向平台发送请求。

现有优惠券 train/val 的 Mock 开发运行 accuracy 为 **0.661576**，与此前基线一致；没有使用 test 进行模型选择。上述模拟测试和 Mock 分数都不是正式课程结果。真实 API、真实计费和课程平台兼容性尚未验证。

## 后续更新：真实 API 开发验证（2026-11-13）

本节是对上文历史记录的更新。本轮按用户要求显式启用未定价开发测试，未更改 `configs/model_prices.yaml`，也没有解除正式模式限制。`RealLLMClient` 的 `development_mode=True` 不需要价格表或输入 Token 上界，但必须给出有限 `max_calls`；只允许列出的校方 Qwen/DeepSeek 模型或别名，使用 `requests.Session(trust_env=False)` 直连。每次记录真实输入/输出 Token，`cost_rmb=null` 表示尚未计价；未知用量或网络故障会阻断客户端后续请求。该开发模式**不能证明**每任务 20 元预算合规。

使用用户提供的凭据，仅在进程内设置环境变量；没有把密钥写入代码、文件、日志或命令示例。先通过 `main.py llm-test` 实际调用项目的 `RealLLMClient`，再运行多轮任务：

| 运行 | 真实调用与 Token（输入 / 输出） | 结果和日志 |
|---|---|---|
| Qwen 单次 | 1 次，59 / 17 | 成功；有 `reasoning_content`，记录存在标志，不保存其内容。`outputs/llm-test-20261113T071653Z-df4cc8ad/events.jsonl` |
| DeepSeek 单次 | 1 次，37 / 3 | 成功；`outputs/llm-test-20261113T071656Z-f4592105/events.jsonl` |
| Qwen 多轮 | 4 次，3,988 / 3,856 | 两轮真实训练：`C=1`、`C=10`，accuracy 均为 0.661576。一次无效 JSON 被拒绝后由模型修正；最后回复达到输出上限，再遇调用次数上限而失败。`outputs/20261113T071658Z-7df45dc9/events.jsonl` |
| DeepSeek 多轮 | 2 次，1,811 / 696 | 根据第一轮 0.661576 的反馈自主提出 `C=10`；两轮 accuracy 相同，控制器到实验上限后按规则保留先完成的第一轮。`outputs/20261113T072045Z-8625761a/events.jsonl` |

共 8 次真实调用，输入 **5,895** Token、输出 **4,572** Token；历史官方价格未配置，费用未知。上述是开发验证而非课程成绩。Qwen 失败后修复了已验证模型存在时的调用次数上限停止路径：只对请求发出前的本地上限停止并选取最佳结果；未知用量等错误仍失败关闭。DeepSeek 的最终停止是**控制器达到实验上限**，不是模型自行提出的停止。模拟测试另覆盖模型提出停止、无效参数修正及工具失败后的恢复。

补充上限停止路径的 Runner 集成测试后，在 `coursework` 环境执行全部 pytest：**31 passed**。该回退在模拟测试中验证；Qwen 那次真实失败运行未被事后改写。

`main.py llm-test --model <模型名>` 显式发一次未定价开发调用。多轮运行必须显式指定 `--planner llm-dev --max-llm-calls <次数>` 和实验上限，设置 `SUPERAGENT_API_KEY`、`SUPERAGENT_BASE_URL`、`SUPERAGENT_MODEL`；未提供密钥时 `scripts/live_validation.py` 会跳过，不伪造结果。不要将密钥粘贴到源码或文档。`--llm-max-output-tokens` 用于限制单次输出，模型可能在达到上限时只返回部分内容。

### A 对 B/C 的最小接口

1. **任务输入**：`TaskSpec` 提供模态、分类类型、目标、指标、train/val 路径及明确划分；test 路径只能留作最终评估，不能传给模型决策作为实验数据。
2. **实验动作**：Agent 输出 JSON `action=experiment`、`reason`、`experiment={algorithm, parameters, random_seed, max_seconds}`；停止动作包含 `action=stop`、`reason`、`stop_condition`。当前校验只允许逻辑回归的 `C` 和 `max_iter`，B 扩展算法时须同步声明校验规则。
3. **工具入口**：C 通过 `train_validate(task: TaskSpec, config: ExperimentConfig, model_path: Path)` 调用 B。每次实验提供独立模型路径；预处理仅在训练数据上拟合。
4. **工具返回**：`ExperimentResult` 必须包含 `status`、`validation_metrics`（至少含任务指定指标）、`model_path`、`runtime_seconds` 和失败时的 `error`。无有效指标或模型路径的结果不可选为最优。失败信息供下一次规划修正。
5. **时间限制**：`max_seconds` 当前只是契约字段，B/C 尚未强制中断。后续必须实现可验证的超时与异常恢复，并记录真实耗时。
6. **审计和停止**：C 保存每次调用、用量状态、动作、结果、拒绝原因、停止条件和选中模型。真实费用、未知用量跨进程恢复和严格预算保护仍是正式模式阻塞项。

课程 Full Assignment Requirements 第 1 页第 3 节允许自行设计 Agent 和工具机制，要求依据实验反馈自主决定并记录；没有要求供应商原生 Function Calling。当前实现采用 JSON 动作、参数验证与 Runner 分派，不应描述为供应商原生工具调用。
