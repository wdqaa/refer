# Project Instructions

## 1. Project Goal

本项目研究如何将分学科科学 QA 数据转化为跨学科推理能力，并最终将昂贵的 Multi-Agent Teacher System 所获得的跨学科知识与推理能力蒸馏到一个单一 Student LLM 中。

最终目标分为两个阶段：

### Stage A — Adaptive Multi-Agent RL Teacher

训练一个能够根据问题本身，自主决定以下行为的 Multi-Agent Teacher System：

- 是否需要跨学科推理；
- 应该调用哪些学科专家；
- 是否需要继续调用其他专家；
- 是否需要 Retrieval；
- 是否需要 Critic；
- 什么时候继续推理；
- 什么时候停止并输出最终答案。

概念结构：

```text
                         ┌─ Physics Expert
                         ├─ Chemistry Expert
Question → Planner ──────├─ Mathematics Expert
                         ├─ Biology Expert
                         ├─ Astronomy Expert
                         ├─ Retrieval
                         ├─ Critic
                         └─ Finalize
                              ↑
                         Online RL
```

Planner 的候选 action 包括：

```text
call_expert(physics)
call_expert(chemistry)
call_expert(mathematics)
call_expert(biology)
call_expert(astronomy)

retrieve(domain, query)

critique(current_reasoning)

continue_reasoning()

finalize(answer)
```

最终希望通过 SFT Cold Start + Online RL / GRPO，使 Planner 学会一个 adaptive interdisciplinary reasoning policy。

---

### Stage B — Multi-Agent Reasoning Internalization

使用 Multi-Agent Teacher 作为 privileged teacher，将其获得的：

- 多学科知识；
- 跨学科推理路径；
- 知识融合方式；
- 问题分解能力；
- reasoning / critique / revise 能力；

蒸馏到单一 Student LLM。

**重要：最终 Student 不需要调用任何 Expert Agent。**

本项目不以“把 Agent 调用策略复制给 Student”为最终目标，而是希望 Student 将 Teacher 的跨学科推理能力内化到模型参数中。

最终形式：

```text
Adaptive Multi-Agent RL Teacher
            ↓
high-quality interdisciplinary reasoning
            ↓
Privileged / On-Policy Distillation
            ↓
Single Student LLM
            ↓
Question → reasoning → answer

Inference 阶段：
- 不调用 Physics Agent
- 不调用 Math Agent
- 不调用其他 Expert
- 不依赖 Multi-Agent System
```

On-Policy Distillation 后期可能采用：

```text
Student rollout
      ↓
Student visited state / reasoning prefix
      ↓
Multi-Agent Teacher with privileged context
      ↓
Teacher supervision / correction
      ↓
Student update
```

---

# 2. Important Research Principle

这个项目的重点不是简单地：

```text
Physics data
+ Chemistry data
+ Math data
+ Biology data
+ Astronomy data
→ mixed SFT
```

而是研究：

> 模型如何识别一个科学问题真正需要哪些不同学科知识，并将这些知识组织成自然、必要且正确的联合推理过程。

因此必须特别避免：

- 强行跨学科；
- 为了融合而融合；
- 把多个学科答案简单拼接；
- 在题面中直接暴露需要哪些学科；
- 生成与真实科学题风格明显不一致的 synthetic questions。

---

# 3. Current Execution Scope

## IMPORTANT

当前阶段只执行：

```text
Stage 00 — Project Bootstrap
Stage 01 — Agentic RL Question Pool
```

暂时不要：

- 训练 LoRA Experts；
- 进行 GRPO；
- 训练 Reward Model；
- 进行 OPD；
- 部署多个不同的大模型；
- 启动大规模训练任务。

当前目标是先回答：

> 这个研究问题和数据构造方式是否成立？

也就是说，先验证数据和 Multi-Agent workflow，再进入训练阶段。

---

# 4. Stage 00 — Project Bootstrap

首先检查当前 repository 状态，然后建立清晰、可持续维护的项目结构和文档。

建议结构：

```text
multi_agent_opd_single_student/
│
├── AGENTS.md
├── README.md
├── how_to_start.md
│
├── configs/
│   ├── model/
│   ├── agents/
│   ├── data/
│   └── experiments/
│
├── docs/
│   ├── PROJECT.md
│   ├── architecture.md
│   ├── agents.md
│   ├── data_protocol.md
│   ├── decisions.md
│   ├── experiments.md
│   │
│   └── stages/
│       ├── stage_00_bootstrap.md
│       ├── stage_01_question_pool.md
│       ├── stage_02_cold_start.md
│       ├── stage_03_agentic_rl.md
│       └── stage_04_opd.md
│
├── datasets/
│   ├── science_dataset/
│   ├── processed/
│   ├── question_pool/
│   └── metadata/
│
├── src/
│   ├── agents/
│   ├── data/
│   ├── generation/
│   ├── evaluation/
│   ├── prompts/
│   └── utils/
│
├── scripts/
│
├── outputs/
│   ├── question_pool/
│   ├── evaluations/
│   └── logs/
│
└── tests/
```

结构可以根据现有 repository 调整，不要为了匹配该模板而破坏已有合理结构。

---

# 5. Documentation Rules

以下文档需要持续维护。

## docs/PROJECT.md

记录整个项目的当前状态。

至少包括：

```text
Current Stage
Current Goal
Completed
In Progress
Next Steps
Known Problems
Important Paths
Latest Experiments
```

这是项目的主状态文件。

---

## docs/stages/

每个阶段单独记录。

例如：

```text
docs/stages/stage_01_question_pool.md
```

应包括：

```text
Objective
Research Questions
Current Design
Implementation
Experiments
Results
Problems
Decisions
Next Step
```

---

## docs/experiments.md

记录所有实验，不允许只保留“成功结果”。

每个实验至少记录：

```text
Experiment ID
Date
Purpose
Model
Dataset
Config
Command
Output Path
Main Results
Observations
Conclusion
```

---

## docs/decisions.md

记录关键设计选择及原因。

例如：

```text
Decision:
Stage 01 暂时不训练 Domain LoRA Expert。

Reason:
当前阶段先验证 Question Pool 与 Multi-Agent workflow 是否成立，
避免 expert specialization 与数据构造两个变量同时变化。

Possible Future Revision:
若 Prompt Expert 与 Single Model baseline 差异较小，
Stage 02 再加入 Domain LoRA Experts。
```

---

## docs/agents.md

统一定义所有 Agent：

- Planner
- Physics Expert
- Chemistry Expert
- Mathematics Expert
- Biology Expert
- Astronomy Expert
- Critic
- Question Generator
- Question Verifier

每个 Agent 至少定义：

```text
Role
Input
Output
System Prompt
Allowed Actions
Forbidden Behaviors
Expected Output Schema
```

不要把 Agent prompt 分散在多个 Python 文件中。

---

# 6. Stage 01 — Agentic RL Question Pool

## 6.1 Objective

第一阶段不是训练 RL。

第一阶段先验证：

> 是否可以从现有分学科科学 QA 中构造自然、合理、具有真实跨学科依赖的问题，并使 Prompt-based Multi-Agent System 在这些问题上表现出有意义的协作行为。

---

# 7. Stage 01 Research Questions

重点回答以下问题。

### RQ1 — Prompt Expert 是否足够？

第一阶段暂时不训练 LoRA Experts。

使用同一个基础模型，通过不同 system prompt 扮演：

```text
Physics Expert
Chemistry Expert
Mathematics Expert
Biology Expert
Astronomy Expert
```

首先验证 role prompting 是否足够产生有区分度的专家行为。

后续再考虑：

```text
Prompt Expert
vs
Domain-LoRA Expert
```

---

### RQ2 — 原始问题是否天然需要跨学科？

现有数据中大量问题可能只需要单一学科。

因此需要判断：

```text
原始问题
→ 是否已经存在自然跨学科依赖？
```

如果没有，可以进行 minimal transformation。

但改造原则不是：

```text
“请结合物理、数学和天文学回答……”
```

而应该：

> 保留原始科学场景和 benchmark 风格，只增加必要条件，使问题自然产生额外的推理需求。

---

### RQ3 — 生成问题是否具有明显 synthetic / fusion 痕迹？

需要重点检查：

- 是否直接出现学科名称；
- 是否使用“从多个学科角度分析”之类提示；
- 是否明显为了融合而人为增加无关信息；
- 是否比原始 benchmark 问题异常冗长；
- 是否形成固定模板；
- 是否和真实教材 / benchmark 风格差异明显。

---

### RQ4 — 如何判断是真融合而不是伪融合？

需要验证：

```text
某个 auxiliary discipline
是否真的对解题有必要？
```

例如：

```text
Astronomy + Physics + Mathematics
```

需要检查：

```text
without Physics → 是否仍然可以正常求解？
without Math → 是否仍然可以正常求解？
without Astronomy → 是否只是背景？
```

如果删除某个所谓 required domain 后仍完全不影响推理，则它不应被标记为必要学科。

---

### RQ5 — Multi-Agent 是否真的比 Single Model 有优势？

在没有训练的情况下先比较：

```text
Single Model Direct Answer

vs

Fixed Multi-Agent

vs

Prompt Router + Multi-Agent
```

重点观察 Medium / Hard Questions。

只有当 Multi-Agent workflow 本身表现出正向信号后，再进入 SFT / RL 阶段。

---

# 8. Base Model

Stage 01 使用：

```text
Qwen/Qwen3-8B
```
/home/dqwang/Model/Qwen3-8B

第一阶段只需要部署一份模型权重。

不同 Agent 通过不同 system prompts 实现。

例如：

```text
Qwen3-8B
│
├── Planner Prompt
├── Physics Prompt
├── Chemistry Prompt
├── Mathematics Prompt
├── Biology Prompt
├── Astronomy Prompt
└── Critic Prompt
```

第一阶段不要训练模型。

优先保证：

- prompt 可管理；
- 输出结构统一；
- rollout 可复现；
- Agent 调用过程可保存；
- 后续容易替换为 LoRA Expert。

---

# 9. Existing Dataset

当前主要数据路径：

```text
/home/dqwang/Projects/multi_agent_opd_single_student/datasets/science_dataset/mixed_dataset
```

首先进行只读分析。

不要修改原始数据。

所有加工数据放到：

```text
datasets/processed/
datasets/question_pool/
```

---

# 10. Question Pool Construction Principle

## 10.1 Scenario-first, not Discipline-first

不要首先抽：

```text
Physics + Math + Astronomy
```

然后强行生成一道问题。

优先流程：

```text
Real Scientific Question / Scenario
            ↓
Analyze required latent skills
            ↓
Determine required disciplines
            ↓
If necessary, minimally transform question
```

而不是：

```text
Sample disciplines
      ↓
Force them into one question
```

---

## 10.2 Real Question Anchoring

优先使用真实原始问题作为 anchor。

例如：

```text
Original Astronomy Question
        ↓
Preserve scientific scenario
        ↓
Add one quantitative / physical constraint
        ↓
Natural interdisciplinary problem
```

只允许 minimal transformation。

---

## 10.3 Hidden Domain Metadata

学科需求应该保存在 metadata 中，而不是暴露在题面。

正确：

```json
{
  "question": "...",
  "required_domains": [
    "astronomy",
    "physics",
    "mathematics"
  ]
}
```

模型只能看到：

```text
question
```

Planner 不可以看到：

```text
required_domains
```

---

## 10.4 Avoid Explicit Fusion Wording

原则上禁止生成：

```text
“请从物理和数学角度……”
“请结合化学、生物学……”
“Using knowledge from physics and mathematics...”
“From an interdisciplinary perspective...”
```

除非原始真实题本身就是这种表达。

跨学科性应该体现在 reasoning dependency，而不是 question wording。

---

# 11. Initial Pilot Pool

第一轮不要直接生成大规模数据。

目标：

```text
Pilot Question Pool: 1000–2000 samples
```

初步建议：

```text
25% Real Untouched Questions
50% Real-Anchored Transformed Questions
25% Fully Synthetic Questions
```

这个比例只是初始实验配置，不作为最终固定设计。

---

# 12. Difficulty Design

Question Pool 需要包含不同难度。

建议：

```text
Easy:   30%
Medium: 50%
Hard:   20%
```

粗略定义：

### Easy

```text
single-domain
1–2 reasoning steps
```

### Medium

```text
1–2 required domains
2–4 reasoning steps
```

### Hard

```text
2–3 required domains
multi-step reasoning
```

不要仅根据文本长度判断难度。

---

# 13. Initial Discipline Combinations

第一轮优先研究容易验证答案的组合：

```text
Physics + Mathematics

Astronomy + Physics

Astronomy + Physics + Mathematics

Chemistry + Mathematics

Physics + Chemistry
```

Biology 可以保留在原始数据和 Prompt Expert 中，但第一轮不要求大量构造 Biology-based open-ended RL questions。

原因：

开放式 Biology / Chemistry reasoning 的自动 reward 更困难。

---

# 14. Question Pool Record Schema

为每条问题保存结构化 metadata。

建议 JSONL：

```json
{
  "id": "sample_xxx",

  "source": {
    "type": "real | transformed | synthetic",
    "dataset": "...",
    "original_id": "...",
    "original_question": "..."
  },

  "question": "...",

  "answer": "...",

  "metadata": {
    "primary_domain": "astronomy",
    "required_domains": [
      "astronomy",
      "physics",
      "mathematics"
    ],
    "difficulty": "medium",
    "reasoning_type": [
      "conceptual",
      "quantitative"
    ]
  },

  "generation": {
    "generator_model": "...",
    "prompt_version": "...",
    "timestamp": "..."
  },

  "verification": {
    "scientific_validity": null,
    "naturalness": null,
    "fusion_necessity": null,
    "answer_correctness": null,
    "synthetic_smell": null
  }
}
```

Schema 可以调整，但修改后必须更新：

```text
docs/data_protocol.md
```

---

# 15. Question Pool Quality Evaluation

Stage 01 至少实现以下检查。

## 15.1 Explicit Fusion Leakage

检测：

```text
interdisciplinary
from the perspective of
using physics and mathematics
combine knowledge from
结合……学科
从……角度
```

输出统计比例。

---

## 15.2 Length Distribution

比较：

```text
Real Questions
vs
Transformed Questions
vs
Synthetic Questions
```

统计：

```text
mean
median
P25
P75
P90
```

避免 synthetic question 明显异常长。

---

## 15.3 Domain / Type Distribution

检查不同数据源的：

```text
discipline
difficulty
question type
reasoning type
```

分布。

---

## 15.4 Fusion Necessity

使用 verifier 判断：

> 每个 required domain 是否真正有必要。

必要时使用 domain ablation：

```text
full reasoning
vs
remove one domain
```

---

## 15.5 Naturalness

使用 Judge + 人工抽查。

至少随机抽查：

```text
50–100 samples
```

关注：

```text
Scientific validity
Naturalness
Difficulty
Answer correctness
Fusion necessity
Synthetic smell
```

---

## 15.6 Real-vs-Synthetic Separability

后续可以实现一个简单 diagnostic：

```text
Question → Real / Synthetic classifier
```

目的不是追求 classifier 本身，而是判断生成问题和真实科学题之间是否存在明显 distribution shift。

---

# 16. Initial Agent Definitions

## Planner

职责：

```text
读取 Question
↓
判断是否需要外部 Expert
↓
选择下一步 action
```

Planner 不能看到：

```text
required_domains
ground-truth routing
```

---

## Discipline Expert

职责：

只从自己的专业领域提供：

```text
relevant knowledge
equations
constraints
reasoning
```

禁止：

```text
强行参与与本学科无关的问题
重复其他 Agent 内容
为了显示专业性添加无关知识
```

---

## Critic

职责：

检查：

```text
Scientific correctness
Reasoning consistency
Missing constraints
Incorrect expert usage
Unnecessary domain fusion
```

Critic 不应该只是重写答案。

---

## Question Generator

职责：

基于真实 Question 进行最小必要改造。

优先：

```text
real-question anchoring
```

而不是完全自由生成。

---

## Question Verifier

职责：

判断：

```text
question validity
answer correctness
naturalness
fusion necessity
difficulty
synthetic smell
```

---

# 17. Prompt Management

所有 Prompt 应统一存放，例如：

```text
src/prompts/
```

或者：

```text
configs/agents/
```

不要把长 system prompt 硬编码在不同 Python 文件中。

Prompt 应有版本号，例如：

```text
physics_expert_v1
planner_v1
question_generator_v2
```

Question Pool metadata 中保存 prompt version。

---

# 18. Reproducibility

所有生成任务至少记录：

```text
model
model revision if available
prompt version
temperature
top_p
max_tokens
seed if applicable
input sample id
output path
```

禁止只保存最终文本而无法恢复生成配置。

---

# 19. Python Script Requirements

每个 Python 文件必须有清晰的模块级说明。

可执行脚本必须在文件开头或结尾明确说明：

```text
Purpose
Input
Output
Example Command
```

例如：

```python
"""
Purpose:
    Analyze the raw science dataset and generate statistics.

Input:
    datasets/science_dataset/mixed_dataset/

Output:
    outputs/data_analysis/summary.json

Example:
    python scripts/analyze_dataset.py \
        --input datasets/science_dataset/mixed_dataset \
        --output outputs/data_analysis
"""
```

所有脚本应尽量：

- 使用 argparse 或配置文件；
- 避免硬编码绝对路径；
- 有明确 input/output；
- 输出日志；
- 固定随机种子；
- 对关键输入做 validation；
- 出错时提供清晰错误信息。

---

# 20. Testing Requirements

至少为以下内容建立轻量测试：

```text
dataset loading
schema validation
prompt loading
agent output parsing
question pool serialization
metadata preservation
```

第一阶段不追求高覆盖率，但核心数据链不能完全没有 tests。

---

# 21. First Tasks for Codex

请严格按照以下顺序开展初步工作。

## Task 1 — Inspect Existing Repository

首先查看：

```text
project tree
existing source files
existing datasets
existing documentation
```

不要立即大规模重构。

输出当前 repository 状态总结。

---

## Task 2 — Create / Update Documentation

建立并填写：

```text
AGENTS.md
README.md

docs/PROJECT.md
docs/architecture.md
docs/agents.md
docs/data_protocol.md
docs/decisions.md
docs/experiments.md

docs/stages/stage_00_bootstrap.md
docs/stages/stage_01_question_pool.md
```

后续阶段文档可以先建立 skeleton：

```text
stage_02_cold_start.md
stage_03_agentic_rl.md
stage_04_opd.md
```

---

## Task 3 — Analyze Existing Dataset

分析：

```text
/home/dqwang/Projects/multi_agent_opd_single_student/datasets/science_dataset/mixed_dataset
```

输出：

```text
sample counts
domain distribution
question length
answer length
question types if possible
difficulty proxy if possible
data schema
example samples
```

不要修改原始数据。

---

## Task 4 — Define Prompt-based Agents

先实现：

```text
Planner
Physics Expert
Chemistry Expert
Mathematics Expert
Biology Expert
Astronomy Expert
Critic
Question Generator
Question Verifier
```

全部使用：

```text
Qwen/Qwen3-8B
```

通过不同 system prompts 区分角色。

此阶段不训练 LoRA。

---

## Task 5 — Design Question Pool Generation Pipeline

先实现一个小规模 pipeline：

```text
raw question
      ↓
question analysis
      ↓
keep / transform / generate
      ↓
answer generation
      ↓
verification
      ↓
JSONL output
```

先测试：

```text
20–50 samples
```

确认格式正确之后，再扩大到：

```text
100–200 samples
```

不要直接生成 2000 条。

---

## Task 6 — Build Initial Diagnostics

至少输出：

```text
real vs transformed length distribution
discipline combination distribution
difficulty distribution
explicit fusion leakage rate
verification score summary
```

随机生成一个：

```text
human_review_samples.jsonl
```

供人工检查。

---

## Task 7 — Run Initial Multi-Agent Baseline

仅在 Question Pool 初步验证通过后实现：

```text
Single Model Direct Answer
Fixed Multi-Agent
Prompt Router + Multi-Agent
```

比较：

```text
answer correctness
number of agent calls
token usage
qualitative reasoning quality
```

暂时不进行 RL。

---

# 22. Stage 01 Exit Criteria

只有满足以下条件，才进入 Cold-Start SFT / Agentic RL：

1. Question Pool 中大部分 transformed questions 看起来像正常科学题；
2. 明显的 explicit fusion wording 很少；
3. required_domains 与实际 reasoning dependency 基本一致；
4. Question / Answer scientific validity 可以接受；
5. Multi-Agent 在 Medium / Hard subset 上至少表现出值得继续研究的正向信号；
6. 当前 pipeline 可复现；
7. 数据、Prompt、Agent 和实验均已有文档记录。

如果这些条件不满足：

> 优先修改 Question Pool / Prompt / Agent workflow，而不是直接使用 RL 强行优化。

---

# 23. Future Stages — Do Not Implement Yet

## Stage 02 — Cold Start Agent SFT

计划：

```text
Multi-Agent Teacher
↓
high-quality state-action-observation trajectories
↓
Agent SFT
```

得到初始 Planner / Agent Policy。

---

## Stage 03 — Agentic Online RL

计划采用 GRPO / Online RL。

候选 reward：

```text
R =
R_answer
+ λ1 R_fusion
+ λ2 R_process
+ λ3 R_evidence
- λ4 C_tools
- λ5 P_irrelevant
```

但第一版应优先从简单 reward 开始：

```text
R = R_answer - λ * tool_cost
```

先验证模型是否能够学习：

```text
easy problem → fewer calls
hard problem → more useful calls
```

---

## Stage 04 — Privileged / On-Policy Distillation

Teacher：

```text
RL-optimized Adaptive Multi-Agent System
```

Student：

```text
Single LLM
```

Student 最终输入只有：

```text
Question
```

Student 最终输出：

```text
Reasoning
+
Answer
```

最终 inference 不允许依赖专家 Agent。

---

# 24. Working Principle for Codex

在整个项目中遵守以下原则：

### 1. Small experiment first

始终：

```text
20 samples
→ 100 samples
→ 1000 samples
→ large-scale
```

不要直接开始大规模任务。

### 2. One variable at a time

例如：

先：

```text
Prompt Expert
```

再比较：

```text
LoRA Expert
```

不要同时更换：

```text
model
dataset
prompt
training method
reward
```

否则无法解释结果。

### 3. Document before large changes

进行重大架构调整前：

先更新：

```text
docs/decisions.md
```

说明：

```text
What changes
Why
Expected effect
Possible risk
```

### 4. Preserve failed experiments

失败实验也记录在：

```text
docs/experiments.md
```

不要删除。

### 5. Do not over-engineer early

Stage 01 的目标是验证 research hypothesis。

优先：

```text
simple
clear
reproducible
```

而不是：

```text
complex framework
large abstraction hierarchy
premature optimization
```

---
### 6.不要浪费太多资源在数据阅读处理上
比如对于数据集进行统计，你没必要自己一条条读完，可以每个文件都选取部分，进行分析，设计一些py，或者调用模型进行数据处理

### 7.一步步来
由于当前项目其实很繁杂，涉及复杂的一步步流程，以及大量数据，所以我对于项目的掌握也不是很轻松，所以你可以小阶段小阶段地向我汇报，我也好一步步地掌握进度，效果，实现方法。独立的模块完成或是阶段性完成，都可以向我汇报

### 8.不要繁琐地记录
那些md文件，如果是我需要进行阅读的，不要写的非常冗长

# 25. Immediate Priority

当前最高优先级不是：

```text
GRPO
OPD
LoRA Experts
Reward Model
```

而是：

> 构建并验证一小批“看起来像正常科学问题，但求解时确实存在隐式跨学科依赖”的 Question Pool。

如果 Question Pool 本身不成立，后续 Agentic RL 和 Distillation 都没有可靠研究基础。

因此请首先完成：

```text
Repository bootstrap
        ↓
Dataset analysis
        ↓
Prompt-based Agent definitions
        ↓
Small Question Pool prototype
        ↓
Quality diagnostics
        ↓
Initial Multi-Agent baseline
```

完成后更新：

```text
docs/PROJECT.md
docs/stages/stage_01_question_pool.md
docs/experiments.md
docs/decisions.md
```

并根据 Stage 01 的结果再决定是否进入下一阶段。