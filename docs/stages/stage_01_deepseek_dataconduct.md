# 当前任务：重新设计 Stage 01 — DeepSeek 科学训练数据构造协议

## 一、重要背景

当前项目最终目标为：

```text
Stage 1:
构建 Adaptive Multi-Agent Teacher
→ Cold-start
→ Agentic RL / GRPO

Stage 2:
使用 OPD / privileged distillation
将 Multi-Agent Teacher 的跨学科知识和融合推理能力
内化到 Single Student LLM

最终 Student inference 不依赖任何专家 Agent。
```

目前只处理 Stage 1 的第一步：

> **训练数据构造。**

当前已决定使用 DeepSeek API 作为外部数据生产模型和裁判模型。

暂时不要训练 Qwen，不要实现 GRPO，不要实现 OPD。

## Stage 01 数据领域与专家领域

本阶段只从五个领域抽取源训练题：

```text
physics
chemistry
biology
astronomy
geography
```

不使用当前 `math/math_train.jsonl`。现有数学题以基础算术和简单代数为主，既不能代表真正需要独立数学专家的建模能力，也几乎不能提供跨学科训练价值。

但 Planner 可以调用六个专家：

```text
physics / chemistry / biology / astronomy / geography / mathematics
```

数学专家第一轮采用同一基础模型加数学角色 prompt，属于 zero-shot prompt expert，不使用当前数学数据训练。普通计算由相应科学专家直接完成；如果原题确实依赖高等数学、统计推断、数值方法或优化等独立数学方法，则将 mathematics 记为 required domain，并保留题目。不能仅因为需要数学而拒绝一条有效题目。

本轮不主动把其他科学题改写成“科学 + 数学”题，以免重新引入简单计算伪跨域；但原生数学依赖可以保留，并允许后续 Planner 调用数学专家。

地理专家的边界限定为：空间分布、自然地理、气候与地形水文、GIS/遥感、区域尺度的人地系统。数据目录只表示来源，不是领域金标；地理目录内实际属于物理或化学的问题仍由 Source Analyzer 重新判断。

---

# 二、请纠正之前的一个设计偏差

之前方案提出：

> 第一版只生成选择题，因为选择题便于 RL 精确奖励。

这个设计过于狭窄。

现有数据实际包含：

```text
1. 选择题 MCQ

2. 客观开放题
   - numeric
   - short answer
   - formula / calculation
   - 有明确 Gold

3. 定性开放题
   - explanation
   - causal analysis
   - mechanism analysis
   - biology 中较多
   - 不一定存在唯一短答案
```

同时每一种题型又可能是：

```text
single-domain
或
native cross-domain
```

因此首先应设计能够覆盖整个数据分布的数据协议，而不是为了 RL 方便提前丢弃开放题。

---

# 三、首先分析现有数据

请先只读分析：

```text
/home/dqwang/Projects/multi_agent_opd_single_student/datasets/science_dataset/mixed_dataset
```

不要修改任何原始文件。

需要输出真实统计：

```text
各学科样本数

MCQ 比例

objective open-ended 比例

qualitative open-ended 比例

已有 Gold answer 比例

无明确 Gold / 长文本 reference answer 比例

可能天然 cross-domain 的比例

question / answer length

典型题型示例
```

不要完全依赖目录学科标签作为真实 domain。

已有实验已经发现：

```text
source folder != true required domain
```
这里如何进行分析，可以考虑调用deepseek api
---

# 四、建立统一 Source Taxonomy

将源题首先划分为：

```text
A1 = single-domain MCQ
A2 = native cross-domain MCQ

B1 = single-domain objective open-ended
B2 = native cross-domain objective open-ended

C1 = single-domain qualitative open-ended
C2 = native cross-domain qualitative open-ended
```

其中：

## objective open-ended

指：

```text
numeric
short answer
symbolic
formula
明确唯一事实答案
```

## qualitative open-ended

指：

```text
解释
机制分析
因果分析
开放式科学论述
```

不要求存在唯一字符串答案。

---

# 五、训练数据必须服务三个不同目标

不要构造一个数据文件解决所有问题。

最终 Verified Dataset 应可以导出至少三种训练视图：

```text
1. Reasoning / Answer SFT

2. Planner Cold-start SFT

3. RL Question Pool
```

---

# 六、不同数据类型的处理策略

## A1 — Single-domain MCQ

不要全部 transform。

一部分：

```text
KEEP
```

作为：

```text
single-domain control
```

用于未来训练 Planner：

> 不是什么题都需要多个专家。

对于保留题：

DeepSeek 仍需要 blind solve，并生成：

```text
Gold correct choice
Gold answer
Standard reasoning
```

另一部分只有满足自然跨域条件时：

```text
TRANSFORM
```

---

## A2 — Native Cross-domain MCQ

优先：

```text
KEEP
```

不要为了“生成数据”再次修改一个已经自然跨域的真实题。

DeepSeek 负责：

```text
blind solve
dependency verification
standard reasoning generation
answer uniqueness verification
```

这些题应该被明确标记：

```text
origin_type = native_cross_domain
```

后续单独统计。

---

## B1 — Single-domain Objective Open-ended

保留其中高质量样本作为 control。

如果场景允许自然增加新的 knowledge dependency，则允许 transform。

不要强制转换成选择题。

保留原来的自然回答形式。

Gold 应根据 answer type 保存。

例如 numeric：

{
  "type": "numeric",
  "value": ...,
  "unit": "...",
  "tolerance": ...
}

short answer：

{
  "type": "short_answer",
  "canonical": "...",
  "aliases": [...]
}

symbolic：

{
  "type": "symbolic",
  "canonical_expression": "...",
  "equivalence_rule": "..."
}

这种 objective open-ended 数据同样适合第一版 RL。

---

## B2 — Native Cross-domain Objective Open-ended

这是高价值 RL 数据。

优先 KEEP。

DeepSeek 负责：

```text
independent solve
standard derivation
answer normalization
dependency verification
```

---

## C1 — Single-domain Qualitative Open-ended

不要删除。

主要用于：

```text
Reasoning SFT
Cold-start
未来 Reward Model / Judge
```

第一版基于精确 reward 的 RL pilot 可以少量使用或暂不使用。

DeepSeek 需要生成：

```text
reference_answer

rubric:
    required_points
    optional_points
    fatal_errors
```

而不是强制生成一个所谓“唯一答案”。

---

## C2 — Native Cross-domain Qualitative Open-ended

这是研究跨学科融合推理的重要数据。

优先 KEEP。

用于：

```text
Cross-domain reasoning SFT

未来 Process Reward

未来 Reward Model / LLM Judge
```

不要因为第一版 RL reward 不容易计算而删除。

---

# 七、单学科题 Keep / Transform / Reject Protocol

## KEEP

如果：

```text
科学有效
题型自然
benchmark 风格合理
有代表性
有训练价值
或适合作为 single-domain routing control
```

即使完全没有跨学科，也可以 KEEP。

---

## TRANSFORM

只有当：

```text
1. 原科学场景天然支持一个新的知识依赖；

2. 新知识依赖真正改变 solution path；

3. 原 domain 仍然必要；

4. 新 domain 也不可缺少；

5. 两个 domain 必须进入同一条 reasoning chain；

6. 不是两个独立问题的拼接；

7. 题面看起来仍然像正常科学题；

8. 题干不能直接提示应该调用哪些学科。
```

才允许 transform。

---

## REJECT

例如：

```text
scientifically invalid

underspecified

ambiguous

Gold unreliable

无法自然扩展

transform 后会明显人工拼接

数据本身质量过低且也没有 control 价值
```

---

# 八、跨学科融合必须基于“关系”，而不是简单学科组合

不要：

```text
随机选 Biology + Physics
→ 强制融合
```

请定义 `fusion_relation`。

第一版至少支持：

```text
mechanism_explanation

quantitative_modeling

constraint_coupling

measurement_observation
```

---

## mechanism_explanation

一个 domain 为另一个 domain 的现象提供机制。

例如：

```text
physics → chemistry

chemistry → biology

physics → astronomy
```

---

## quantitative_modeling

数学专家提供独立的建模或推理方法，使科学问题能够被求解。

注意：

```text
ordinary arithmetic
basic algebra
formula substitution
unit conversion
```

不能形成独立 required domain；这些计算属于科学专家应具备的基本定量能力。

如果题目必须使用微分方程、统计推断、优化、数值分析或其他独立数学方法，则 mathematics 可以成为 required domain。该标签本身不构成拒绝理由；题目只有在缺少数据、条件不足或答案不可靠时才拒绝。

---

## constraint_coupling

两个领域分别提供不可缺少的条件。

例如：

```text
chemical reaction rate
+
protein denaturation
→ enzyme activity
```

---

## measurement_observation

一个领域提供 observation / measurement，
另一个领域提供解释机制。

例如：

```text
astronomical observation
+
physical law
```

---

# 九、建立 Natural Domain Relation Prior

请建立配置文件，例如：

```text
configs/data/domain_relation_prior.yaml
```

用于指导 DeepSeek 的 transformation proposal。

例如：

physics:
  chemistry:
    - mechanism_explanation
    - constraint_coupling
  biology:
    - mechanism_explanation
  geography:
    - mechanism_explanation
    - measurement_observation

astronomy:
  physics:
    - mechanism_explanation
    - measurement_observation

chemistry:
  physics:
    - mechanism_explanation
  biology:
    - mechanism_explanation
    - constraint_coupling
  geography:
    - mechanism_explanation
    - constraint_coupling

biology:
  chemistry:
    - mechanism_explanation
    - constraint_coupling
  physics:
    - mechanism_explanation
  geography:
    - mechanism_explanation
    - constraint_coupling

geography:
  physics:
    - mechanism_explanation
    - measurement_observation
  chemistry:
    - mechanism_explanation
    - constraint_coupling
  biology:
    - mechanism_explanation
    - constraint_coupling
```

这只是 prior。

不能强制 Transformer 一定成功。

DeepSeek 必须可以：

```text
reject_transform
```

数学可以出现在原题的 required domains 和 Planner 调用空间中，但当前 relation prior 不把数学作为新增 transformation domain。

---

# 十、DeepSeek 数据生产 Pipeline

建议重新设计成：

```text
Raw Source
    ↓
Source Analyzer
    ↓
Question Type Classification
    ↓
KEEP / TRANSFORM / REJECT

KEEP
    ↓
Blind Solver

TRANSFORM
    ↓
Relation Planner
    ↓
Question Transformer
    ↓
Blind Solver

然后统一：
    ↓
Answer Normalizer / Rubric Builder
    ↓
Scientific Judge
    ↓
Dependency Judge
    ↓
Deterministic Rules
    ↓
Verified Record
```

---

# 十一、Call 1 — Source Analyzer

DeepSeek 输入：

```text
question
choices if available
source reference answer if necessary only for validity analysis
```

但不能直接相信：

```text
source folder domain
```

Analyzer 输出：

```json
{
  "scientifically_valid": true,

  "question_type":
    "mcq | numeric | short_answer | symbolic | qualitative",

  "primary_domain": "biology",

  "required_domains": ["biology"],

  "helpful_domains": ["chemistry"],

  "mathematics_level": "none | ordinary | independent",

  "already_cross_domain": false,

  "answerability": "objective | qualitative",

  "decision":
    "keep_control | keep_cross_domain | transform_candidate | reject",

  "natural_extension_possible": true,

  "confidence": 0.92
}
```

---

# 十二、Call 2 — Relation Planner

只对：

```text
transform_candidate
```

调用。

它不直接写新题。

只负责判断：

```text
新增哪个 domain 最自然？

为什么？

属于哪种 fusion relation？

新增知识具体参与哪一步求解？

如果找不到自然关系，reject_transform。
```

输出例如：

```json
{
  "decision": "propose | reject_transform",

  "new_domain": "chemistry",

  "fusion_relation": "constraint_coupling",

  "scientific_bridge": "...",

  "new_dependency": "...",

  "why_natural": "...",

  "expected_solution_change": "..."
}
```

---

# 十三、Call 3 — Question Transformer

输入：

```text
原题
已验证 Source Analysis
Relation Plan
```

要求：

```text
尽量保留原 scientific scenario

只增加必要条件

不要直接写学科名称

不要出现：
interdisciplinary
from a physics perspective
combine biology and chemistry
等显式 routing hints

不要拼接两个独立子问题

不要简单把原答案后面增加一个解释段落

必须改变 solution path
```

无法自然完成：

```text
reject_transform
```

---

# 十四、Call 4 — Blind Solver

Blind Solver 只能看到最终题。

不能看到：

```text
source question

source answer

source domains

relation plan

generator answer

required domains

transformation summary
```

根据题型产生对应 Gold。

---

# 十五、Gold Answer Protocol

## MCQ

保存：

```text
correct_choice
answer_text
standard_solution
```

即使原始题只有 choice，也应生成标准推理。

---

## Numeric

保存：

```text
value
unit
tolerance
standard_solution
```

---

## Short Answer

保存：

```text
canonical_answer
accepted_aliases
standard_solution
```

---

## Symbolic

保存：

```text
canonical_expression
equivalence_rule
standard_derivation
```

---

## Qualitative

保存：

```text
reference_answer

required_points

optional_points

fatal_errors

standard_reasoning
```

不要伪造“唯一字符串 Gold”。

---

# 十六、Call 5 — Scientific Judge

只评价：

```text
题是否科学有效

Gold 是否正确

条件是否充分

是否存在歧义

推导是否正确

MCQ 是否唯一答案

Numeric / symbolic 是否可验证
```

不要负责判断跨学科质量。

---

# 十七、Call 6 — Dependency Judge

独立判断：

```text
required_domains 是否真实必要

helpful_domains 是否只是可选增益而非最小必要条件

transform 的 new domain 是否真正新增

source domain 是否仍必要

new domain 是否 indispensable

是否是单一 integrated reasoning chain

是否两个独立子问题拼接

是否只是背景知识

是否显式泄露学科信息

是否只是普通数学操作
```

---

# 十八、确定性规则

至少实现：

```text
schema valid

domain labels valid

answer format valid

MCQ correct choice exists

numeric unit/tolerance valid

Generator / Blind Solver objective answer agreement

no explicit routing language

no unsupported domain labels

transformed question != source question

new domain not in source required domains

source required domains preserved

exactly one new domain during first transformation version

no repair loop
```

---

# 十九、最终 Verified Record Schema

不要只针对 MCQ。

设计通用 schema，例如：

```json
{
  "id": "...",

  "source": {
    "dataset": "...",
    "source_id": "...",
    "original_question": "...",
    "original_answer": "..."
  },

  "origin_type":
    "native_single | native_cross_domain | transformed_cross_domain",

  "question_type":
    "mcq | numeric | short_answer | symbolic | qualitative",

  "question": "...",

  "choices": null,

  "gold": {
    "answer": "...",
    "normalized_answer": {},
    "standard_solution": [],
    "rubric": null
  },

  "primary_domain": "biology",

  "required_domains": [
    "biology",
    "chemistry"
  ],

  "helpful_domains": [
    "mathematics"
  ],

  "fusion_relation": "constraint_coupling",

  "difficulty": "medium",

  "verification": {
    "scientifically_valid": true,
    "answer_correct": true,
    "dependency_verified": true,
    "integrated_reasoning": true,
    "accepted": true
  },

  "provenance": {
    "source_type": "...",
    "deepseek_model": "...",
    "prompt_versions": {}
  }
}
```

---

# 二十、训练数据导出视图

## 1. Reasoning SFT

所有高质量：

```text
MCQ
Numeric
Short Answer
Symbolic
Qualitative
```

都可以进入。

格式：

```text
Question
→ Standard Reasoning + Gold Answer
```

---

## 2. Planner Cold-start SFT

使用：

```text
Question

→ minimal domain plan
```

例如：

```text
call_expert(astronomy)
call_expert(physics)
finalize
```

注意：

required_domains 表示最小必要专家集合，只是 cold-start supervision。

helpful_domains 可以作为探索提示，但不作为强制调用轨迹。

不要假设它就是未来 RL 的最优 action trajectory。

RL 后续允许学习不同 routing。

多调用一个有帮助但非必要的专家不应直接判错，只施加较小的效率成本；答案正确性仍是主要奖励。

---

## 3. RL Pool

第一版主要：

```text
MCQ
numeric
short_answer
symbolic
```

原因：

reward 可以自动验证。

Qualitative 数据先保留，未来配合：

```text
Reward Model
Process Reward
DeepSeek Judge
```

---

# 二十一、Single-domain 数据不要全部修改

最终数据池必须明确保留：

```text
native_single
```

否则 Planner 会形成：

```text
every question
→ multiple experts
```

的错误策略。

不要先规定固定 60/20/20 比例。

先分析真实数据和 transformation yield。

然后根据：

```text
difficulty

question type

domain

native / synthetic

single / cross-domain
```

进行分层采样。

---

# 二十二、第一轮实验不是正式生产训练集

先做 protocol batch。

建议：

```text
120–180 source questions
```

而不是只选择 MCQ。

尽量覆盖：

```text
A1
A2
B1
B2
C1
C2
```

以及：

```text
physics
chemistry
biology
astronomy
geography
```

源数据仍然只有上述五域，但 dependency label 与 Planner action space 允许额外的 mathematics。

第一轮固定抽取 150 条：每个领域先随机抽取 20 条代表性样本，再加入 50 条边界/稀有题型样本（astronomy 5、biology 10、chemistry 10、physics 10、geography 15）。代表性样本用于估计真实分布，边界样本只用于暴露协议失败，不与前者混合报告总体比例。

第一轮主要回答：

```text
这套协议能不能覆盖现有数据分布？

哪些类型最容易自然 transform？

哪些类型 transformation failure 最大？

哪些 domain pair 最自然？

DeepSeek 对 qualitative Gold 是否稳定？

objective open-ended 的自动 verifier 是否可行？
```

## 当前协议结果（2026-09-27）

Source Analyzer v2.1 已完成 150 条分析。随后选取 30 条进行第一轮 QA 生产：10 条单域 control、8 条原生跨域、12 条改写候选。DeepSeek 机器链路接受 25 条，独立语义复核最终保留 17 条：9 条单域、4 条原生跨域、4 条改写跨域。

本轮最终种子数据为：

```text
datasets/processed/deepseek_qa_pilot_v1/verified_qa_curated.jsonl
```

该结果说明 Blind Solver 生成 Gold 的流程可用，但同模型 Judge 会放过部分伪融合和论文特定事实幻觉，因此暂不直接扩大规模，也不使用未经过语义复核的 25 条机器接受结果。

## Prompt v2 修订（2026-09-28）

第一轮八条语义淘汰样本表明，Judge v1 会在 `issues` 中指出实质风险却仍
返回接受。Transformer/Judge v2 因此只增加四项精度约束：逐域反事实必要性、
题面不得直接提供依赖、成熟子领域不得虚假拆域、缺少研究或法规依据时拒绝。
Judge v2 的 `issues` 仅记录阻断性问题，任何非空 `issues` 都由确定性规则拒绝。

下一步只运行冻结失败样本回放和六条新改写候选的小规模验证；通过前不生产
第二批数据，不生成 Qwen 轨迹，也不启动训练或 RL。

### Prompt v2 验证结果

- 冻结回放：八条已知坏例全部拒绝，四条边界保留例中三条保留。
- 一条半导体掺杂题被确认是材料/固体物理内部拆分，第一轮 curated seed
  由 17 条修正为 16 条。
- 新改写六条中两条通过，四条因新增领域并非必需而拒绝。
- 第二批候选已离线固定为 78 条：47 单域 control、9 原生跨域、22 改写候选；
  首轮计划 200 次调用，另预留约 10 次定向失败重试。确认前不执行。

### 第二批生产结果

- 78 条来源产生 37 条机器接受记录；语义复核后保留 31 条。
- 保留构成：18 单域、5 原生跨域、8 改写跨域。
- 与第一轮修正后的 16 条及验证阶段 2 条合并后，共 49 条 verified QA。
- 最终路径：`datasets/processed/deepseek_qa_curated_v2/verified_qa_curated_49.jsonl`。
- 不从已拒绝或预筛高风险候选中补数；后续补量应重新抽取小批源题并复用
  Analyzer v2.1、Blind Solver、Judge v2 和人工语义复核。

---

# 二十三、输出实验报告

第一轮结束后请输出：

```text
各 Source Type 数量

KEEP / TRANSFORM / REJECT 比例

每类题 transform success rate

native cross-domain 比例

新增 domain 分布

fusion_relation 分布

各 question type scientific validity

objective answer agreement rate

qualitative rubric consistency

dependency verification pass rate

synthetic smell failure

主要失败类型

代表性成功案例

代表性失败案例
```

并更新：

```text
docs/PROJECT.md

docs/stages/stage_01_question_pool.md

docs/experiments.md

docs/decisions.md
```

新增：

```text
docs/stages/stage_01_deepseek_data_protocol.md
```

---

# 二十四、当前不要做的事情

本任务结束前不要：

```text
训练 Qwen

训练 LoRA Expert

跑 GRPO

实现完整 Agent RL

做 OPD

批量生成数千条数据

把所有开放题转成 MCQ

让 DeepSeek 生产 Qwen Agent rollout trajectory
```

当前唯一任务是：

> **建立一套能够覆盖选择题、客观开放题和定性开放题，并能够区分天然单学科、天然跨学科和自然改造跨学科数据的 DeepSeek 数据生产协议。**

先用一个小型、分层的 protocol batch 验证整个数据协议。

在协议通过之前，不扩大数据生产规模。
