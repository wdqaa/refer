# 新环境与新 AI 接管入口

> 新的 AI 请先完整阅读本文件，不要立即启动模型推理、API 调用或训练。

## 1. 仓库与恢复方式

GitHub 仓库：

`https://github.com/wdqaa/refer`

仓库包含代码、文档、processed 数据、原始 science dataset 和实验输出。原始数据通过 Git LFS 管理。

新环境请执行：

```bash
git lfs install
git clone git@github.com:wdqaa/refer.git
cd refer
git lfs pull
```

如果仓库已经克隆，只需更新：

```bash
cd refer
git pull --ff-only
git lfs pull
```

然后检查：

```bash
git status
git log -1 --oneline
git lfs fsck
git lfs ls-files | wc -l
```

预期：

- 工作分支为 `main`；
- Git LFS 文件为 20 个；
- 工作区干净；
- 本文件和 `docs/HANDOFF_2026-10-14.md` 均存在。

## 2. 开始前必须完整阅读

请按顺序阅读：

1. `AGENTS.md`
2. `how_to_start.md`
3. `START_HERE_NEW_AI.md`
4. `docs/HANDOFF_2026-10-14.md`
5. `docs/PROJECT.md`
6. `docs/decisions.md`
7. `docs/experiments.md`
8. `docs/stages/stage_01_domain_model_benchmark.md`

重要说明：

- `docs/PROJECT.md` 目前只正式记录到 `EXP-037`。
- astronomy expansion50 的最新状态、人工审计笔记和准确停点记录在 `docs/HANDOFF_2026-10-14.md`。
- 涉及最新进展时，以 handoff 文档和实际输出文件为准。

## 3. 项目目标

本项目研究科学 QA 中的 Multi-Agent 专家协作，并希望最终将经过验证的跨学科推理能力蒸馏到单一 Student 模型。

当前仍处于 Stage 01/01.5 的数据与架构可行性验证阶段。现阶段没有授权开始 Planner cold-start、Expert SFT、RL、GRPO、Reward Model 或 OPD。

## 4. 当前已经完成的工作

已经完成：

- 原始 49 条 verified QA 的 Direct、Routed、Oracle 三臂 baseline；
- baseline v2 回答完整性实验；
- native eval50 和 audited eval48；
- 32 道定性题 rubric audit；
- Expert-Critique 与 matched-compute Self-Refine；
- astronomy eval24 的 Qwen3-8B 与 AstroSage-8B 对比；
- astronomy expansion50 的候选筛选、Gold enrichment 和人工 Gold 审计；
- expansion50 上 Qwen3-8B 与 AstroSage-8B 的全部推理；
- expansion50 中 25 道非 MCQ 的 DeepSeek blinded first-pass judging。

当前不需要重新调用 Qwen、AstroSage 或 DeepSeek。

## 5. 当前唯一任务

继续完成 astronomy expansion50 的离线人工 adjudication。

### 5.1 阅读现有判断

DeepSeek blinded first-pass：

`outputs/domain_model_benchmark_v2/astronomy_expansion50/judgment_v1/blind_judge_traces.jsonl`

展开后的 model-item review rows：

`outputs/domain_model_benchmark_v2/astronomy_expansion50/judgment_v1/review_packet.jsonl`

人工审计工作笔记：

`docs/HANDOFF_2026-10-14.md` 第 7 节。

### 5.2 创建 versioned adjudication

参考：

`configs/evaluation/astronomy_eval24_domain_model_adjudications_v1.json`

创建：

`configs/evaluation/astronomy_expansion50_domain_model_adjudications_v1.json`

只记录需要覆盖 DeepSeek first-pass 的 criterion coverage、scientific errors 或 rubric status。不得重新定义 frozen rubric，也不得根据希望得到的模型排名调整标准。

### 5.3 生成审计后 judgments

```bash
/home/dqwang/.venv/bin/python scripts/audit_astronomy_domain_judgments.py \
  --dataset datasets/processed/domain_model_benchmark_v2/astronomy_eval60_candidates/gold_v1/astronomy_eval60_expansion_gold_curated.jsonl \
  --traces outputs/domain_model_benchmark_v2/astronomy_expansion50/judgment_v1/blind_judge_traces.jsonl \
  --adjudications configs/evaluation/astronomy_expansion50_domain_model_adjudications_v1.json \
  --output outputs/domain_model_benchmark_v2/astronomy_expansion50/evaluation_v1/audited_judgments.jsonl
```

如果新机器没有 `/home/dqwang/.venv/bin/python`，请使用新环境中安装好项目依赖的 Python，不要机械创建旧绝对路径。

### 5.4 最终离线计分

```bash
/home/dqwang/.venv/bin/python scripts/score_astronomy_domain_models.py \
  --dataset datasets/processed/domain_model_benchmark_v2/astronomy_eval60_candidates/gold_v1/astronomy_eval60_expansion_gold_curated.jsonl \
  --trace qwen3_8b=outputs/domain_model_benchmark_v2/astronomy_expansion50/qwen3_8b/traces_canonical_single.jsonl \
  --trace astrosage_8b=outputs/domain_model_benchmark_v2/astronomy_expansion50/astrosage_8b/traces.jsonl \
  --judgments outputs/domain_model_benchmark_v2/astronomy_expansion50/evaluation_v1/audited_judgments.jsonl \
  --output-dir outputs/domain_model_benchmark_v2/astronomy_expansion50/evaluation_v1 \
  --expected-items 50
```

不要在最终结果中使用 `--allow-unaudited`。

### 5.5 归档结果

最终计分完成后更新：

- `docs/PROJECT.md`
- `docs/experiments.md`
- `docs/decisions.md`，仅在形成新 go/no-go 决策时更新
- `docs/stages/stage_01_domain_model_benchmark.md`

先报告 expansion50 独立结果。eval24 与 expansion50 的 pooled 74 只能作为次要汇总。

## 6. 必须使用的 Qwen trace

Qwen 的原始 append-only trace 有 53 行，包含并发竞态造成的重复记录：

`outputs/domain_model_benchmark_v2/astronomy_expansion50/qwen3_8b/traces.jsonl`

严禁直接使用它计分。唯一合法的 Qwen 计分输入是：

`outputs/domain_model_benchmark_v2/astronomy_expansion50/qwen3_8b/traces_canonical_single.jsonl`

## 7. 不得执行的操作

不要：

- 重新运行 Qwen、AstroSage 或 DeepSeek；
- 修改 frozen QA、Gold 或 rubric；
- 修改现有 prompt；
- 重新运行 baseline；
- 继续 synthetic cross-domain generation；
- 把 evaluation-only 数据用于训练、few-shot 或 prompt tuning；
- 开始 Planner cold-start；
- 开始 Expert SFT；
- 开始 RL、GRPO、Reward Model 或 OPD；
- 删除失败 trace、重复 trace 或 rejected Gold；
- 覆盖已有 raw output。

## 8. 环境参考

原项目模型推理环境：

```text
Python       3.10.19
torch        2.4.0
transformers 4.57.6
tokenizers   0.22.2
pydantic     2.9.2
PyYAML       6.0.3
```

原机器模型路径：

- `/home/dqwang/Model/Qwen3-8B`
- `/home/dqwang/Model/AstroSage-8B`

当前任务完全可以离线完成。新机器如果没有这两个模型，不要为当前 adjudication 任务下载模型。

DeepSeek API key 不在仓库内。不得把任何 API key 写入代码、配置、日志或文档。

## 9. 接管后的第一次回复

阅读和检查完成后，请先向用户报告：

1. 是否成功拉取全部 Git LFS 数据；
2. 你理解的项目当前状态；
3. expansion50 当前停在哪一步；
4. 你准备如何完成离线 adjudication；
5. 是否发现文件缺失、checksum 不一致或状态冲突。

报告完成后再继续当前唯一任务。不要自动进入训练或新实验。
