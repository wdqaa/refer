# Multi-Agent OPD to a Single Student

This project studies whether a prompt-based multi-agent teacher can construct and
solve natural science questions with genuine cross-domain dependencies, and later
distill that reasoning into one student model.

## Current scope

Stage 00 is complete and Stage 01A is active:

1. inspect and normalize the existing science QA data;
2. define versioned prompt-based agents using `/home/dqwang/Model/Qwen3-8B`;
3. benchmark original-question domain dependency analysis on a 30-item gold set;
4. only after that gate, construct and audit a small transformed question pool;
5. if quality passes, compare direct, fixed multi-agent, and routed baselines.

No model training is part of the current scope.

Stage 01 draws source questions from physics, chemistry, biology, astronomy, and
geography. The answering system exposes those five experts plus a zero-shot
mathematics prompt expert; the current math source dataset is not used for training.

The current curated QA seed contains 17 precision-reviewed records at
`datasets/processed/deepseek_qa_pilot_v1/verified_qa_curated.jsonl`. No model
training or agent-trajectory collection has started.

## Current runtime

Qwen3 inference uses the project-local `.venv`, which inherits the host CUDA 12.1
PyTorch and installs the versions in `requirements-stage01.txt`. The shared Conda
environments are not modified.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Layout

- `configs/`: model, agent, data, and experiment settings
- `datasets/science_dataset/mixed_dataset/`: immutable source data
- `datasets/processed/`: normalized derived data
- `datasets/question_pool/`: candidate question pools
- `src/prompts/`: versioned agent prompts
- `src/`: loading, generation, agent, and evaluation code
- `scripts/`: executable entry points
- `outputs/`: diagnostics, evaluations, and logs
- `tests/`: lightweight data-chain tests
- `docs/PROJECT.md`: current project status

See `how_to_start.md` for the full research brief and `docs/data_protocol.md` for
the canonical record format.
