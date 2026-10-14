# Architecture

## Stage 01 data path

```text
immutable source QA
  -> normalized source record
  -> explicit source analysis (validity + existing required domains)
  -> keep / minimal transform / limited synthesis
  -> answer construction
  -> verifier and rule checks
  -> candidate JSONL + diagnostics + human-review sample
```

The planner and answering agents receive only the question and conversation state.
Evaluation-only metadata such as `required_domains` stays outside model context.

## Stage 01 agent path

```text
question -> planner -> zero or more prompted domain experts -> optional critic
         -> planner/finalizer -> answer + trace
```

All roles use one local Qwen3-8B checkpoint with different versioned system prompts.
The first baseline will remain simple: direct answer, fixed expert set, and prompt
router. Every run stores prompts, calls, outputs, usage, and configuration.

## Stage 01.5 baseline isolation

```text
Direct: question -> final solver
Routed: question -> question-only router -> selected experts -> shared synthesizer
Oracle: question -> evaluation orchestrator selects gold domains -> selected experts
        -> shared synthesizer
```

The routed and oracle paths share expert and synthesizer prompts. Gold domains select
calls only in the oracle orchestrator and are never serialized into a model message.
This comparison separates collaboration potential from routing quality.

## Boundaries

Training, online RL, domain LoRAs, and student distillation are future stages and
must not be coupled to the Stage 01 implementation.

Generator output is checked before invoking the verifier. A transform must add a
necessary allowed domain and change the solution; paraphrases are rejected. Verifier
domain-ablation entries must match `required_domains` exactly.
