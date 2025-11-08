# Multi-Turn Constraint Environment

This environment wraps a multi-turn RL task where every assistant turn must satisfy
per-turn constraint checks. The implementation follows the guidelines in
[`docs/source/environments.md`](../../docs/source/environments.md) and mirrors the
multi-turn rollout loop provided by `verifiers`.

## Dataset Schema

Each dataset example is expected to follow this schema:

```json
{
  "prompt": [
    {"role": "user", "content": "initial prompt"},
    {"role": "user", "content": "follow up prompt"}
  ],
  "constraint_id": [
    ["constraint_turn1_id1"],
    ["constraint_turn2_id1", "constraint_turn2_id2"]
  ],
  "constraint_status": [
    ["enabled"],
    ["disabled", "enabled"]
  ]
}
```

The loader splits the first user message into the initial prompt and keeps the
remaining messages as follow-up turns. Constraint identifiers and status flags
are carried through the `info` field so reward functions and verifiers can access
them during scoring.

## Verifier Pool

`load_environment` requires a `verifier_pool` mapping constraint identifiers to
callables. Each callable receives the generated response, the running
conversation history, and metadata describing the current turn. The helper
`placeholder_verifier` illustrates the signature; replace it with task-specific
logic.

A rollout receives reward `1.0` when **all** enabled verifiers pass for every
turn and `0.0` otherwise.

## Usage

```python
import verifiers as vf
from multi_turn_constraints import load_environment, placeholder_verifier

verifier_pool = {
    "constraint_turn1_id1": placeholder_verifier,  # replace with real function
}

env = load_environment(
    dataset_name="your/dataset",
    dataset_split="train",
    verifier_pool=verifier_pool,
)
```

The environment works with any chat-capable model and tokenizer sourced from
Hugging Face, allowing multi-sample rollouts per dataset example.
