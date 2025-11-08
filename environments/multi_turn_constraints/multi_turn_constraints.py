"""Multi-turn RL environment with constraint-based verification."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Sequence, TypedDict

from datasets import Dataset, DatasetDict, load_dataset

import verifiers as vf
from verifiers.types import ChatMessage, Info, Messages, State
from verifiers.utils.async_utils import maybe_await


class ConstraintTurn(TypedDict):
    """Metadata for a single turn's constraint configuration."""

    constraint_ids: Sequence[str]
    statuses: Sequence[str]


ConstraintVerifier = Callable[..., bool | Awaitable[bool]]


def _normalize_chat_message(message: Mapping[str, Any]) -> ChatMessage:
    """Ensure each chat message contains the required fields."""

    if "role" not in message or "content" not in message:
        raise ValueError(
            "Each message must contain 'role' and 'content' keys for chat formatting."
        )
    role = str(message["role"]).strip()
    content = str(message["content"])
    if not role:
        raise ValueError("Message role cannot be empty.")
    return {"role": role, "content": content}


def _prepare_example(example: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Convert a raw dataset example into the format expected by the environment."""

    prompts = example.get("prompt")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(
            "Expected the dataset to provide a non-empty list of chat prompts per example."
        )
    chat_prompts = [_normalize_chat_message(message) for message in prompts]

    constraint_ids = example.get("constraint_id", [])
    constraint_status = example.get("constraint_status", [])
    if len(constraint_ids) != len(constraint_status):
        raise ValueError(
            "`constraint_id` and `constraint_status` must have the same number of turns."
        )
    if len(chat_prompts) != len(constraint_ids):
        raise ValueError(
            "The number of chat prompts must match the number of constraint turns."
        )

    turn_constraints: list[ConstraintTurn] = []
    for ids, statuses in zip(constraint_ids, constraint_status):
        if not isinstance(ids, list) or not isinstance(statuses, list):
            raise ValueError(
                "Each constraint turn must be represented as a list of ids and statuses."
            )
        normalized_statuses = [str(status).lower() for status in statuses]
        turn_constraints.append(
            {
                "constraint_ids": [str(constraint_id) for constraint_id in ids],
                "statuses": normalized_statuses,
            }
        )

    follow_up_messages = chat_prompts[1:]
    example["prompt"] = [chat_prompts[0]]
    example.setdefault("answer", "")
    example.setdefault("task", "multi-turn-constraints")
    example["info"] = {
        "follow_up_messages": follow_up_messages,
        "turn_constraints": turn_constraints,
        "total_turns": len(turn_constraints),
    }
    return example


async def _verify_turn(
    *,
    verifier: ConstraintVerifier,
    response_text: str,
    turn_index: int,
    conversation: list[ChatMessage],
    state: State,
    info: Info,
    constraint_id: str,
) -> bool:
    return bool(
        await maybe_await(
            verifier,
            response=response_text,
            turn_index=turn_index,
            messages=conversation,
            state=state,
            info=info,
            constraint_id=constraint_id,
        )
    )


async def _verify_rollout(
    *,
    verifier_pool: Mapping[str, ConstraintVerifier],
    prompt: Messages,
    completion: Messages,
    state: State,
    info: Info,
) -> bool:
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise TypeError("Multi-turn constraint environment expects chat-formatted messages.")

    turn_constraints: Sequence[ConstraintTurn] = info.get("turn_constraints", [])  # type: ignore[assignment]
    total_turns = info.get("total_turns", len(turn_constraints))
    if total_turns != len(turn_constraints):
        raise ValueError("Mismatch between total_turns and provided constraint metadata.")
    if total_turns == 0:
        return False

    conversation_prefix: list[ChatMessage] = list(prompt)
    verified_turns = 0
    for message in completion:
        if not isinstance(message, dict):
            raise TypeError("Each completion message must be a mapping for chat format.")
        conversation_prefix.append(message)  # conversation so far including current message
        if message.get("role") != "assistant":
            continue
        if verified_turns >= total_turns:
            break
        turn = turn_constraints[verified_turns]
        if len(turn["constraint_ids"]) != len(turn["statuses"]):
            raise ValueError("Constraint ids and statuses must align for each turn.")
        response_text = str(message.get("content", ""))
        for constraint_id, status in zip(turn["constraint_ids"], turn["statuses"]):
            if status != "enabled":
                continue
            verifier = verifier_pool.get(constraint_id)
            if verifier is None:
                raise KeyError(
                    f"No verifier registered for constraint id '{constraint_id}'."
                )
            is_valid = await _verify_turn(
                verifier=verifier,
                response_text=response_text,
                turn_index=verified_turns,
                conversation=list(conversation_prefix),
                state=state,
                info=info,
                constraint_id=constraint_id,
            )
            if not is_valid:
                return False
        verified_turns += 1
    return verified_turns >= total_turns


class MultiTurnConstraintEnv(vf.MultiTurnEnv):
    """Multi-turn environment that enforces per-turn constraint verifiers."""

    def __init__(
        self,
        *,
        verifier_pool: Mapping[str, ConstraintVerifier],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.verifier_pool = dict(verifier_pool)

    async def setup_state(self, state: State, **kwargs: Any) -> State:  # noqa: D401
        state = await super().setup_state(state, **kwargs)
        info = state.setdefault("info", {})
        if "turn_constraints" not in info:
            info["turn_constraints"] = []
        if "follow_up_messages" not in info:
            info["follow_up_messages"] = []
        info.setdefault("total_turns", len(info.get("turn_constraints", [])))
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        base_completed = await super().is_completed(messages, state, **kwargs)
        if base_completed:
            return True
        info = state.get("info", {})
        total_turns = info.get("total_turns", 0)
        return state["turn"] >= total_turns

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> tuple[Messages, State]:
        follow_ups: Sequence[ChatMessage] = state.get("info", {}).get(
            "follow_up_messages", []
        )
        next_index = state["turn"] - 1
        if 0 <= next_index < len(follow_ups):
            next_message = follow_ups[next_index]
            return [next_message], state
        return [], state


async def placeholder_verifier(**_: Any) -> bool:
    """Placeholder verifier implementation. Replace with task-specific logic."""

    raise NotImplementedError("Provide a task-specific verifier implementation.")


def _prepare_dataset(dataset: Dataset | DatasetDict) -> Dataset:
    """Normalize the dataset structure regardless of whether a split dict is provided."""

    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict must contain a 'train' split for training.")
        dataset = dataset["train"]
    return dataset.map(_prepare_example)


def load_environment(
    dataset_name: str | None = None,
    *,
    dataset_split: str = "train",
    dataset_config: str | None = None,
    dataset: Dataset | DatasetDict | None = None,
    verifier_pool: Mapping[str, ConstraintVerifier] | None = None,
    max_turns: int = -1,
    **kwargs: Any,
) -> vf.Environment:
    """Load the multi-turn constraint environment."""

    if dataset is None:
        if dataset_name is None:
            raise ValueError("Either a dataset or dataset_name must be provided.")
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)  # type: ignore[arg-type]
    prepared_dataset = _prepare_dataset(dataset)

    if verifier_pool is None:
        raise ValueError(
            "verifier_pool must be provided. Use `placeholder_verifier` as a template for your implementation."
        )

    parser = vf.Parser()

    async def constraint_reward(
        prompt: Messages,
        completion: Messages,
        state: State,
        info: Info,
        **_: Any,
    ) -> float:
        passed = await _verify_rollout(
            verifier_pool=verifier_pool,
            prompt=prompt,
            completion=completion,
            state=state,
            info=info,
        )
        return 1.0 if passed else 0.0

    rubric = vf.Rubric(parser=parser, funcs=[constraint_reward])

    environment = MultiTurnConstraintEnv(
        dataset=prepared_dataset,
        parser=parser,
        rubric=rubric,
        verifier_pool=verifier_pool,
        max_turns=max_turns,
        **kwargs,
    )
    return environment
