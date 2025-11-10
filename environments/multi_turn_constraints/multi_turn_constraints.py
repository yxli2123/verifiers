"""Multi-turn RL environment with constraint-based verification."""

import asyncio
import builtins
import inspect
import logging
from collections import defaultdict
from typing import (
    Any,
    Mapping,
    Sequence,
    TypedDict,
    List,
    Literal,
    Dict,
    Tuple,
)

from datasets import load_dataset

import verifiers as vf

import signal
import functools

ALPHA = 2


# ----- Define Types -----
class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class VerifierPtr(TypedDict):
    """Metadata for a single turn's constraint configuration."""

    constraint_ids: Sequence[str]
    statuses: Sequence[str]
    placeholders: Sequence[str]


class Verifier(TypedDict):
    """Metadata for a single verifier."""

    func: str
    placeholder: str
    constraint_id: str
    result: bool | None


class Constraint(TypedDict):
    """Metadata for a single constraint."""

    id: str
    constraint: str
    func: Sequence[str]

    verifier_type: str
    type: str
    description: str
    content_template: bool
    verifier_prompt: str


class State(TypedDict):
    prompt: List[Message]
    completion: List[Message] | None
    answer: str | None
    task: str | None
    info: Dict[str, Any] | None
    example_id: int | None
    responses: List[Any]
    turn: int  # init 0, + 1 after calling rollout
    timing: Dict[Literal["generation_ms", "scoring_ms", "total_ms"], float]


class Info(TypedDict):
    follow_up_messages: List[Message]
    verifiers: List[VerifierPtr]
    total_turns: int


class MultiTurnInstance(TypedDict):
    """Metadata for a single multi-turn data."""

    prompt: List[Message]
    follow_up_messages: List[Message]
    verifiers: List[VerifierPtr]
    total_turns: int


class MultiTurnTrainingInstance(TypedDict):
    """Metadata for a single multi-turn training data in `verifiers` style (schema)."""

    prompt: List[Message]
    completion: List[Message] | None
    answer: str | None
    task: str | None
    info: Info | None
    example_id: int | None


class SingleVerification(TypedDict):
    pass_rate: float
    constraint_id: str


# ----- Helper functions -----
def _prepare_example(example: MultiTurnInstance) -> MultiTurnTrainingInstance:
    """Convert a raw dataset example into the format expected by the environment."""

    training_example: MultiTurnTrainingInstance = {
        "prompt": example["prompt"],
        "answer": "",
        "task": "multi-turn-instruction-following",
        "info": {
            "follow_up_messages": example["follow_up_messages"],
            "verifiers": example["verifiers"],
            "total_turns": example["total_turns"],
        },
        "completion": None,
        "example_id": None,
    }

    return training_example


def timeout(seconds: float):
    """Raise TimeoutError if the wrapped function runs longer than `seconds`.
    Unix-only (posix). Must be called from the main thread.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(signal, "SIGALRM"):
                raise RuntimeError("signal-based timeout requires Unix (SIGALRM).")

            def _handle_alarm(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds} seconds")

            old_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, _handle_alarm)
                # setitimer supports fractional seconds
                signal.setitimer(signal.ITIMER_REAL, seconds)
                return func(*args, **kwargs)
            finally:
                # always clean up timer & restore handler
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


@timeout(5.0)
def _execute_function(
    func_src: str,
    args: dict[str, Any],
    func_name: str = "evaluate",
) -> Any:
    """
    Compile `func_src`, locate the function `func_name` (or the first function defined),
    and call it with kwargs from `args`. Extra keys in `args` are ignored.
    Supports both def and async def.
    """

    # Give user code full access to Python builtins (HIGH RISK).
    # This includes `__import__`, so `import ...` inside func_src will work.
    g: dict[str, Any] = {"__builtins__": builtins}

    code = compile(func_src, filename="<user_function>", mode="exec")
    exec(code, g, g)

    if func_name not in g or not callable(g[func_name]):
        raise ValueError(f"Function '{func_name}' not found after compiling source.")

    fn = g[func_name]

    # Filter kwargs to the function signature
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in args.items() if k in sig.parameters}
    sig.bind(**filtered)  # raises if required args are missing

    # Call (await if coroutine)
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(**filtered))

    return fn(**filtered)


# ----- Define Environment -----
class MultiTurnConstraintEnv(vf.MultiTurnEnv):
    """Multi-turn environment that enforces per-turn constraint verifiers."""

    async def is_completed(
        self, messages: List[Message], state: State, **kwargs: Any
    ) -> bool:
        base_completed = await super().is_completed(messages, state, **kwargs)
        if base_completed:
            return True
        total_turns = state["info"]["total_turns"]
        return state["turn"] >= total_turns

    async def env_response(
        self, messages: List[Message], state: State, **kwargs: Any
    ) -> tuple[List[Message], State]:
        """Add a follow-up message (constraint) after each rollout."""
        follow_up_messages: List[Message] = state["info"]["follow_up_messages"]

        # Follow-up messages indexing from 0 and state["turn"] = 1, after the 1st rollout.
        follow_up_msg_idx = state["turn"] - 1
        if 0 <= follow_up_msg_idx < len(follow_up_messages):
            next_message = follow_up_messages[follow_up_msg_idx]
            return [next_message], state
        return [], state


# ----- Define Rubrics: Verifiers and Reward -----
def verify_single_turn(
    *,
    verifier_ptr: VerifierPtr,
    response: str,
    constraint_pool: Mapping[str, Constraint],
) -> Tuple[bool, List[SingleVerification]]:
    verifiers: List[Verifier] = []

    constraint_ids = verifier_ptr["constraint_ids"]
    statuses = verifier_ptr["statuses"]
    placeholders = verifier_ptr["placeholders"]
    for c_id, status, placeholder in zip(constraint_ids, statuses, placeholders):
        funcs = constraint_pool[c_id]["func"]
        if status == "enabled":
            for func in funcs:
                _verifier: Verifier = {
                    "func": func,
                    "placeholder": placeholder,
                    "constraint_id": c_id,
                    "result": None,
                }
                verifiers.append(_verifier)

    for _verifier in verifiers:
        # If placeholder != "", pass the placeholder to the verifier function.
        if _verifier["placeholder"]:
            args = {"response": response, "placeholder": _verifier["placeholder"]}
        else:
            args = {"response": response}
        result = _execute_function(_verifier["func"], args)

        # TODO: Return False early.
        # if return_early and not bool(result):
        #     return False, verifiers

        _verifier.setdefault("result", result)

    final_result = all(ver["result"] for ver in verifiers)

    pass_flags = defaultdict(list)
    for _verifier in verifiers:
        pass_flags[_verifier["constraint_id"]].append(1 if _verifier["result"] else 0)

    verification: List[SingleVerification] = []
    for c_id, flag in pass_flags.items():
        s_ver: SingleVerification = {
            "pass_rate": sum(flag) / len(flag) if len(flag) else 0.0,
            "constraint_id": c_id,
        }
        verification.append(s_ver)

    return final_result, verification


def verify_multi_turn(
    *,
    prompt: List[Message],
    completion: List[Message],
    state: State,
    info: Info,
    constraint_pool: Mapping[str, Constraint],
) -> Tuple[List[bool | None], List[List[SingleVerification] | None]]:
    total_turns = info["total_turns"]
    verifiers = info["verifiers"]
    verified_result: List[bool | None] = [None] * total_turns
    verifier_log: List[List[SingleVerification] | None] = [None] * total_turns

    # Obtain responses from all turns.
    responses = []
    for conversation in completion:
        if conversation["role"] == "assistant":
            responses.append(conversation["content"])

    if len(responses) < total_turns:
        logging.error(
            f"Conversation has {len(responses)} turns less than expected {total_turns} turns."
        )
        return verified_result, verifier_log

    for i, response, verifier in zip(range(total_turns), responses, verifiers):
        passed, log_verifier = verify_single_turn(
            verifier_ptr=verifier,
            response=response,
            constraint_pool=constraint_pool,
        )
        verified_result[i] = passed
        verifier_log[i] = log_verifier

    return verified_result, verifier_log


async def placeholder_verifier(**_: Any) -> bool:
    """Placeholder verifier implementation. Replace with task-specific logic."""

    raise NotImplementedError("Provide a task-specific verifier implementation.")


def load_environment(
    *dataset_path: str,
    constraint_path: str = "yxli2123/verifiable-constraints",
    max_turns: int = 8,
    **kwargs: Any,
) -> vf.Environment:
    """Load the multi-turn constraint environment."""

    dataset = load_dataset(dataset_path, split="train")  # type: ignore[arg-type]
    prepared_dataset = _prepare_example(dataset)

    constraint_pool: List[Constraint] = load_dataset(constraint_path, split="train")
    indexed_constraint = {c["id"]: c for c in constraint_pool}

    parser = vf.Parser()

    async def reward_func(
        prompt: List[Message],
        completion: List[Message],
        state: State,
        info: Info,
        **_: Any,
    ) -> float:
        reward = 0.0
        ver_result, ver_log = verify_multi_turn(
            prompt=prompt,
            completion=completion,
            state=state,
            info=info,
            constraint_pool=indexed_constraint,
        )

        # Progress reward.
        for passed in ver_result:
            if passed:
                reward += 1.0 / len(ver_result)
            else:
                # No reward for all following turns, even if they are correct.
                break

        # All turns are correct. Give more reward if the turns are long.
        if all(ver_result):
            reward += 1 * ALPHA ** len(ver_result)

        return reward

    rubric = vf.Rubric(parser=parser, funcs=[reward_func])

    environment = MultiTurnConstraintEnv(
        dataset=prepared_dataset,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )
    return environment
