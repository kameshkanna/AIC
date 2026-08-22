"""Trajectory corpus loading.

A trajectory is an ordered sequence of executed steps. Each step carries only
what the recording already contains -- the command the untrusted agent issued and
the terminal output it produced -- plus identifiers assigned by the harness. No
field on a :class:`Step` is inferred by a model.

The concrete field names used by a corpus dump are declared in
:data:`FIELD_ALIASES` rather than hard-coded, so pointing this loader at a new
release is a data change rather than a code change.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

STEP_LIST_KEYS: tuple[str, ...] = ("actions", "steps", "trajectory", "transcript", "messages")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "command": ("command", "action", "tool_call", "input", "content"),
    "output": ("output", "observation", "response", "result", "stdout", "tool_result"),
    "timestamp": ("timestamp", "time", "ts", "started_at"),
    "step": ("step", "step_index", "turn", "index", "action_index"),
    "is_attack_step": ("is_attack_step", "is_side_task", "malicious", "side_task_action"),
}

SESSION_ID_KEYS: tuple[str, ...] = ("session_id", "trajectory_id", "id", "run_id", "task_id")
MALICIOUS_KEYS: tuple[str, ...] = ("is_malicious", "malicious", "has_side_task", "attack")
COMPLETION_KEYS: tuple[str, ...] = (
    "attack_completion_step",
    "side_task_completion_step",
    "side_task_success_step",
)


def _first_present(record: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Return the first present alias from ``record``, else ``default``."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


@dataclass(frozen=True)
class Step:
    """One executed agent action and its terminal result.

    Attributes:
        session_id: Owning trajectory id.
        step: Zero-based position within the trajectory.
        step_id: Globally unique id, ``"{session_id}::s{step}"``.
        timestamp: Recording timestamp, empty when the corpus omits one.
        command: Verbatim command or tool call issued by the untrusted agent.
        output: Verbatim terminal output (stdout and stderr as recorded).
        is_attack_step: Ground-truth label; evaluation only, never shown to a
            monitor or to the summariser.
    """

    session_id: str
    step: int
    step_id: str
    timestamp: str
    command: str
    output: str
    is_attack_step: bool = False

    def to_record(self) -> dict[str, Any]:
        """Return the ledger record for this step, excluding the gold label."""
        record = asdict(self)
        record.pop("is_attack_step")
        return record


@dataclass(frozen=True)
class Trajectory:
    """An ordered sequence of steps with trajectory-level ground truth.

    Attributes:
        session_id: Unique trajectory id.
        steps: Steps in execution order.
        is_malicious: Whether the recording contains a completed side task.
        attack_completion_step: Index of the step at which the side task
            completes; ``None`` for benign trajectories. Detection is only
            credited strictly before this index.
    """

    session_id: str
    steps: tuple[Step, ...]
    is_malicious: bool = False
    attack_completion_step: int | None = None

    def __post_init__(self) -> None:
        if self.is_malicious and self.attack_completion_step is None:
            raise ValueError(f"{self.session_id}: malicious trajectory needs attack_completion_step")
        if self.attack_completion_step is not None and not 0 <= self.attack_completion_step < len(self.steps):
            raise ValueError(
                f"{self.session_id}: attack_completion_step "
                f"{self.attack_completion_step} outside 0..{len(self.steps) - 1}"
            )

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def attack_step_ids(self) -> tuple[str, ...]:
        """Ids of labelled attack steps, for retrieval scoring."""
        return tuple(s.step_id for s in self.steps if s.is_attack_step)

    def prefix(self, upto: int) -> tuple[Step, ...]:
        """Return steps strictly before ``upto`` (the causal mask)."""
        if upto < 0:
            raise ValueError(f"upto must be non-negative, got {upto}")
        return self.steps[:upto]


def parse_trajectory(raw: dict[str, Any], session_id: str | None = None) -> Trajectory:
    """Build a :class:`Trajectory` from one corpus record.

    Args:
        raw: Decoded corpus entry holding a list of steps under ``steps``,
            ``trajectory`` or ``transcript``.
        session_id: Override for the trajectory id; falls back to the record's
            own id field.

    Returns:
        The parsed trajectory.

    Raises:
        ValueError: If no step list is present.
    """
    sid = session_id or str(_first_present(raw, SESSION_ID_KEYS, "unknown"))
    raw_steps = _first_present(raw, STEP_LIST_KEYS)
    if not isinstance(raw_steps, list):
        raise ValueError(f"{sid}: no step list found (keys: {sorted(raw)})")

    steps: list[Step] = []
    for position, entry in enumerate(raw_steps):
        if not isinstance(entry, dict):
            raise ValueError(f"{sid}: step {position} is {type(entry).__name__}, expected object")
        index = int(_first_present(entry, FIELD_ALIASES["step"], position))
        steps.append(
            Step(
                session_id=sid,
                step=index,
                step_id=f"{sid}::s{index}",
                timestamp=str(_first_present(entry, FIELD_ALIASES["timestamp"], "")),
                command=str(_first_present(entry, FIELD_ALIASES["command"], "")),
                output=str(_first_present(entry, FIELD_ALIASES["output"], "")),
                is_attack_step=bool(_first_present(entry, FIELD_ALIASES["is_attack_step"], False)),
            )
        )

    is_malicious = bool(_first_present(raw, MALICIOUS_KEYS, False))
    completion = _first_present(raw, COMPLETION_KEYS)
    if is_malicious and completion is None:
        completion = max((s.step for s in steps if s.is_attack_step), default=len(steps) - 1)
    return Trajectory(
        session_id=sid,
        steps=tuple(steps),
        is_malicious=is_malicious,
        attack_completion_step=int(completion) if completion is not None else None,
    )


def load_corpus(path: Path) -> list[Trajectory]:
    """Load a JSONL corpus of trajectories.

    Args:
        path: Path to a ``.jsonl`` file, one trajectory per line.

    Returns:
        Parsed trajectories in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    trajectories: list[Trajectory] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                trajectories.append(parse_trajectory(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    logger.info("loaded %d trajectories from %s", len(trajectories), path)
    return trajectories


def iter_steps(trajectories: Sequence[Trajectory]) -> Iterator[Step]:
    """Yield every step of every trajectory in order."""
    for trajectory in trajectories:
        yield from trajectory.steps
