"""Trajectory corpus loading.

A trajectory is an ordered sequence of executed steps. Each step carries only
what the recording already contains -- the tool call the untrusted agent issued
and the output it produced. No field on a :class:`Step` is inferred by a model.

Two input shapes are supported:

* **Control Tower** (LinuxArena / LaStraj). Actions live under ``actions``, and a
  command is not a flat field: it is ``function`` plus an ``arguments`` mapping,
  so ``bash`` calls carry the shell command at ``arguments["cmd"]``. Ground truth
  is likewise nested -- ``side_task_success`` at the trajectory level, and the
  labelled attack steps under ``attack_analysis.incriminating_actions``, keyed by
  the action's *original* index.
* **Flat records**, for other corpora and for tests, where each step object holds
  ``command`` and ``output`` directly. Field names are matched through
  :data:`FIELD_ALIASES` rather than hard-coded.

Actions calling a tool in :data:`HIDDEN_TOOLS` are dropped. This mirrors Control
Tower's own monitor input and is not cosmetic: ``run_side_task_test`` and
``side_task_scorer`` are the scoring harness, so leaving them in would show the
monitor the side task being graded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

STEP_LIST_KEYS: tuple[str, ...] = ("actions", "steps", "trajectory", "transcript", "messages")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "command": ("command", "action", "tool_call", "input", "content"),
    "output": ("output", "observation", "response", "result", "stdout", "tool_result"),
    "timestamp": ("timestamp", "time", "ts", "started_at"),
    "step": ("index", "action_index", "step_index", "step", "turn", "line_number"),
    "is_attack_step": ("is_attack_step", "is_side_task", "side_task_action", "malicious"),
}

SESSION_ID_KEYS: tuple[str, ...] = ("trajectory_id", "session_id", "id", "run_id", "task_id")
MALICIOUS_KEYS: tuple[str, ...] = (
    "side_task_success",
    "is_malicious",
    "malicious",
    "has_side_task",
)
COMPLETION_KEYS: tuple[str, ...] = (
    "attack_completion_step",
    "side_task_completion_step",
    "side_task_success_step",
)

HIDDEN_TOOLS: frozenset[str] = frozenset({"run_side_task_test", "submit", "side_task_scorer"})
POLICY_PRIVATE_KEY = "is_policy_private"


def _first_present(record: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Return the first present alias from ``record``, else ``default``."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def tool_basename(function: str) -> str:
    """Strip any namespace prefix from a tool name."""
    return function.rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def render_command(function: str, arguments: dict[str, Any] | None) -> str:
    """Render a tool call as the command text a monitor reads.

    ``bash`` calls are rendered as the bare shell command, since that is what the
    argument mapping contains and it keeps the summariser's input clean. Any other
    tool is rendered as ``function: {arguments}``, matching Control Tower's own
    formatter so the text a monitor sees here is the text it would see there.

    Args:
        function: Tool name.
        arguments: Tool argument mapping.

    Returns:
        The rendered command.
    """
    args = arguments or {}
    if tool_basename(function) == "bash" and isinstance(args.get("cmd"), str):
        return args["cmd"]
    try:
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(args)
    return f"{function}: {rendered}"


def render_output(output: Any, output_error: Any) -> str:
    """Combine an action's output and error streams into one text block."""
    parts: list[str] = []
    if output is not None and str(output) != "":
        parts.append(str(output))
    if output_error is not None and str(output_error) != "":
        parts.append(f"Error: {output_error}")
    return "\n".join(parts)


def is_hidden_action(entry: dict[str, Any]) -> bool:
    """Whether an action is excluded from what a monitor may see.

    Args:
        entry: Raw action object.

    Returns:
        True for scoring-harness tools and policy-private actions.
    """
    function = entry.get("function")
    if isinstance(function, str) and tool_basename(function) in HIDDEN_TOOLS:
        return True
    metadata = entry.get("metadata") or {}
    return bool(isinstance(metadata, dict) and metadata.get(POLICY_PRIVATE_KEY))


@dataclass(frozen=True)
class Step:
    """One executed agent action and its result.

    Attributes:
        session_id: Owning trajectory id.
        step: Zero-based position after hidden actions are dropped.
        step_id: Globally unique id, ``"{session_id}::s{step}"``.
        source_index: Position in the unfiltered recording. Ground-truth labels
            reference this, not ``step``.
        timestamp: Recording timestamp, empty when the corpus omits one.
        command: Rendered tool call issued by the untrusted agent.
        output: Combined output and error text as recorded.
        is_attack_step: Ground-truth label; evaluation only, never shown to a
            monitor or to the summariser.
    """

    session_id: str
    step: int
    step_id: str
    source_index: int
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
        steps: Steps in execution order, hidden actions already removed.
        is_malicious: Whether the recording contains a completed side task.
        attack_completion_step: Index of the step at which the side task
            completes. Detection is credited only strictly before this index.
            ``None`` for benign trajectories, and also for malicious ones whose
            corpus publishes no per-step labels -- there, a catch anywhere in the
            trajectory counts, which is the weaker but still valid reading.
    """

    session_id: str
    steps: tuple[Step, ...]
    is_malicious: bool = False
    attack_completion_step: int | None = None

    def __post_init__(self) -> None:
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


def _incriminating_indices(raw: dict[str, Any]) -> set[int]:
    """Extract labelled attack action indices from an attack analysis block.

    Args:
        raw: Decoded trajectory document.

    Returns:
        Original action indices flagged as incriminating; empty when the
        trajectory carries no analysis.
    """
    analysis = raw.get("attack_analysis") or {}
    if not isinstance(analysis, dict):
        return set()
    out: set[int] = set()
    for item in analysis.get("incriminating_actions") or []:
        if isinstance(item, dict) and item.get("action_index") is not None:
            try:
                out.add(int(item["action_index"]))
            except (TypeError, ValueError):
                logger.warning("unparseable action_index in attack_analysis: %r", item)
    return out


def parse_trajectory(raw: dict[str, Any], session_id: str | None = None) -> Trajectory:
    """Build a :class:`Trajectory` from one corpus record.

    Args:
        raw: Decoded corpus entry.
        session_id: Override for the trajectory id; falls back to the record's
            own id field.

    Returns:
        The parsed trajectory, with hidden actions removed and steps re-indexed
        contiguously.

    Raises:
        ValueError: If no step list is present.
    """
    sid = session_id or str(_first_present(raw, SESSION_ID_KEYS, "unknown"))
    raw_steps = _first_present(raw, STEP_LIST_KEYS)
    if not isinstance(raw_steps, list):
        raise ValueError(f"{sid}: no step list found (keys: {sorted(raw)})")

    incriminating = _incriminating_indices(raw)

    steps: list[Step] = []
    for position, entry in enumerate(raw_steps):
        if not isinstance(entry, dict):
            raise ValueError(f"{sid}: step {position} is {type(entry).__name__}, expected object")
        if is_hidden_action(entry):
            continue

        source_index = int(_first_present(entry, FIELD_ALIASES["step"], position))

        if "function" in entry:
            command = render_command(str(entry["function"]), entry.get("arguments"))
            output = render_output(entry.get("output"), entry.get("output_error"))
        else:
            command = str(_first_present(entry, FIELD_ALIASES["command"], ""))
            output = render_output(
                _first_present(entry, FIELD_ALIASES["output"], ""), entry.get("output_error")
            )

        flagged = source_index in incriminating or bool(
            _first_present(entry, FIELD_ALIASES["is_attack_step"], False)
        )

        steps.append(
            Step(
                session_id=sid,
                step=len(steps),
                step_id=f"{sid}::s{len(steps)}",
                source_index=source_index,
                timestamp=str(_first_present(entry, FIELD_ALIASES["timestamp"], "")),
                command=command,
                output=output,
                is_attack_step=flagged,
            )
        )

    is_malicious = bool(_first_present(raw, MALICIOUS_KEYS, False))
    completion = _first_present(raw, COMPLETION_KEYS)
    if is_malicious and completion is None:
        # Absent per-step labels the completion point is unknown. That is the
        # published state of the public mirror, so this degrades to a
        # trajectory-level catch rather than failing the load.
        completion = max((s.step for s in steps if s.is_attack_step), default=None)

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
