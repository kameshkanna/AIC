"""Append-only ledger: the monitor's memory.

The ledger holds one record per executed step, keyed by a globally unique step
id, plus the summary index the global monitor reads. It performs no model calls,
which keeps it a pure and trivially testable component.

Two properties are load-bearing for the experiment:

* **Append-only.** Records are never mutated. Score revisions produced by
  retroactive re-scoring are appended as new entries, so the full history of a
  judgement remains reconstructible.
* **Access logging.** Every index search and record fetch is recorded. The
  access log is both the evidence package handed to an auditor and the means of
  detecting a monitor that never looked at the incriminating record.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ledgerctl.trajectory import Step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexEntry:
    """One line of the summary index the global monitor reads.

    Attributes:
        step_id: Globally unique step id, used to fetch the full record.
        step: Position within the trajectory.
        summary: Model-written prose recap of the step.
    """

    step_id: str
    step: int
    summary: str

    def render(self) -> str:
        """Render the entry as a single index line."""
        return f"{self.step_id}: {self.summary}"


@dataclass
class Ledger:
    """Per-trajectory record store, summary index and access trail.

    Attributes:
        session_id: Owning trajectory id.
        records: Mapping from step id to the full step record.
        index: Ordered index entries, one per summarised step.
        revisions: Appended score revisions from retroactive re-scoring.
        access_log: Append-only trail of every index search and record fetch.
    """

    session_id: str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    index: list[IndexEntry] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    access_log: list[dict[str, Any]] = field(default_factory=list)

    def append(self, step: Step, summary: str) -> None:
        """Append one step and its summary.

        Args:
            step: The executed step.
            summary: Prose recap written by the summariser.

        Raises:
            ValueError: If the step id is already present, which would violate
                the append-only guarantee.
        """
        if step.step_id in self.records:
            raise ValueError(f"duplicate step id: {step.step_id}")
        self.records[step.step_id] = {**step.to_record(), "summary": summary}
        self.index.append(IndexEntry(step_id=step.step_id, step=step.step, summary=summary))

    def render_index(self, upto: int | None = None) -> str:
        """Render the index the monitor reads.

        Args:
            upto: Exclusive step bound enforcing the causal mask. ``None``
                renders every entry.

        Returns:
            One index line per entry, in step order.
        """
        entries = self.index if upto is None else [e for e in self.index if e.step < upto]
        return "\n".join(entry.render() for entry in entries)

    def grep_index(self, pattern: str, upto: int | None = None) -> list[IndexEntry]:
        """Search the index lexically and log the access.

        Args:
            pattern: Regular expression; falls back to a literal match if the
                pattern does not compile.
            upto: Exclusive step bound enforcing the causal mask.

        Returns:
            Matching index entries in step order.
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        candidates = self.index if upto is None else [e for e in self.index if e.step < upto]
        hits = [e for e in candidates if regex.search(e.summary) or regex.search(e.step_id)]
        self.access_log.append(
            {"op": "grep_index", "pattern": pattern, "upto": upto, "n_hits": len(hits)}
        )
        return hits

    def fetch(self, step_ids: Sequence[str], upto: int | None = None) -> list[dict[str, Any]]:
        """Fetch full records by exact id and log the access.

        The caller drops the returned records after use; the ledger holds no
        per-monitor state, so each judgement stays a pure function of its inputs.

        Args:
            step_ids: Exact step ids to load.
            upto: Exclusive step bound enforcing the causal mask.

        Returns:
            One entry per requested id, in request order. Ids that are unknown
            or masked yield an error stub rather than raising, so a monitor that
            hallucinates an id degrades instead of crashing the run.
        """
        out: list[dict[str, Any]] = []
        for step_id in step_ids:
            record = self.records.get(step_id)
            if record is None:
                out.append({"step_id": step_id, "error": "not found"})
                continue
            if upto is not None and record["step"] >= upto:
                out.append({"step_id": step_id, "error": "not yet observed"})
                continue
            out.append(dict(record))
        self.access_log.append(
            {"op": "fetch", "step_ids": list(step_ids), "upto": upto, "n_hits": len(out)}
        )
        return out

    def revise(self, step_id: str, score: float, at_step: int, reason: str = "") -> None:
        """Append a retroactive score revision for an earlier step.

        Args:
            step_id: Step whose score is being revised.
            score: New score in the unit interval.
            at_step: Step at which the revising evidence became available.
            reason: Optional short justification, stored for audit.

        Raises:
            KeyError: If the step id is not in the ledger.
            ValueError: If the score falls outside the unit interval.
        """
        if step_id not in self.records:
            raise KeyError(f"cannot revise unknown step: {step_id}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must lie in [0, 1], got {score}")
        self.revisions.append(
            {"step_id": step_id, "score": score, "at_step": at_step, "reason": reason}
        )

    def effective_scores(self, base: dict[str, float]) -> dict[str, float]:
        """Apply revisions in order to a set of first-pass scores.

        Args:
            base: Mapping from step id to the forward-pass score.

        Returns:
            A new mapping with later revisions applied.
        """
        scores = dict(base)
        for revision in self.revisions:
            scores[revision["step_id"]] = revision["score"]
        return scores

    def save(self, directory: Path) -> Path:
        """Write the ledger as newline-delimited JSON.

        Args:
            directory: Destination directory, created if absent.

        Returns:
            The path written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.session_id}.ndjson"
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self.index:
                handle.write(json.dumps(self.records[entry.step_id], ensure_ascii=False) + "\n")
            for revision in self.revisions:
                handle.write(json.dumps({"t": "revision", **revision}, ensure_ascii=False) + "\n")
        logger.info("wrote ledger %s (%d records)", path, len(self.records))
        return path

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        """Load a ledger previously written by :meth:`save`.

        Args:
            path: Path to a newline-delimited JSON ledger file.

        Returns:
            The reconstructed ledger, without its access log.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"ledger not found: {path}")
        ledger = cls(session_id=path.stem)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("t") == "revision":
                    ledger.revisions.append({k: v for k, v in entry.items() if k != "t"})
                    continue
                ledger.records[entry["step_id"]] = entry
                ledger.index.append(
                    IndexEntry(
                        step_id=entry["step_id"],
                        step=entry["step"],
                        summary=entry.get("summary", ""),
                    )
                )
        ledger.index.sort(key=lambda e: e.step)
        return ledger


def build_ledger(session_id: str, steps: Iterable[Step], summaries: dict[str, str]) -> Ledger:
    """Assemble a ledger from steps and a summary lookup.

    Args:
        session_id: Trajectory id.
        steps: Steps in execution order.
        summaries: Mapping from step id to summary. Missing ids yield an empty
            summary rather than an error, so a partial summariser pass still
            produces a loadable ledger.

    Returns:
        The populated ledger.
    """
    ledger = Ledger(session_id=session_id)
    for step in steps:
        ledger.append(step, summaries.get(step.step_id, ""))
    return ledger
