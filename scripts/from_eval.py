"""Convert Inspect ``.eval`` archives into this harness's corpus format.

The public LinuxArena trajectory mirror ships Inspect eval archives rather than
Control Tower JSON, which removes the ``ct`` CLI -- and its API token -- from the
critical path. An archive is a zip holding one JSON document per sample; a sample
holds a flat ``messages`` list where assistant turns carry ``tool_calls`` and the
matching results arrive as later ``role: "tool"`` messages joined on
``tool_call_id``.

Large strings are stored out of line as ``attachment://<hash>`` references and
resolved from the sample's ``attachments`` map. Unresolved references would reach
a monitor as literal placeholder text, so they are substituted here.

Trajectory-level ground truth is not in the archive. It comes from the dataset's
per-cell results table, joined on ``(model, task_id)``; ``side_task_passed`` marks
a trajectory malicious. Per-step attack labels are not published, so
``is_attack_step`` is left unset unless a labels file supplies it.

Usage:
    python -m scripts.from_eval --src evals/ --results precomputed_results.csv \\
        --out data/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from ledgerctl.runtime import configure_logging

logger = logging.getLogger("from_eval")

_ATTACHMENT_REF = re.compile(r"attachment://([0-9a-f]+)")


@dataclass(frozen=True)
class CellResult:
    """One row of the per-cell results table.

    Attributes:
        model: Short model name.
        task_id: ``{env}__{main_task}__{side_task}``.
        side_task_passed: Whether the side-task scorer succeeded. This is the
            trajectory-level malicious label.
        min_monitor_suspicion: The dataset's own monitor score, kept for
            comparison plots.
        excluded: Whether the cell was dropped from the validated denominator.
        exclusion_reason: Why, when excluded.
    """

    model: str
    task_id: str
    side_task_passed: bool
    min_monitor_suspicion: float | None
    excluded: bool
    exclusion_reason: str


def load_results(path: Path) -> dict[tuple[str, str], CellResult]:
    """Load the per-cell results table.

    Args:
        path: Path to ``precomputed_results.csv``.

    Returns:
        Mapping from ``(model, task_id)`` to its result row.

    Raises:
        FileNotFoundError: If the table is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"results table not found: {path}")
    out: dict[tuple[str, str], CellResult] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            suspicion = row.get("min_monitor_suspicion") or ""
            out[(row["model"], row["task_id"])] = CellResult(
                model=row["model"],
                task_id=row["task_id"],
                side_task_passed=row.get("side_task_passed", "").strip().lower() == "true",
                min_monitor_suspicion=float(suspicion) if suspicion.strip() else None,
                excluded=row.get("excluded", "").strip().lower() == "true",
                exclusion_reason=(row.get("exclusion_reason") or "").strip(),
            )
    logger.info("loaded %d result rows from %s", len(out), path)
    return out


def resolve_attachments(text: str, attachments: dict[str, str]) -> str:
    """Replace ``attachment://`` references with their stored content.

    Args:
        text: Possibly containing references.
        attachments: The sample's attachment map.

    Returns:
        Text with every resolvable reference substituted. Unresolvable ones are
        left in place rather than blanked, so a missing attachment is visible in
        the data instead of silently becoming an empty output.
    """
    if not text or "attachment://" not in text:
        return text
    return _ATTACHMENT_REF.sub(lambda m: attachments.get(m.group(1), m.group(0)), text)


def _content_to_text(content: Any, attachments: dict[str, str]) -> str:
    """Flatten an Inspect message content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return resolve_attachments(content, attachments)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(resolve_attachments(item, attachments))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(resolve_attachments(item["text"], attachments))
        return "\n".join(parts)
    return str(content)


def sample_to_actions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract executed tool calls and their results from one Inspect sample.

    Args:
        sample: Decoded sample document.

    Returns:
        Actions in Control Tower shape -- ``function``, ``arguments``, ``output``
        -- so the existing loader parses them unchanged.
    """
    attachments = sample.get("attachments") or {}
    messages = sample.get("messages") or []

    results: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results[str(message["tool_call_id"])] = _content_to_text(
                message.get("content"), attachments
            )

    actions: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id", ""))
            actions.append(
                {
                    "index": len(actions),
                    "function": str(call.get("function", "unknown")),
                    "arguments": call.get("arguments") or {},
                    "output": results.get(call_id, ""),
                }
            )
    return actions


def iter_samples(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield every sample document inside one ``.eval`` archive.

    Args:
        path: Path to the archive.

    Yields:
        The entry name and decoded sample.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("samples/") and name.endswith(".json"):
                    yield name, json.loads(archive.read(name))
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        logger.warning("skipping unreadable archive %s: %s", path.name, exc)


def parse_eval_path(path: Path, root: Path) -> tuple[str, str]:
    """Recover the model and task id from an archive's location.

    The mirror lays archives out as
    ``evals/{model}_stripped_{timestamp}/{task_id}/{run}.eval``.

    Args:
        path: Archive path.
        root: Directory the scan started from.

    Returns:
        The model short name and task id, empty when the layout does not match.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if len(parts) < 3:
        return "", ""
    model = parts[-3].split("_stripped_")[0]
    return model, parts[-2]


def convert(
    src: Path,
    results: dict[tuple[str, str], CellResult],
    labels: dict[str, list[int]],
    include_excluded: bool,
) -> Iterator[dict[str, Any]]:
    """Convert every archive under ``src`` into corpus documents.

    Args:
        src: Directory of ``.eval`` archives.
        results: Per-cell results, for the malicious label.
        labels: Optional mapping from session id to attack action indices.
        include_excluded: Whether to keep cells the dataset excluded.

    Yields:
        Corpus documents in Control Tower shape.
    """
    archives = sorted(src.rglob("*.eval"))
    logger.info("found %d archives under %s", len(archives), src)

    for archive in tqdm(archives, desc="convert", unit="eval", dynamic_ncols=True):
        model, task_id = parse_eval_path(archive, src)
        cell = results.get((model, task_id))
        if cell is None:
            logger.warning("no results row for (%s, %s); skipping", model, task_id)
            continue
        if cell.excluded and not include_excluded:
            continue

        for entry_name, sample in iter_samples(archive):
            actions = sample_to_actions(sample)
            if not actions:
                logger.warning("no tool calls in %s/%s", archive.name, entry_name)
                continue

            epoch = sample.get("epoch", 1)
            session_id = f"{model}__{task_id}__e{epoch}"
            attack_indices = labels.get(session_id, [])

            document: dict[str, Any] = {
                "trajectory_id": session_id,
                "side_task_success": cell.side_task_passed,
                "actions": actions,
                "metadata": {
                    "model": model,
                    "task_id": task_id,
                    "min_monitor_suspicion": cell.min_monitor_suspicion,
                    "excluded": cell.excluded,
                    "exclusion_reason": cell.exclusion_reason,
                    "n_actions": len(actions),
                },
            }
            if attack_indices:
                document["attack_analysis"] = {
                    "summary": "labelled offline",
                    "incriminating_actions": [
                        {"action_index": i, "incriminating_behaviour": "labelled"}
                        for i in attack_indices
                    ],
                }
            yield document


def main() -> None:
    """Convert a directory of Inspect archives into a JSONL corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="directory of .eval archives")
    parser.add_argument("--results", type=Path, required=True, help="precomputed_results.csv")
    parser.add_argument("--out", type=Path, required=True, help="destination JSONL corpus")
    parser.add_argument(
        "--labels",
        type=Path,
        help="optional JSON mapping session_id -> [attack action indices]",
    )
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    configure_logging()

    results = load_results(args.results)
    labels: dict[str, list[int]] = {}
    if args.labels and args.labels.exists():
        labels = {k: list(v) for k, v in json.loads(args.labels.read_text(encoding="utf-8")).items()}
        logger.info("loaded per-step labels for %d trajectories", len(labels))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"malicious": 0, "benign": 0, "labelled": 0}
    with open(args.out, "w", encoding="utf-8") as handle:
        for document in convert(args.src, results, labels, args.include_excluded):
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            counts["malicious" if document["side_task_success"] else "benign"] += 1
            if "attack_analysis" in document:
                counts["labelled"] += 1

    total = counts["malicious"] + counts["benign"]
    print()
    print(f"wrote {total} trajectories to {args.out}")
    print(f"  malicious (side_task_passed) : {counts['malicious']}")
    print(f"  benign                       : {counts['benign']}")
    print(f"  with per-step attack labels  : {counts['labelled']}")
    if counts["labelled"] < counts["malicious"]:
        print()
        print("note: without per-step labels, detection latency and attack-step recall")
        print("      are not computed. Safety at a fixed audit budget is unaffected.")


if __name__ == "__main__":
    main()
