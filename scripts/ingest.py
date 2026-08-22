"""Convert Control Tower trajectory JSON into this harness's corpus format.

Control Tower stores one JSON document per trajectory, with actions under
``actions`` and side-task metadata at the top level. This script normalises a
directory of those documents into a single JSONL corpus, and reports which field
aliases actually matched so a schema drift is visible rather than silent.

Fetch the data first (see README):

    pip install git+https://github.com/linuxarena/control-tower
    ct traj download --dataset LaStraj
    ct traj download -e <env> --tag baseline -n 100     # benign runs

Then:

    python -m scripts.ingest --src ~/.control-tower/trajectories --out data/trajectories.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from ledgerctl.runtime import configure_logging
from ledgerctl.trajectory import (
    FIELD_ALIASES,
    MALICIOUS_KEYS,
    SESSION_ID_KEYS,
    STEP_LIST_KEYS,
    parse_trajectory,
)

logger = logging.getLogger("ingest")


def iter_documents(src: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield every JSON document under ``src``.

    Args:
        src: File or directory of ``.json``/``.jsonl`` trajectory documents.

    Yields:
        The source path and the decoded document.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
    """
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")
    paths = [src] if src.is_file() else sorted(
        [*src.rglob("*.json"), *src.rglob("*.jsonl")]
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    yield path, json.loads(line)
        else:
            yield path, json.loads(text)


def audit_schema(documents: list[dict[str, Any]]) -> dict[str, Counter]:
    """Report which known aliases are present in a document set.

    Run this before trusting a conversion: an alias with a zero count means the
    corpus uses a name this harness does not know, and every downstream number
    would be computed over empty fields.

    Args:
        documents: Decoded trajectory documents.

    Returns:
        Mapping from logical field name to a counter over matched aliases.
    """
    found: dict[str, Counter] = {
        "step_list": Counter(),
        "session_id": Counter(),
        "malicious": Counter(),
        **{name: Counter() for name in FIELD_ALIASES},
    }
    for document in documents:
        for key in STEP_LIST_KEYS:
            if key in document:
                found["step_list"][key] += 1
                break
        for key in SESSION_ID_KEYS:
            if key in document:
                found["session_id"][key] += 1
                break
        for key in MALICIOUS_KEYS:
            if key in document:
                found["malicious"][key] += 1
                break
        steps = next((document[k] for k in STEP_LIST_KEYS if k in document), [])
        if not isinstance(steps, list):
            continue
        for entry in steps:
            if not isinstance(entry, dict):
                continue
            for logical, aliases in FIELD_ALIASES.items():
                for alias in aliases:
                    if alias in entry:
                        found[logical][alias] += 1
                        break
    return found


def main() -> None:
    """Convert a Control Tower export into the harness corpus format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="file or directory of trajectory JSON")
    parser.add_argument("--out", type=Path, help="destination JSONL corpus; omit with --audit-only")
    parser.add_argument("--audit-only", action="store_true", help="report schema match and exit")
    args = parser.parse_args()

    configure_logging()

    documents = [document for _, document in iter_documents(args.src)]
    logger.info("read %d documents from %s", len(documents), args.src)

    audit = audit_schema(documents)
    print("\nschema audit (alias -> documents matched)")
    print("-" * 60)
    for logical, counter in audit.items():
        if counter:
            matched = ", ".join(f"{alias}={count}" for alias, count in counter.most_common())
        else:
            matched = "NO MATCH -- add the real name to FIELD_ALIASES"
        print(f"{logical:<16} {matched}")
    print("-" * 60)

    if args.audit_only:
        return

    if args.out is None:
        raise SystemExit("--out is required unless --audit-only is given")

    unmatched = [name for name in ("step_list", "command", "output") if not audit[name]]
    if unmatched:
        raise SystemExit(
            f"refusing to convert: no alias matched for {unmatched}. "
            "Inspect a source document and extend FIELD_ALIASES in trajectory.py."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for document in tqdm(documents, desc="convert", unit="traj", dynamic_ncols=True):
            trajectory = parse_trajectory(document)
            handle.write(
                json.dumps(
                    {
                        "session_id": trajectory.session_id,
                        "is_malicious": trajectory.is_malicious,
                        "attack_completion_step": trajectory.attack_completion_step,
                        "steps": [
                            {
                                "step": step.step,
                                "timestamp": step.timestamp,
                                "command": step.command,
                                "output": step.output,
                                "is_attack_step": step.is_attack_step,
                            }
                            for step in trajectory.steps
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    logger.info("wrote %d trajectories to %s", written, args.out)


if __name__ == "__main__":
    main()
