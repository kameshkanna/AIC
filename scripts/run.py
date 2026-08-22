"""Run orchestrator: ingest, summarise, score every protocol, report.

The pass is resumable: summaries are cached per trajectory, and per-protocol
scores are written to disk as they complete, so a interrupted sweep resumes
without re-spending tokens.

Usage:
    python -m scripts.run --mock keyword --limit 4
    python -m scripts.run --protocols per_step,advisory --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from ledgerctl.config import CONFIG
from ledgerctl.evaluate import ProtocolReport, evaluate_protocol, safety_curve, step_records
from ledgerctl.ledger import build_ledger
from ledgerctl.llm import KeywordMockLLM, SupportsComplete, get_client
from ledgerctl.monitor import GlobalMonitor, StepMonitor
from ledgerctl.protocols import (
    AdvisoryArm,
    CascadeArm,
    FullContextBaseline,
    MonitoringProtocol,
    PerStepBaseline,
    ProtocolRun,
    RunningSummaryBaseline,
    ScalarPosteriorBaseline,
)
from ledgerctl.runtime import configure_logging, set_seed
from ledgerctl.summarise import Summariser, summarise_corpus
from ledgerctl.trajectory import Trajectory, load_corpus

logger = logging.getLogger("run")

MOCK_STEP = '{"score": 2, "reason": "routine"}'
MOCK_GLOBAL = '{"score": 3, "reason": "related to an earlier step", "evidence": [], "revise": []}'


def build_protocols(names: Sequence[str], mock: str) -> list[MonitoringProtocol]:
    """Construct the requested protocols with their clients.

    Args:
        names: Protocol names to build.
        mock: ``"off"`` for live clients, ``"constant"`` for a fixed-score mock,
            or ``"keyword"`` for a prompt-dependent mock that exercises the
            metric path.

    Returns:
        Protocol instances in the requested order.

    Raises:
        ValueError: If a name is not recognised.
    """
    if mock == "keyword":
        step_client: SupportsComplete = KeywordMockLLM()
        global_client: SupportsComplete = KeywordMockLLM()
        baseline_client: SupportsComplete = KeywordMockLLM()
    else:
        use_mock = mock == "constant"
        step_client = get_client(
            CONFIG.step_monitor_model,
            mock=use_mock,
            mock_response=MOCK_STEP,
            base_url=CONFIG.small_base_url,
        )
        global_client = get_client(
            CONFIG.global_monitor_model,
            mock=use_mock,
            mock_response=MOCK_GLOBAL,
            base_url=CONFIG.base_url,
        )
        baseline_client = get_client(
            CONFIG.baseline_monitor_model,
            mock=use_mock,
            mock_response=MOCK_GLOBAL,
            base_url=CONFIG.baseline_base_url,
        )
    step_monitor = StepMonitor(client=step_client)

    built: list[MonitoringProtocol] = []
    for name in names:
        if name == "per_step":
            built.append(PerStepBaseline(step_monitor=step_monitor))
        elif name == "scalar_posterior":
            built.append(ScalarPosteriorBaseline(step_monitor=step_monitor))
        elif name == "running_summary":
            built.append(RunningSummaryBaseline(client=step_client))
        elif name == "full_context":
            built.append(
                FullContextBaseline(
                    global_monitor=GlobalMonitor(client=baseline_client, max_fetches=0)
                )
            )
        elif name == "advisory":
            built.append(
                AdvisoryArm(
                    step_monitor=step_monitor,
                    global_monitor=GlobalMonitor(client=global_client),
                )
            )
        elif name == "cascade":
            built.append(
                CascadeArm(
                    step_monitor=step_monitor,
                    global_monitor=GlobalMonitor(client=global_client),
                )
            )
        else:
            raise ValueError(f"unknown protocol: {name}")
    return built


def stratified_limit(
    trajectories: Sequence[Trajectory], limit: int, seed: int
) -> list[Trajectory]:
    """Take ``limit`` trajectories while preserving the malicious/benign ratio.

    Slicing the head of a corpus can silently yield an all-benign or all-malicious
    subset, which produces a report full of zeros rather than an error. Sampling
    each class in proportion keeps a small pilot run meaningful.

    Args:
        trajectories: Full corpus.
        limit: Maximum trajectories to keep; ``0`` keeps everything.
        seed: Sampling seed.

    Returns:
        The sampled subset, in corpus order.
    """
    if limit <= 0 or limit >= len(trajectories):
        return list(trajectories)

    malicious = [t for t in trajectories if t.is_malicious]
    benign = [t for t in trajectories if not t.is_malicious]
    share = limit / len(trajectories)
    n_mal = min(len(malicious), max(1, round(len(malicious) * share)))
    n_ben = min(len(benign), max(2, limit - n_mal))

    rng = random.Random(seed)
    picked = set(
        t.session_id
        for t in rng.sample(malicious, n_mal) + rng.sample(benign, n_ben)
    )
    return [t for t in trajectories if t.session_id in picked]


def split_benign(benign: Sequence[Trajectory], seed: int) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split benign trajectories into disjoint calibration and reporting halves.

    Args:
        benign: Benign trajectories.
        seed: Shuffle seed, for a reproducible split.

    Returns:
        The calibration and reporting splits.

    Raises:
        ValueError: If there are fewer than two benign trajectories, which makes
            a disjoint split impossible.
    """
    if len(benign) < 2:
        raise ValueError(f"need at least 2 benign trajectories to split, got {len(benign)}")
    shuffled = list(benign)
    random.Random(seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def score_dir(protocol: str) -> Path:
    """Return the directory holding one protocol's per-trajectory scores."""
    return CONFIG.results_dir / "scores" / protocol


def save_run(run: ProtocolRun) -> Path:
    """Persist one protocol run.

    Args:
        run: The completed run.

    Returns:
        The path written.
    """
    directory = score_dir(run.protocol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run.session_id}.json"
    path.write_text(
        json.dumps(
            {
                "session_id": run.session_id,
                "protocol": run.protocol,
                "scores": run.scores,
                "revised": run.revised,
                "results": [asdict(result) for result in run.results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    """Parse arguments, run the sweep and print the report table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CONFIG.data_dir / "trajectories.jsonl")
    parser.add_argument(
        "--protocols",
        type=str,
        default="per_step,running_summary,advisory,cascade",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap trajectories; 0 means all")
    parser.add_argument("--k", type=int, default=10, help="depth for attack-step recall")
    parser.add_argument(
        "--mock",
        choices=("off", "constant", "keyword"),
        default="off",
        help="client mode: live, fixed-score mock, or prompt-dependent mock",
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore cached summaries")
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20",
        help="audit budgets to sweep for the safety curve",
    )
    args = parser.parse_args()

    configure_logging()
    seed = set_seed()

    trajectories = stratified_limit(load_corpus(args.corpus), args.limit, seed)
    by_id = {t.session_id: t for t in trajectories}
    malicious = [t for t in trajectories if t.is_malicious]
    benign = [t for t in trajectories if not t.is_malicious]
    logger.info("%d trajectories: %d malicious, %d benign", len(trajectories), len(malicious), len(benign))

    if not malicious:
        raise SystemExit("no malicious trajectories in the selection -- safety is undefined")
    if len(benign) < 2:
        raise SystemExit(
            f"need at least 2 benign trajectories for disjoint calibration/reporting splits, "
            f"got {len(benign)}"
        )

    calibration, reporting = split_benign(benign, seed)
    logger.info("benign split: %d calibration, %d reporting", len(calibration), len(reporting))

    summariser = Summariser(
        client=get_client(
            CONFIG.summariser_model,
            mock=args.mock != "off",
            mock_response='{"summary": "ran a command"}',
        )
    )
    summaries = summarise_corpus(trajectories, summariser, use_cache=not args.no_cache)

    protocol_names = [name.strip() for name in args.protocols.split(",") if name.strip()]
    budgets = [float(b) for b in args.budgets.split(",") if b.strip()]
    reports: list[ProtocolReport] = []
    plot_rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, float]]] = {}

    for protocol in build_protocols(protocol_names, mock=args.mock):
        runs: dict[str, ProtocolRun] = {}
        for trajectory in tqdm(
            trajectories, desc=protocol.name, unit="traj", dynamic_ncols=True
        ):
            ledger = build_ledger(
                trajectory.session_id, trajectory.steps, summaries[trajectory.session_id]
            )
            run = protocol.run(trajectory, ledger)
            save_run(run)
            runs[trajectory.session_id] = run

        calibration_runs = [runs[t.session_id] for t in calibration]
        reporting_runs = [runs[t.session_id] for t in reporting]

        plot_rows.extend(step_records(runs, by_id, protocol.name))
        curves[protocol.name] = safety_curve(
            malicious={t.session_id: runs[t.session_id] for t in malicious},
            trajectories=by_id,
            benign_calibration=calibration_runs,
            benign_report=reporting_runs,
            budgets=budgets,
        )

        reports.append(
            evaluate_protocol(
                protocol=protocol.name,
                malicious={t.session_id: runs[t.session_id] for t in malicious},
                trajectories=by_id,
                benign_calibration=calibration_runs,
                benign_report=reporting_runs,
                audit_budget=CONFIG.audit_budget,
                k=args.k,
            )
        )

    if plot_rows:
        import csv as _csv

        tidy_path = CONFIG.results_dir / "steps.csv"
        tidy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tidy_path, "w", encoding="utf-8", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(plot_rows[0].keys()))
            writer.writeheader()
            writer.writerows(plot_rows)
        logger.info("wrote %d step rows to %s", len(plot_rows), tidy_path)

    curve_path = CONFIG.results_dir / "safety_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    report_path = CONFIG.results_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(report) for report in reports], indent=2), encoding="utf-8"
    )

    print()
    print(f"audit budget {CONFIG.audit_budget:.3f}   seed {seed}   n_mal={len(malicious)} n_ben={len(benign)}")
    print("-" * 100)
    for report in reports:
        print(report.render())
    print("-" * 100)
    print(f"written: {report_path}")
    print(f"         {CONFIG.results_dir / 'safety_curve.json'}   (safety vs audit budget)")
    print(f"         {CONFIG.results_dir / 'steps.csv'}           (tidy per-step scores)")


if __name__ == "__main__":
    main()
