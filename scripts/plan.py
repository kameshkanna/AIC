"""Estimate token cost and wall-clock for a sweep, from the actual corpus.

Cost is dominated by the global monitor's prefill, which grows with the index and
is therefore quadratic in trajectory length. That makes a hand estimate unreliable
once trajectory lengths are skewed, so this walks the real corpus step by step.

Throughput defaults are for a single H100-class GPU serving Qwen2.5-14B and Qwen2.5-7B
co-resident at bf16, with prefix caching on. They are estimates: run
``scripts.preflight`` on the box and pass the observed numbers for a real projection.

Usage:
    python -m scripts.plan --corpus data/trajectories.jsonl
    python -m scripts.plan --corpus data/trajectories.jsonl --no-prefix-caching
    python -m scripts.plan --protocols per_step,running_summary --budget-hours 12
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ledgerctl.config import CONFIG
from ledgerctl.runtime import configure_logging
from ledgerctl.trajectory import Trajectory, load_corpus

logger = logging.getLogger("plan")

CHARS_PER_TOKEN = 3.6
SUMMARY_PROMPT_OVERHEAD = 80
STEP_PROMPT_OVERHEAD = 130
GLOBAL_PROMPT_OVERHEAD = 220
SELECT_PROMPT_OVERHEAD = 120
ADVISORY_TOKENS = 120
FETCH_DETAIL_CHARS = 800
INDEX_ID_TOKENS = 10

STEP_OUT = 64
SUMMARY_OUT = CONFIG.summary_max_tokens
RUNNING_OUT = 150
GLOBAL_OUT = 200
SELECT_OUT = 60


@dataclass(frozen=True)
class Load:
    """Token load for one model class.

    Attributes:
        prefill: Input tokens, before any prefix-cache saving.
        cached_prefill: Input tokens actually computed once caching is applied.
        decode: Generated tokens.
        calls: Number of requests.
    """

    prefill: int = 0
    cached_prefill: int = 0
    decode: int = 0
    calls: int = 0

    def __add__(self, other: "Load") -> "Load":
        return Load(
            prefill=self.prefill + other.prefill,
            cached_prefill=self.cached_prefill + other.cached_prefill,
            decode=self.decode + other.decode,
            calls=self.calls + other.calls,
        )

    def hours(self, prefill_tps: float, decode_tps: float, prefix_caching: bool) -> float:
        """Wall-clock hours at the given throughputs."""
        prefill = self.cached_prefill if prefix_caching else self.prefill
        return (prefill / prefill_tps + self.decode / decode_tps) / 3600.0


def _tokens(text: str) -> int:
    """Approximate token count for a string."""
    return int(len(text) / CHARS_PER_TOKEN)


def _index_tokens(trajectory: Trajectory, upto: int, summary_tokens: int) -> int:
    """Tokens in the rendered index visible at step ``upto``."""
    return upto * (INDEX_ID_TOKENS + summary_tokens)


def summariser_load(trajectories: Sequence[Trajectory]) -> Load:
    """Token load of the one-time summariser pass."""
    prefill = 0
    calls = 0
    for trajectory in trajectories:
        for step in trajectory.steps:
            body = step.command + step.output[: int(FETCH_DETAIL_CHARS * 5)]
            prefill += SUMMARY_PROMPT_OVERHEAD + _tokens(body)
            calls += 1
    return Load(prefill=prefill, cached_prefill=prefill, decode=calls * SUMMARY_OUT, calls=calls)


def protocol_load(
    trajectories: Sequence[Trajectory], protocol: str, summary_tokens: int, max_fetches: int
) -> tuple[Load, Load]:
    """Token load of one protocol, split into small-model and large-model work.

    Args:
        trajectories: Corpus to cost.
        protocol: Protocol name.
        summary_tokens: Mean tokens per index line.
        max_fetches: Fetch cap per step, for the global monitor's record detail.

    Returns:
        The 7B load and the 32B load.

    Raises:
        ValueError: If the protocol is unknown.
    """
    small = Load()
    large = Load()
    fetch_tokens = max_fetches * int(FETCH_DETAIL_CHARS / CHARS_PER_TOKEN)

    for trajectory in trajectories:
        for step in trajectory.steps:
            command_tokens = _tokens(step.command)
            index_tokens = _index_tokens(trajectory, step.step, summary_tokens)
            # With prefix caching the index is a shared growing prefix, so only the
            # newly appended line is computed; the tail after it is not cacheable.
            new_index_tokens = INDEX_ID_TOKENS + summary_tokens

            if protocol == "per_step":
                p = STEP_PROMPT_OVERHEAD + command_tokens
                small += Load(prefill=p, cached_prefill=p, decode=STEP_OUT, calls=1)
            elif protocol == "scalar_posterior":
                p = STEP_PROMPT_OVERHEAD + command_tokens
                small += Load(prefill=p, cached_prefill=p, decode=STEP_OUT, calls=1)
            elif protocol == "running_summary":
                p = STEP_PROMPT_OVERHEAD + command_tokens + RUNNING_OUT
                small += Load(prefill=p, cached_prefill=p, decode=RUNNING_OUT, calls=1)
            elif protocol == "full_context":
                p = GLOBAL_PROMPT_OVERHEAD + index_tokens + command_tokens
                c = GLOBAL_PROMPT_OVERHEAD + new_index_tokens + command_tokens
                large += Load(prefill=p, cached_prefill=c, decode=GLOBAL_OUT, calls=1)
            elif protocol in ("advisory", "cascade"):
                select_p = SELECT_PROMPT_OVERHEAD + index_tokens + command_tokens
                select_c = SELECT_PROMPT_OVERHEAD + new_index_tokens + command_tokens
                large += Load(prefill=select_p, cached_prefill=select_c, decode=SELECT_OUT, calls=1)

                score_p = GLOBAL_PROMPT_OVERHEAD + index_tokens + fetch_tokens + command_tokens
                score_c = GLOBAL_PROMPT_OVERHEAD + new_index_tokens + fetch_tokens + command_tokens
                large += Load(prefill=score_p, cached_prefill=score_c, decode=GLOBAL_OUT, calls=1)

                step_p = STEP_PROMPT_OVERHEAD + command_tokens
                if protocol == "advisory":
                    step_p += ADVISORY_TOKENS
                small += Load(prefill=step_p, cached_prefill=step_p, decode=STEP_OUT, calls=1)
            else:
                raise ValueError(f"unknown protocol: {protocol}")

    return small, large


def main() -> None:
    """Print a per-protocol cost and wall-clock projection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CONFIG.data_dir / "trajectories.jsonl")
    parser.add_argument("--protocols", type=str, default="per_step,running_summary,advisory,cascade")
    parser.add_argument("--summary-tokens", type=int, default=25, help="mean tokens per index line")
    parser.add_argument("--small-prefill-tps", type=float, default=80_000.0)
    parser.add_argument("--small-decode-tps", type=float, default=6_000.0)
    parser.add_argument("--large-prefill-tps", type=float, default=40_000.0)
    parser.add_argument("--large-decode-tps", type=float, default=3_500.0)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--budget-hours", type=float, default=12.0)
    args = parser.parse_args()

    configure_logging()
    prefix_caching = not args.no_prefix_caching

    trajectories = load_corpus(args.corpus)
    lengths = np.array([len(t) for t in trajectories])
    n_mal = sum(1 for t in trajectories if t.is_malicious)

    print()
    print(f"corpus            {args.corpus}")
    print(f"trajectories      {len(trajectories)}  ({n_mal} malicious, {len(trajectories) - n_mal} benign)")
    print(f"steps             {int(lengths.sum())} total | mean {lengths.mean():.0f} "
          f"| p50 {np.percentile(lengths, 50):.0f} | p95 {np.percentile(lengths, 95):.0f} "
          f"| max {lengths.max()}")
    print(f"prefix caching    {'on' if prefix_caching else 'OFF'}")
    print()

    summ = summariser_load(trajectories)
    summ_hours = summ.hours(args.small_prefill_tps, args.small_decode_tps, prefix_caching=True)
    print(f"{'stage':<20} {'calls':>9} {'prefill':>13} {'decode':>11} {'hours':>7}")
    print("-" * 66)
    print(f"{'summarise (once)':<20} {summ.calls:>9,} {summ.cached_prefill:>13,} {summ.decode:>11,} {summ_hours:>7.2f}")

    total_hours = summ_hours
    for name in [p.strip() for p in args.protocols.split(",") if p.strip()]:
        small, large = protocol_load(
            trajectories, name, args.summary_tokens, CONFIG.max_fetches_per_step
        )
        hours = small.hours(
            args.small_prefill_tps, args.small_decode_tps, prefix_caching
        ) + large.hours(args.large_prefill_tps, args.large_decode_tps, prefix_caching)
        total_hours += hours
        prefill = (small.cached_prefill + large.cached_prefill) if prefix_caching else (small.prefill + large.prefill)
        print(
            f"{name:<20} {small.calls + large.calls:>9,} {prefill:>13,} "
            f"{small.decode + large.decode:>11,} {hours:>7.2f}"
        )

    print("-" * 66)
    print(f"{'TOTAL':<20} {'':>9} {'':>13} {'':>11} {total_hours:>7.2f}")
    print()
    headroom = args.budget_hours - total_hours
    verdict = "FITS" if headroom > 0 else "OVER BUDGET"
    print(f"budget {args.budget_hours:.1f} h -> {verdict}, headroom {headroom:+.2f} h")
    if headroom < 0:
        max_traj = int(len(trajectories) * args.budget_hours / total_hours)
        print(f"  fits at roughly --limit {max_traj} trajectories at this length distribution")
    print()


if __name__ == "__main__":
    main()
