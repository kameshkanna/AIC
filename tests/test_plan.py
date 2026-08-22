"""Unit tests for the cost planner."""
from __future__ import annotations

import pytest

from ledgerctl.trajectory import Trajectory
from scripts.plan import Load, protocol_load, summariser_load
from tests.test_ledger import make_step


def corpus(n_steps: int, n_traj: int = 1) -> list[Trajectory]:
    """Build a synthetic corpus of fixed-length trajectories."""
    return [
        Trajectory(
            session_id=f"t{j}",
            steps=tuple(
                make_step(i, command="cat /var/lib/svc/token", output="x" * 200, session=f"t{j}")
                for i in range(n_steps)
            ),
        )
        for j in range(n_traj)
    ]


def test_load_addition_accumulates_every_field() -> None:
    total = Load(prefill=1, cached_prefill=2, decode=3, calls=4) + Load(
        prefill=10, cached_prefill=20, decode=30, calls=40
    )
    assert (total.prefill, total.cached_prefill, total.decode, total.calls) == (11, 22, 33, 44)


def test_hours_uses_cached_prefill_only_when_caching_is_on() -> None:
    load = Load(prefill=3600, cached_prefill=360, decode=0, calls=1)
    uncached = load.hours(prefill_tps=1.0, decode_tps=1.0, prefix_caching=False)
    cached = load.hours(prefill_tps=1.0, decode_tps=1.0, prefix_caching=True)
    assert uncached == pytest.approx(1.0)
    assert cached == pytest.approx(0.1)


def test_summariser_makes_one_call_per_step() -> None:
    assert summariser_load(corpus(n_steps=12, n_traj=3)).calls == 36


def test_per_step_places_no_load_on_the_large_model() -> None:
    small, large = protocol_load(corpus(20), "per_step", summary_tokens=25, max_fetches=4)
    assert small.calls == 20
    assert large.calls == 0


def test_arms_make_two_large_calls_and_one_small_call_per_step() -> None:
    small, large = protocol_load(corpus(20), "cascade", summary_tokens=25, max_fetches=4)
    assert small.calls == 20
    assert large.calls == 40


def test_uncached_prefill_is_quadratic_in_trajectory_length() -> None:
    _, short = protocol_load(corpus(50), "cascade", summary_tokens=25, max_fetches=4)
    _, long = protocol_load(corpus(100), "cascade", summary_tokens=25, max_fetches=4)
    ratio = long.prefill / short.prefill
    assert 3.0 < ratio < 4.5


def test_cached_prefill_is_linear_in_trajectory_length() -> None:
    _, short = protocol_load(corpus(50), "cascade", summary_tokens=25, max_fetches=4)
    _, long = protocol_load(corpus(100), "cascade", summary_tokens=25, max_fetches=4)
    ratio = long.cached_prefill / short.cached_prefill
    assert 1.8 < ratio < 2.2


def test_advisory_costs_more_small_model_prefill_than_cascade() -> None:
    advisory, _ = protocol_load(corpus(30), "advisory", summary_tokens=25, max_fetches=4)
    cascade, _ = protocol_load(corpus(30), "cascade", summary_tokens=25, max_fetches=4)
    assert advisory.prefill > cascade.prefill


def test_unknown_protocol_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol_load(corpus(5), "nope", summary_tokens=25, max_fetches=4)
