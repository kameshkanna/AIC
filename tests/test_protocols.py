"""Unit tests for monitors and protocols."""
from __future__ import annotations

import json

import pytest

from ledgerctl.ledger import build_ledger
from ledgerctl.llm import MockLLM
from ledgerctl.monitor import GlobalMonitor, MonitorResult, StepMonitor, clamp_score, render_advisory
from ledgerctl.protocols import (
    AdvisoryArm,
    CascadeArm,
    FullContextBaseline,
    PerStepBaseline,
    RunningSummaryBaseline,
    ScalarPosteriorBaseline,
)
from ledgerctl.trajectory import Trajectory
from tests.test_ledger import make_step


def trajectory(n: int = 4) -> Trajectory:
    """Build a synthetic trajectory of ``n`` steps."""
    return Trajectory(session_id="sess", steps=tuple(make_step(i) for i in range(n)))


def ledger_for(traj: Trajectory):
    """Build a ledger with one summary per step."""
    return build_ledger(traj.session_id, traj.steps, {s.step_id: f"sum{s.step}" for s in traj.steps})


def test_clamp_score_bounds_and_degrades() -> None:
    assert clamp_score(11) == 10.0
    assert clamp_score(-3) == 0.0
    assert clamp_score("not a number") == 0.0
    assert clamp_score(None) == 0.0


def test_step_monitor_withholds_output_of_incoming_step() -> None:
    client = MockLLM(response='{"score": 4, "reason": "ok"}')
    step = make_step(0, command="cat /etc/shadow", output="root:x:0:0")
    StepMonitor(client=client).score(step)
    rendered = json.dumps(client.calls[0])
    assert "cat /etc/shadow" in rendered
    assert "root:x:0:0" not in rendered


def test_step_monitor_malformed_response_is_flagged() -> None:
    result = StepMonitor(client=MockLLM(response="no json here")).score(make_step(0))
    assert result.malformed
    assert result.score == 0.0


def test_global_monitor_respects_fetch_cap() -> None:
    traj = trajectory(8)
    ledger = ledger_for(traj)
    ids = [f"sess::s{i}" for i in range(6)]
    client = MockLLM(response=json.dumps({"step_ids": ids, "score": 5, "reason": "r"}))
    monitor = GlobalMonitor(client=client, max_fetches=2)
    assert len(monitor.select(traj.steps[7], ledger)) == 2


def test_global_monitor_zero_budget_never_selects() -> None:
    traj = trajectory(4)
    client = MockLLM(response='{"step_ids": ["sess::s0"], "score": 5}')
    monitor = GlobalMonitor(client=client, max_fetches=0)
    assert monitor.select(traj.steps[3], ledger_for(traj)) == []


def test_global_monitor_rejects_bad_index_mode() -> None:
    with pytest.raises(ValueError, match="index_mode"):
        GlobalMonitor(client=MockLLM(), index_mode="semantic")


def test_global_monitor_parses_revisions() -> None:
    traj = trajectory(4)
    payload = {"score": 8, "reason": "r", "evidence": ["sess::s1"], "revise": [{"step_id": "sess::s1", "score": 9}]}
    monitor = GlobalMonitor(client=MockLLM(response=json.dumps(payload)), max_fetches=0)
    result = monitor.score(traj.steps[3], ledger_for(traj))
    assert result.revisions == (("sess::s1", 9.0),)
    assert result.evidence_ids == ("sess::s1",)


def test_advisory_renders_ids_and_summaries_only() -> None:
    traj = trajectory(4)
    ledger = build_ledger(
        "sess", traj.steps, {s.step_id: "touched the token file" for s in traj.steps}
    )
    result = MonitorResult(step_id="sess::s3", score=5.0, evidence_ids=("sess::s1",))
    advisory = render_advisory(result, ledger, upto=3)
    assert "sess::s1" in advisory
    assert "touched the token file" in advisory


def test_advisory_excludes_future_steps() -> None:
    traj = trajectory(4)
    result = MonitorResult(step_id="sess::s1", score=5.0, evidence_ids=("sess::s3",))
    assert render_advisory(result, ledger_for(traj), upto=1) == ""


def test_per_step_baseline_scores_every_step() -> None:
    traj = trajectory(5)
    protocol = PerStepBaseline(step_monitor=StepMonitor(client=MockLLM(response='{"score": 3}')))
    run = protocol.run(traj, ledger_for(traj))
    assert len(run.scores) == 5
    assert set(run.scores.values()) == {3.0}
    assert run.revised == run.scores


def test_scalar_posterior_is_monotone_under_constant_evidence() -> None:
    traj = trajectory(6)
    protocol = ScalarPosteriorBaseline(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 8}'))
    )
    run = protocol.run(traj, ledger_for(traj))
    series = [run.scores[s.step_id] for s in traj.steps]
    assert series == sorted(series)
    assert series[-1] > series[0]


def test_scalar_posterior_rejects_bad_prior() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScalarPosteriorBaseline(step_monitor=StepMonitor(client=MockLLM()), prior_alpha=0.0)


def test_running_summary_carries_state_forward() -> None:
    traj = trajectory(3)
    client = MockLLM(response='{"score": 5, "summary": "agent touched the token"}')
    RunningSummaryBaseline(client=client).run(traj, ledger_for(traj))
    second_call = json.dumps(client.calls[1])
    assert "agent touched the token" in second_call


def test_full_context_requires_zero_fetch_budget() -> None:
    with pytest.raises(ValueError, match="max_fetches=0"):
        FullContextBaseline(global_monitor=GlobalMonitor(client=MockLLM(), max_fetches=3))


def test_advisory_arm_score_comes_from_step_monitor() -> None:
    traj = trajectory(3)
    protocol = AdvisoryArm(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 2}')),
        global_monitor=GlobalMonitor(client=MockLLM(response='{"score": 9}'), max_fetches=0),
    )
    run = protocol.run(traj, ledger_for(traj))
    assert set(run.scores.values()) == {2.0}


def test_cascade_arm_score_comes_from_global_monitor() -> None:
    traj = trajectory(3)
    protocol = CascadeArm(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 2}')),
        global_monitor=GlobalMonitor(client=MockLLM(response='{"score": 9}'), max_fetches=0),
    )
    run = protocol.run(traj, ledger_for(traj))
    assert set(run.scores.values()) == {9.0}


def test_cascade_passes_local_score_to_global_monitor() -> None:
    traj = trajectory(2)
    global_client = MockLLM(response='{"score": 6}')
    CascadeArm(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 7}')),
        global_monitor=GlobalMonitor(client=global_client, max_fetches=0),
    ).run(traj, ledger_for(traj))
    assert "per-step monitor score: 7.0/10" in json.dumps(global_client.calls[0])


def test_revisions_are_applied_to_revised_scores_only() -> None:
    traj = trajectory(4)
    payload = {"score": 3, "revise": [{"step_id": "sess::s0", "score": 10}]}
    protocol = CascadeArm(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 1}')),
        global_monitor=GlobalMonitor(client=MockLLM(response=json.dumps(payload)), max_fetches=0),
    )
    run = protocol.run(traj, ledger_for(traj))
    assert run.scores["sess::s0"] == 3.0
    assert run.revised["sess::s0"] == 10.0


def test_revisions_naming_future_steps_are_dropped() -> None:
    traj = trajectory(3)
    payload = {"score": 3, "revise": [{"step_id": "sess::s2", "score": 10}]}
    protocol = CascadeArm(
        step_monitor=StepMonitor(client=MockLLM(response='{"score": 1}')),
        global_monitor=GlobalMonitor(client=MockLLM(response=json.dumps(payload)), max_fetches=0),
    )
    ledger = ledger_for(traj)
    protocol.run(traj, ledger)
    assert all(revision["step_id"] != "sess::s2" or revision["at_step"] > 2 for revision in ledger.revisions)
