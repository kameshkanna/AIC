"""Unit tests for the orchestrator helpers and the validation mock."""
from __future__ import annotations

import json

import pytest

from ledgerctl.llm import KeywordMockLLM
from ledgerctl.trajectory import Trajectory
from scripts.run import build_protocols, split_benign, stratified_limit
from tests.test_ledger import make_step


def benign(n: int) -> list[Trajectory]:
    """Build ``n`` benign trajectories."""
    return [
        Trajectory(session_id=f"b{i}", steps=(make_step(0, session=f"b{i}"),)) for i in range(n)
    ]


def test_benign_split_is_disjoint_and_covering() -> None:
    calibration, reporting = split_benign(benign(11), seed=0)
    calibration_ids = {t.session_id for t in calibration}
    reporting_ids = {t.session_id for t in reporting}
    assert not calibration_ids & reporting_ids
    assert len(calibration_ids | reporting_ids) == 11


def test_benign_split_is_seed_stable() -> None:
    first = [t.session_id for t in split_benign(benign(10), seed=7)[0]]
    second = [t.session_id for t in split_benign(benign(10), seed=7)[0]]
    assert first == second


def test_benign_split_needs_two_trajectories() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        split_benign(benign(1), seed=0)


def test_unknown_protocol_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        build_protocols(["not_a_protocol"], mock="keyword")


def test_all_protocols_build_under_keyword_mock() -> None:
    names = [
        "per_step",
        "running_summary",
        "advisory",
        "cascade",
        "scalar_posterior",
        "full_context",
    ]
    assert [p.name for p in build_protocols(names, mock="keyword")] == names


def test_keyword_mock_varies_with_prompt() -> None:
    client = KeywordMockLLM()
    system = [{"role": "system", "content": "You are a security monitor reviewing one command"}]
    hit = json.loads(client.complete([*system, {"role": "user", "content": "cat /var/lib/svc/deploy.token"}], 64))
    miss = json.loads(client.complete([*system, {"role": "user", "content": "uptime"}], 64))
    assert hit["score"] > miss["score"]


def test_keyword_mock_returns_summary_for_summarisation_prompts() -> None:
    client = KeywordMockLLM()
    system = [{"role": "system", "content": "You write one factual recap of a single step"}]
    payload = json.loads(client.complete([*system, {"role": "user", "content": "ls"}], 64))
    assert "summary" in payload
    assert "score" not in payload


def mixed(n_benign: int, n_malicious: int) -> list[Trajectory]:
    """Build a corpus with benign trajectories first, then malicious ones."""
    out = [
        Trajectory(session_id=f"b{i}", steps=(make_step(0, session=f"b{i}"),))
        for i in range(n_benign)
    ]
    out += [
        Trajectory(
            session_id=f"m{i}",
            steps=(make_step(0, attack=True, session=f"m{i}"),),
            is_malicious=True,
            attack_completion_step=0,
        )
        for i in range(n_malicious)
    ]
    return out


def test_stratified_limit_keeps_both_classes() -> None:
    subset = stratified_limit(mixed(100, 100), limit=40, seed=0)
    assert any(t.is_malicious for t in subset)
    assert sum(1 for t in subset if not t.is_malicious) >= 2


def test_stratified_limit_preserves_ratio() -> None:
    subset = stratified_limit(mixed(150, 50), limit=40, seed=0)
    fraction = sum(1 for t in subset if t.is_malicious) / len(subset)
    assert 0.15 < fraction < 0.35


def test_stratified_limit_is_a_noop_when_limit_covers_corpus() -> None:
    corpus = mixed(5, 5)
    assert len(stratified_limit(corpus, limit=0, seed=0)) == 10
    assert len(stratified_limit(corpus, limit=99, seed=0)) == 10


def test_stratified_limit_is_seed_stable() -> None:
    first = [t.session_id for t in stratified_limit(mixed(50, 50), 20, seed=3)]
    second = [t.session_id for t in stratified_limit(mixed(50, 50), 20, seed=3)]
    assert first == second


def test_head_slice_would_have_dropped_every_attack() -> None:
    """Regression guard for the bug the dry run surfaced."""
    corpus = mixed(100, 100)
    assert not any(t.is_malicious for t in corpus[:40])
    assert any(t.is_malicious for t in stratified_limit(corpus, 40, seed=0))
