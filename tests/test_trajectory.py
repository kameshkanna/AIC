"""Unit tests for trajectory parsing and the causal prefix."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ledgerctl.trajectory import Trajectory, load_corpus, parse_trajectory
from tests.test_ledger import make_step


def test_parses_aliased_field_names() -> None:
    raw = {
        "id": "env12_run3",
        "transcript": [
            {"turn": 0, "action": "whoami", "observation": "root"},
            {"turn": 1, "tool_call": "id", "result": "uid=0"},
        ],
    }
    trajectory = parse_trajectory(raw)
    assert trajectory.session_id == "env12_run3"
    assert trajectory.steps[0].command == "whoami"
    assert trajectory.steps[1].output == "uid=0"
    assert trajectory.steps[1].step_id == "env12_run3::s1"


def test_missing_step_list_fails_fast() -> None:
    with pytest.raises(ValueError, match="no step list"):
        parse_trajectory({"id": "x", "notes": "nothing here"})


def test_completion_step_defaults_to_last_attack_step() -> None:
    raw = {
        "id": "run",
        "malicious": True,
        "steps": [
            {"command": "a", "output": "", "is_attack_step": False},
            {"command": "b", "output": "", "is_attack_step": True},
            {"command": "c", "output": "", "is_attack_step": False},
        ],
    }
    assert parse_trajectory(raw).attack_completion_step == 1


def test_completion_step_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        Trajectory(
            session_id="s",
            steps=(make_step(0),),
            is_malicious=True,
            attack_completion_step=7,
        )


def test_malicious_requires_completion_step() -> None:
    with pytest.raises(ValueError, match="attack_completion_step"):
        Trajectory(session_id="s", steps=(make_step(0),), is_malicious=True)


def test_prefix_is_strictly_before() -> None:
    trajectory = Trajectory(session_id="s", steps=tuple(make_step(i) for i in range(5)))
    assert [s.step for s in trajectory.prefix(3)] == [0, 1, 2]


def test_attack_step_ids_are_exposed_for_scoring() -> None:
    trajectory = Trajectory(
        session_id="sess",
        steps=(make_step(0), make_step(1, attack=True), make_step(2, attack=True)),
        is_malicious=True,
        attack_completion_step=2,
    )
    assert trajectory.attack_step_ids == ("sess::s1", "sess::s2")


def test_load_corpus_reports_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"id": "ok", "steps": [{"command": "ls", "output": ""}]}) + "\n"
        + json.dumps({"id": "bad"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"corpus\.jsonl:2"):
        load_corpus(path)
