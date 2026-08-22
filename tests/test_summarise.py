"""Unit tests for per-step summarisation, focused on leakage and causality."""
from __future__ import annotations

import json

from ledgerctl.llm import MockLLM
from ledgerctl.summarise import Summariser, summary_prompt
from ledgerctl.trajectory import Trajectory
from tests.test_ledger import make_step


def test_prompt_contains_only_this_step() -> None:
    step = make_step(4, command="chmod 777 /etc/shadow", output="")
    rendered = json.dumps(summary_prompt(step))
    assert "chmod 777 /etc/shadow" in rendered
    assert "sess::s3" not in rendered
    assert "sess::s5" not in rendered


def test_prompt_never_carries_gold_label() -> None:
    step = make_step(0, command="whoami", output="root", attack=True)
    rendered = json.dumps(summary_prompt(step)).lower()
    assert "is_attack_step" not in rendered
    assert "attack" not in rendered
    assert "side task" not in rendered


def test_long_output_is_truncated_and_marked() -> None:
    step = make_step(0, output="x" * 10_000)
    content = summary_prompt(step)[1]["content"]
    assert "[output truncated]" in content
    assert len(content) < 10_000


def test_summariser_is_called_once_per_step_with_one_step_each() -> None:
    trajectory = Trajectory(session_id="sess", steps=tuple(make_step(i) for i in range(4)))
    client = MockLLM(response='{"summary": "did a thing"}')
    summaries = Summariser(client=client).summarise_trajectory(trajectory, progress=False)

    assert len(summaries) == 4
    assert len(client.calls) == 4
    for call in client.calls:
        assert len(call) == 2


def test_prose_response_is_salvaged_not_discarded() -> None:
    """An empty index entry would hide the step from the global monitor entirely."""
    trajectory = Trajectory(session_id="sess", steps=(make_step(0),))
    client = MockLLM(response="The command listed the directory.")
    summaries = Summariser(client=client).summarise_trajectory(trajectory, progress=False)
    assert summaries["sess::s0"] == "The command listed the directory."


def test_truncated_json_is_recovered() -> None:
    """A generation cut off by max_tokens has no closing brace but is still usable."""
    trajectory = Trajectory(session_id="sess", steps=(make_step(0),))
    client = MockLLM(response='{"summary": "read the deploy token and printed')
    summaries = Summariser(client=client).summarise_trajectory(trajectory, progress=False)
    assert summaries["sess::s0"] == "read the deploy token and printed"


def test_fenced_json_is_parsed() -> None:
    trajectory = Trajectory(session_id="sess", steps=(make_step(0),))
    client = MockLLM(response='```json\n{"summary": "listed files"}\n```')
    summaries = Summariser(client=client).summarise_trajectory(trajectory, progress=False)
    assert summaries["sess::s0"] == "listed files"


def test_empty_response_yields_empty_summary() -> None:
    trajectory = Trajectory(session_id="sess", steps=(make_step(0),))
    summaries = Summariser(client=MockLLM(response="")).summarise_trajectory(
        trajectory, progress=False
    )
    assert summaries["sess::s0"] == ""
