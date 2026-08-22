"""Unit tests for the append-only ledger."""
from __future__ import annotations

from pathlib import Path

import pytest

from ledgerctl.ledger import Ledger, build_ledger
from ledgerctl.trajectory import Step


def make_step(
    index: int,
    command: str = "ls",
    output: str = "",
    attack: bool = False,
    session: str = "sess",
) -> Step:
    """Build a synthetic step for testing."""
    return Step(
        session_id=session,
        step=index,
        step_id=f"{session}::s{index}",
        timestamp="",
        command=command,
        output=output,
        is_attack_step=attack,
    )


def test_append_rejects_duplicate_ids() -> None:
    ledger = Ledger(session_id="sess")
    ledger.append(make_step(0), "first")
    with pytest.raises(ValueError, match="duplicate step id"):
        ledger.append(make_step(0), "again")


def test_record_excludes_gold_label() -> None:
    ledger = Ledger(session_id="sess")
    ledger.append(make_step(0, attack=True), "s")
    assert "is_attack_step" not in ledger.records["sess::s0"]


def test_render_index_respects_causal_mask() -> None:
    ledger = build_ledger("sess", [make_step(i) for i in range(5)], {})
    rendered = ledger.render_index(upto=3)
    assert "sess::s2" in rendered
    assert "sess::s3" not in rendered


def test_fetch_masks_future_steps() -> None:
    ledger = build_ledger("sess", [make_step(i) for i in range(5)], {f"sess::s{i}": "x" for i in range(5)})
    fetched = ledger.fetch(["sess::s1", "sess::s4"], upto=3)
    assert fetched[0]["step_id"] == "sess::s1"
    assert fetched[1]["error"] == "not yet observed"


def test_fetch_unknown_id_degrades() -> None:
    ledger = build_ledger("sess", [make_step(0)], {})
    assert ledger.fetch(["sess::s99"])[0]["error"] == "not found"


def test_accesses_are_logged() -> None:
    ledger = build_ledger("sess", [make_step(0)], {"sess::s0": "touched /etc/passwd"})
    ledger.grep_index("passwd")
    ledger.fetch(["sess::s0"])
    assert [entry["op"] for entry in ledger.access_log] == ["grep_index", "fetch"]


def test_grep_falls_back_to_literal_on_bad_pattern() -> None:
    ledger = build_ledger("sess", [make_step(0)], {"sess::s0": "wrote a[b"})
    assert len(ledger.grep_index("a[b")) == 1


def test_revision_overrides_forward_score() -> None:
    ledger = build_ledger("sess", [make_step(i) for i in range(3)], {})
    base = {f"sess::s{i}": 0.1 for i in range(3)}
    ledger.revise("sess::s0", 0.9, at_step=2, reason="linked to later egress")
    assert ledger.effective_scores(base)["sess::s0"] == 0.9
    assert base["sess::s0"] == 0.1


def test_revision_validates_inputs() -> None:
    ledger = build_ledger("sess", [make_step(0)], {})
    with pytest.raises(KeyError):
        ledger.revise("sess::s9", 0.5, at_step=1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ledger.revise("sess::s0", 1.5, at_step=1)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ledger = build_ledger("sess", [make_step(i) for i in range(3)], {f"sess::s{i}": f"sum{i}" for i in range(3)})
    ledger.revise("sess::s0", 0.8, at_step=2)
    reloaded = Ledger.load(ledger.save(tmp_path))
    assert reloaded.render_index() == ledger.render_index()
    assert reloaded.revisions == ledger.revisions
