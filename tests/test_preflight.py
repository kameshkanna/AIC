"""Unit tests for the preflight memory arithmetic."""
from __future__ import annotations

import pytest

from scripts.preflight import GEOMETRY, ModelGeometry, check_memory

SMALL = "Qwen/Qwen2.5-7B-Instruct"
MID = "Qwen/Qwen2.5-14B-Instruct"
LARGE = "Qwen/Qwen2.5-32B-Instruct"


def test_weight_footprint_matches_bf16_parameter_count() -> None:
    assert GEOMETRY[LARGE].weights_gib == pytest.approx(61.1, abs=0.5)
    assert GEOMETRY[SMALL].weights_gib == pytest.approx(14.2, abs=0.5)


def test_kv_cost_scales_with_layers_and_kv_heads() -> None:
    assert GEOMETRY[SMALL].kv_kib_per_token == pytest.approx(56.0)
    assert GEOMETRY[MID].kv_kib_per_token == pytest.approx(192.0)
    assert GEOMETRY[LARGE].kv_kib_per_token == pytest.approx(256.0)


def test_seven_and_fourteen_fit_together() -> None:
    assert check_memory([SMALL, MID], gpu_gib=96.0, max_model_len=16384)


def test_seven_and_thirtytwo_are_rejected_as_thrashing() -> None:
    assert not check_memory([SMALL, LARGE], gpu_gib=96.0, max_model_len=16384)


def test_thirtytwo_alone_fits() -> None:
    assert check_memory([LARGE], gpu_gib=96.0, max_model_len=16384)


def test_oversized_configuration_is_rejected() -> None:
    assert not check_memory([LARGE, LARGE], gpu_gib=96.0, max_model_len=16384)


def test_longer_context_reduces_concurrency() -> None:
    assert check_memory([SMALL, MID], gpu_gib=96.0, max_model_len=8192)
    assert not check_memory([SMALL, MID], gpu_gib=96.0, max_model_len=131072)


def test_unknown_model_is_reported_rather_than_guessed() -> None:
    assert not check_memory(["some/unknown-model"], gpu_gib=96.0, max_model_len=8192)


def test_geometry_is_self_consistent() -> None:
    geometry = ModelGeometry(params_b=1.0, layers=1, kv_heads=1, head_dim=128)
    assert geometry.kv_kib_per_token == pytest.approx(2 * 1 * 1 * 128 * 2 / 1024)
