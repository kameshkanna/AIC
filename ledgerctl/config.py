"""Centralised, environment-driven configuration.

A single frozen :class:`Config` instance (:data:`CONFIG`) is constructed at import
time from the process environment so that every component -- loaders, summariser,
monitors, evaluation -- reads identical settings and no module reaches for
``os.environ`` on its own.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    """Read a string setting from the environment."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Read an integer setting, failing fast on unparseable values."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    """Read a float setting, failing fast on unparseable values."""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    """Immutable run configuration.

    Attributes:
        backend: ``openai`` for a served endpoint, ``transformers`` to load
            models in-process with no server.
        seed: Global random seed for every stochastic component.
        data_dir: Root directory holding input trajectory corpora.
        results_dir: Root directory for ledgers, indices and evaluation output.
        base_url: Endpoint serving the global monitor.
        small_base_url: Endpoint serving the per-step monitor and summariser.
        baseline_base_url: Endpoint serving the memoryless baseline monitor.
        api_key: Endpoint credential; ``EMPTY`` for a local vLLM server.
        summariser_model: Model that writes the per-step summary.
        step_monitor_model: Per-step monitor (hot path).
        global_monitor_model: Global monitor reading the index (off hot path).
        baseline_monitor_model: Larger memoryless monitor used only by the
            full-context baseline, served alone in its own pass.
        summary_max_tokens: Hard ceiling on a single step summary.
        max_fetches_per_step: Cap on full-record fetches per monitored step.
        local_max_batch_size: Ceiling on requests coalesced into one
            in-process generate call.
        local_max_prompt_tokens: Prompt truncation bound for the in-process
            backend.
        audit_budget: Step-wise false-positive rate used to calibrate the
            decision threshold on held-out benign trajectories.
    """

    backend: str
    seed: int
    data_dir: Path
    results_dir: Path
    base_url: str
    small_base_url: str
    baseline_base_url: str
    api_key: str
    summariser_model: str
    step_monitor_model: str
    global_monitor_model: str
    baseline_monitor_model: str
    summary_max_tokens: int
    max_fetches_per_step: int
    local_max_batch_size: int
    local_max_prompt_tokens: int
    audit_budget: float

    @classmethod
    def from_env(cls) -> "Config":
        """Build a configuration from the current environment."""
        return cls(
            backend=_env_str("LEDGERCTL_BACKEND", "openai"),
            seed=_env_int("LEDGERCTL_SEED", 0),
            data_dir=Path(_env_str("LEDGERCTL_DATA_DIR", "./data")).resolve(),
            results_dir=Path(_env_str("LEDGERCTL_RESULTS_DIR", "./results")).resolve(),
            base_url=_env_str("LEDGERCTL_BASE_URL", "http://localhost:8000/v1"),
            small_base_url=_env_str("LEDGERCTL_SMALL_BASE_URL", "http://localhost:8001/v1"),
            baseline_base_url=_env_str("LEDGERCTL_BASELINE_BASE_URL", "http://localhost:8002/v1"),
            api_key=_env_str("LEDGERCTL_API_KEY", "EMPTY"),
            summariser_model=_env_str("LEDGERCTL_SUMMARISER_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            step_monitor_model=_env_str("LEDGERCTL_STEP_MONITOR_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            global_monitor_model=_env_str("LEDGERCTL_GLOBAL_MONITOR_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
            baseline_monitor_model=_env_str("LEDGERCTL_BASELINE_MONITOR_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
            summary_max_tokens=_env_int("LEDGERCTL_SUMMARY_MAX_TOKENS", 64),
            max_fetches_per_step=_env_int("LEDGERCTL_MAX_FETCHES_PER_STEP", 4),
            local_max_batch_size=_env_int("LEDGERCTL_LOCAL_MAX_BATCH", 16),
            local_max_prompt_tokens=_env_int("LEDGERCTL_LOCAL_MAX_PROMPT_TOKENS", 8192),
            audit_budget=_env_float("LEDGERCTL_AUDIT_BUDGET", 0.01),
        )


CONFIG = Config.from_env()
