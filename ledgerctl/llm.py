"""LLM client abstraction with a deterministic mock for offline testing.

Every component takes a :class:`SupportsComplete` rather than constructing a
client, so the whole pipeline is runnable without a served endpoint.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from ledgerctl.config import CONFIG

logger = logging.getLogger(__name__)

Message = dict[str, str]

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@runtime_checkable
class SupportsComplete(Protocol):
    """Minimal chat-completion interface required by this package."""

    def complete(self, messages: Sequence[Message], max_tokens: int, temperature: float = 0.0) -> str:
        """Return the assistant message content for a chat completion."""


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object embedded in a model response.

    Args:
        text: Raw model output, optionally wrapped in prose or code fences.

    Returns:
        The decoded object.

    Raises:
        ValueError: If no parseable JSON object is present.
    """
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in response: {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def extract_string_field(text: str, key: str) -> str | None:
    """Best-effort recovery of one string field from possibly-truncated JSON.

    A generation cut off by its token limit leaves no closing brace, so strict
    parsing fails and the whole response is lost even though the field of
    interest is fully or partly present. This recovers what is there.

    Args:
        text: Raw model output.
        key: Field name to recover.

    Returns:
        The field value, or ``None`` if no such field is present at all.
    """
    try:
        value = extract_json(text).get(key)
        return str(value) if value is not None else None
    except ValueError:
        pass
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"(.*?)(?:"|$)', text, re.DOTALL)
    if match is None:
        return None
    # Undo the escaping a complete parse would have handled.
    return match.group(1).replace('\\"', '"').replace("\\n", "\n").strip()


def extract_number_field(text: str, key: str) -> float | None:
    """Best-effort recovery of one numeric field from possibly-truncated JSON.

    The global monitor is asked for a score plus a reason, an evidence list and a
    revision list. A generation cut off by its token limit loses the closing
    brace, so strict parsing discards a judgement whose score was already
    complete -- and a discarded judgement scores zero, which biases every
    protocol that uses this monitor downwards.

    Args:
        text: Raw model output.
        key: Field name to recover.

    Returns:
        The field value, or ``None`` if no such field is present.
    """
    try:
        value = extract_json(text).get(key)
        if value is not None:
            return float(value)
    except (ValueError, TypeError):
        pass
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


@dataclass
class OpenAICompatibleClient:
    """Chat client for any OpenAI-compatible endpoint (vLLM, TGI, hosted).

    Attributes:
        model: Model identifier passed to the endpoint.
        base_url: Endpoint root, including the ``/v1`` suffix.
        api_key: Credential; ``EMPTY`` for a local unauthenticated server.
    """

    model: str
    base_url: str = CONFIG.base_url
    api_key: str = CONFIG.api_key

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(self, messages: Sequence[Message], max_tokens: int, temperature: float = 0.0) -> str:
        """Issue one chat completion.

        Args:
            messages: Chat messages in OpenAI format.
            max_tokens: Generation ceiling.
            temperature: Sampling temperature; ``0.0`` for reproducibility.

        Returns:
            The assistant message content, or an empty string if the endpoint
            returned no content.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


@dataclass
class MockLLM:
    """Deterministic stand-in used by unit tests and ``--mock`` runs.

    Attributes:
        response: Fixed payload returned for every call.
        calls: Record of every request, for assertions about what the caller
            actually sent (used to verify causal masking).
    """

    response: str = '{"summary": "mock summary"}'
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], max_tokens: int, temperature: float = 0.0) -> str:
        """Record the request and return the fixed response."""
        self.calls.append([dict(m) for m in messages])
        return self.response


@dataclass
class KeywordMockLLM:
    """Mock client that scores on substring matches, for harness validation.

    A constant-scoring mock cannot produce a detection, because the threshold is
    calibrated from the same constant. This mock varies with the prompt, so it
    exercises calibration, catching, latency and recall without a served model.
    It is a test fixture, not a monitor: numbers produced with it say nothing
    about real attacks.

    Attributes:
        keywords: Substrings that raise the score when present in the prompt.
        hit_score: Score returned when any keyword is present.
        miss_score: Score returned otherwise.
        summary: Summary text returned for summarisation calls.
        calls: Record of every request.
    """

    keywords: tuple[str, ...] = ("token", "curl", "/tmp/.", "cp /var")
    hit_score: float = 8.0
    miss_score: float = 1.0
    summary: str = "ran a command"
    calls: list[list[Message]] = field(default_factory=list)

    def complete(self, messages: Sequence[Message], max_tokens: int, temperature: float = 0.0) -> str:
        """Return a JSON payload whose score depends on the prompt contents."""
        self.calls.append([dict(m) for m in messages])
        system = messages[0].get("content", "") if messages else ""
        body = " ".join(m.get("content", "") for m in messages[1:]).lower()

        if "recap" in system:
            return json.dumps({"summary": self.summary})
        if "choosing which earlier steps" in system:
            return json.dumps({"step_ids": []})

        score = self.hit_score if any(k in body for k in self.keywords) else self.miss_score
        payload: dict[str, Any] = {"score": score, "reason": "keyword mock"}
        if "running summary" in system:
            payload["summary"] = self.summary
        else:
            payload["evidence"] = []
            payload["revise"] = []
        return json.dumps(payload)


_LOCAL_CLIENTS: dict[str, SupportsComplete] = {}


def _local_client(model: str) -> SupportsComplete:
    """Return a process-wide singleton local client for ``model``.

    Weights are large and load slowly, so a model is loaded once and shared by
    every component that asks for it.

    Args:
        model: Model identifier.

    Returns:
        The shared client.
    """
    from ledgerctl.local_backend import TransformersClient

    if model not in _LOCAL_CLIENTS:
        _LOCAL_CLIENTS[model] = TransformersClient(model_name=model)
    return _LOCAL_CLIENTS[model]


def get_client(
    model: str,
    mock: bool = False,
    mock_response: str | None = None,
    base_url: str | None = None,
) -> SupportsComplete:
    """Construct a client for ``model``.

    Args:
        model: Model identifier.
        mock: When true, return a :class:`MockLLM` instead of a live client.
        mock_response: Optional fixed payload for the mock.
        base_url: Endpoint serving this model. Models of different sizes are
            served on separate ports, so this is routed per model rather than
            taken from a single global setting.

    Returns:
        A client satisfying :class:`SupportsComplete`.
    """
    if mock:
        return MockLLM(response=mock_response) if mock_response else MockLLM()
    if CONFIG.backend == "transformers":
        return _local_client(model)
    endpoint = base_url or CONFIG.base_url
    logger.info("constructing client for %s at %s", model, endpoint)
    return OpenAICompatibleClient(model=model, base_url=endpoint)
