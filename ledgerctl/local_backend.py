"""In-process transformers backend, for when a served endpoint is unavailable.

vLLM is the fast path, but it drags in a large dependency surface that can fail
on aarch64 for reasons unrelated to this project. This backend needs only
``torch`` and ``transformers`` and runs the model in the same process, so there
is no server, no port and no container.

Generating one prompt at a time would be far too slow -- decode is
memory-bandwidth-bound, so a batch of one wastes almost all of the card. Requests
from concurrent callers are therefore collected by a background worker and run as
a batch, which keeps the :class:`~ledgerctl.llm.SupportsComplete` interface
unchanged while giving throughput close to what the hardware allows. Callers just
need to be threaded, which the runner arranges with ``--workers``.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ledgerctl.config import CONFIG
from ledgerctl.llm import Message

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    """One pending generation, with a slot for its result."""

    messages: list[Message]
    max_tokens: int
    temperature: float
    done: threading.Event = field(default_factory=threading.Event)
    result: str = ""
    error: BaseException | None = None


@dataclass
class TransformersClient:
    """Chat client backed by a locally loaded transformers model.

    Attributes:
        model_name: Hugging Face model id.
        device: Torch device string.
        dtype: Torch dtype name used to load weights.
        max_batch_size: Most requests coalesced into one generate call. A batch
            that does not fit is split and retried, so this is a ceiling rather
            than a promise.
        max_prompt_tokens: Prompts longer than this are truncated. Without an
            explicit bound the tokenizer falls back to the model maximum, which
            for these models is large enough to exhaust VRAM on a single batch.
        batch_timeout_s: How long the worker waits for a batch to fill before
            running what it has. Small values favour latency, larger ones
            throughput.
    """

    model_name: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_batch_size: int = CONFIG.local_max_batch_size
    max_prompt_tokens: int = CONFIG.local_max_prompt_tokens
    batch_timeout_s: float = 0.05

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA is not available; set device='cpu' to run anyway")

        logger.info("loading %s onto %s (%s)", self.model_name, self.device, self.dtype)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # transformers renamed `torch_dtype` to `dtype` in 4.56. Supporting both
        # keeps this working across the range the project declares.
        torch_dtype = getattr(torch, self.dtype)
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch_dtype, device_map=self.device
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch_dtype, device_map=self.device
            )
        self._model.eval()

        self._stopping = threading.Event()
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        logger.info("%s ready", self.model_name)

    def complete(self, messages: Sequence[Message], max_tokens: int, temperature: float = 0.0) -> str:
        """Queue one generation and block until the batch containing it finishes.

        Args:
            messages: Chat messages in OpenAI format.
            max_tokens: Generation ceiling.
            temperature: Sampling temperature; ``0.0`` generates greedily.

        Returns:
            The generated assistant text.

        Raises:
            RuntimeError: If generation failed for this request.
        """
        request = _Request(
            messages=[dict(m) for m in messages], max_tokens=max_tokens, temperature=temperature
        )
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise RuntimeError(f"generation failed: {request.error}") from request.error
        return request.result

    def _collect_batch(self) -> list[_Request]:
        """Block for one request, then drain up to ``max_batch_size`` more.

        Returns:
            The collected requests, or an empty list once shutdown is requested.
        """
        first = self._queue.get()
        if first is None:
            self._stopping.set()
            return []

        batch = [first]
        deadline = time.monotonic() + self.batch_timeout_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is None:
                # Remember the shutdown request instead of dropping it, or the
                # worker would block forever on the next get and close() would
                # sit out its join timeout.
                self._stopping.set()
                break
            batch.append(item)
        return batch

    def _run_worker(self) -> None:
        """Consume the queue, generating one batch at a time."""
        while True:
            batch = self._collect_batch()
            if not batch:
                return
            try:
                self._generate_batch(batch)
            except Exception as exc:  # noqa: BLE001 - surfaced to every waiter
                logger.exception("batch generation failed")
                for request in batch:
                    request.error = exc
            finally:
                for request in batch:
                    request.done.set()
            if self._stopping.is_set():
                return

    def _generate_batch(self, batch: list[_Request]) -> None:
        """Generate for a whole batch and write each result back to its request.

        Requests are grouped by generation settings, since a single ``generate``
        call applies one configuration to every row.

        Args:
            batch: Pending requests.
        """
        groups: dict[tuple[int, float], list[_Request]] = {}
        for request in batch:
            groups.setdefault((request.max_tokens, request.temperature), []).append(request)

        for (max_tokens, temperature), group in groups.items():
            self._generate_group(group, max_tokens, temperature)

    def _generate_group(self, group: list[_Request], max_tokens: int, temperature: float) -> None:
        """Generate for one uniformly-configured group, halving on OOM.

        A batch that is too large for the remaining VRAM raises rather than
        degrading, and failing every request in it would abort a sweep hours in.
        Splitting and retrying costs one wasted attempt and keeps the run alive.

        Args:
            group: Requests sharing generation settings.
            max_tokens: Generation ceiling.
            temperature: Sampling temperature.
        """
        prompts = [
            self._tokenizer.apply_chat_template(
                r.messages, tokenize=False, add_generation_prompt=True
            )
            for r in group
        ]
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
        ).to(self._model.device)

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)

        try:
            with self._torch.inference_mode():
                generated = self._model.generate(**encoded, **kwargs)
        except self._torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            half = len(group) // 2
            logger.warning("OOM at batch %d; splitting", len(group))
            self._torch.cuda.empty_cache()
            self._generate_group(group[:half], max_tokens, temperature)
            self._generate_group(group[half:], max_tokens, temperature)
            return

        prompt_length = encoded["input_ids"].shape[1]
        for request, row in zip(group, generated, strict=True):
            request.result = self._tokenizer.decode(row[prompt_length:], skip_special_tokens=True)

    def close(self) -> None:
        """Stop the background worker, finishing any in-flight batch first."""
        self._queue.put(None)
        self._worker.join(timeout=60)
        if self._worker.is_alive():
            logger.warning("generation worker did not stop within 60s")
