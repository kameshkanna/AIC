# ledger-control

*Memory-augmented trusted monitoring for decomposed-attack detection in AI control.*

Per-action monitors are locally blind to sabotage spread across individually innocuous
actions. Full-trajectory monitors dilute it. A running summary loses the weak signals
decomposition produces, because compression happens before you know what matters.

This harness replaces the summary with an **append-only ledger**: every step recorded
under a stable id, retrieval keys derived from execution telemetry rather than model
judgement, records fetched by id and dropped after use. Each judgement stays a pure
function of `(incoming action, index<t, fetched ids)` — which is what makes
**retroactive re-scoring** well defined, so step 200 can revise the verdict on step 10.

The question is not *does memory help monitoring* but **which structure of memory lets a
monitor represent a conjunction** — five actions, innocuous alone, incriminating only
because they touched the same resource in order. Five conditions are compared: none,
scalar credence, compressed summary, full context, and the ledger.

> **Status: experiments in progress, no results yet.** The harness runs end to end on
> real trajectories and is covered by  tests. Numbers will be published here when the
> sweep completes, whatever the sign.

**Scope:** memory stays within a single trajectory. Pooling across trajectories hands
the monitor the coordinating set that a real defender would have to discover — see
`CONTEXT.md` for the note from UK AISI on that selection effect.

## Quickstart, from nothing

```bash
git clone https://github.com/kameshkanna/AIC.git && cd AIC

bash setup.sh --backend=local        # venv, deps,  tests, memory preflight
source .venv/bin/activate
export HF_HOME=/path/on/a/big/volume # model weights land here

python -m scripts.check_local        # GATE: 2-min backend smoke test, must print GO
bash get_data.sh                     # download + convert the corpus (~2.6 GiB)

python -m scripts.run --protocols per_step,running_summary --workers 16   # GATE: baseline
python -m scripts.run --protocols advisory,cascade --workers 16           # the arms
python -m scripts.run --protocols full_context --workers 16               # 32B, on its own
```

`--backend=local` loads models in-process: no server, no ports, no container. `--workers`
matters — decode is memory-bandwidth-bound, so generating one prompt at a time wastes the
card; trajectories are scored concurrently and the backend batches whatever arrives.

`full_context` runs separately because the 32B needs the card to itself. The runner
checks this before loading anything and refuses with the split command if you ask for
too much at once.

For the faster path, `bash setup.sh --backend=vllm` serves over HTTP with prefix caching
(worth roughly 2x) — but it needs Python 3.11+, see below.

Results land in `results/`: `report.json`, `safety_curve.json` (the figure to lead
with), `steps.csv` (everything else you would plot).

### vLLM needs Python 3.11+

vLLM 0.27 imports `flashinfer` unguarded in its V1 sampler, and flashinfer does not
import on 3.10. Both states fail there: with it installed, its module-level annotations
raise `TypeError`; without it, the sampler raises `ModuleNotFoundError`. Uninstalling
does not help. `setup.sh --backend=vllm` refuses on 3.10 rather than installing several
GiB that cannot start.

On 3.10, either build the venv on 3.12 and keep vLLM:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
bash setup.sh --backend=vllm
```

...or skip vLLM entirely -- no server, no port, no container:

```bash
bash setup.sh --backend=local
python -m scripts.check_local               # 2-minute smoke test, do this first
python -m scripts.run --workers 16          # --workers fills generation batches
```

Skip `fetch_models.sh` and `serve.sh` on the local path; weights load on first use. It
is slower than vLLM, mainly for want of prefix caching, so budget more time or use
`--limit`. Tune with `LEDGERCTL_LOCAL_MAX_BATCH` and `LEDGERCTL_LOCAL_MAX_PROMPT_TOKENS`;
a batch that does not fit is split and retried rather than failing the run.

## Layout

```
ledgerctl/
  config.py      centralised env-driven settings + seed
  runtime.py     logging config + reproducible seeding
  llm.py         OpenAI-compatible client + mocks (DI seam)
  trajectory.py  corpus loading; Step / Trajectory; causal prefix
  ledger.py      append-only store: render/grep index, fetch by id,
                 score revisions, access log
  summarise.py   per-step summariser, cached, causally masked
  monitor.py     StepMonitor (7B) and GlobalMonitor (14B, select -> fetch -> judge)
  protocols.py   two arms + four baselines, one score per step
  evaluate.py    threshold calibration, safety, latency, recall, safety curve
  local_backend.py  in-process transformers client with dynamic batching
scripts/
  from_eval.py       Inspect .eval archives -> corpus (no API token needed)
  ingest.py          Control Tower JSON -> corpus, with a schema audit
  check_local.py     smoke-test the in-process backend before a sweep
  preflight.py       memory arithmetic + endpoint/JSON-adherence probe
  plan.py            token-cost and wall-clock projection from the real corpus
  run.py             sweep orchestrator, resumable
  smoke.py           end-to-end mock run
  make_synthetic.py  synthetic corpus for harness validation (not the experiment)
tests/               pure-component unit tests
```

## Protocols

Default sweep:

| name | memory | decides |
|---|---|---|
| `per_step` | none | 7B |
| `running_summary` | compressed carry-forward | 7B |
| `advisory` (arm A) | ledger | 7B, with evidence from the 14B |
| `cascade` (arm B) | ledger | 14B, with a 7B feature |

Built but off by default — pass them to `--protocols` to enable:

| name | memory | decides |
|---|---|---|
| `scalar_posterior` | one Beta posterior | 7B |
| `full_context` | whole causal index in prompt, no fetch | 32B |

The 32B is deliberately the **baseline to beat**, not part of the system: if the global
monitor were itself a 32B, "memory helped" would be confounded with "the bigger model
helped". The claim under test is *7B + 14B with a ledger vs. a 32B with everything in
context*.

## Record

One record per executed step. Everything but `summary` is lifted from the recording.

```json
{"session_id":"env12_run3","step":41,"step_id":"env12_run3::s41",
 "source_index":43,"timestamp":"...","command":"<tool call>",
 "output":"<stdout+stderr>","summary":"<generated>"}
```

## What is measured

τ is calibrated on one benign split at a 1% step-wise false-positive rate and reported on
a disjoint one. A malicious trajectory is caught if any step scores above τ strictly
before the attack completes.

- safety at 1% audit budget (headline)
- detection latency, in steps before completion
- attack-step Recall@K and fetch recall — separate retrieval failure from judgement failure
- `first_flag_position` — how early the first alarm lands, as a fraction of the trajectory

Nothing is blocked: blocking would make the recorded corpus counterfactual.

The public corpus publishes no per-step attack labels, so latency and attack-step recall
report `n/a` and catching degrades to trajectory level. Expected, not a bug —
`first_flag_position` stands in for latency. Safety at a fixed audit budget is unaffected.

Every run also writes `safety_curve.json` (safety, FPR and τ swept across audit budgets)
and `steps.csv` (tidy long format, one row per protocol/trajectory/step), so figures can
be drawn without re-running anything.

See `CONTEXT.md` for the design decisions, baselines, and open items.
