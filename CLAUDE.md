# ledger-control — session rules

Read `CONTEXT.md` first. It is the durable record; do not re-derive it from the code.

## Invariants that must not be broken

These are correctness properties of the experiment, not style preferences. Breaking any
one of them silently invalidates every number the harness produces.

1. **Causal mask.** At step *t* a monitor may see `index[< t]` and `command_t`. Never
   `output_t`, never a later step. `render_index`, `grep_index` and `fetch` all take
   `upto` — pass it.
2. **No label leakage.** `is_attack_step`, `is_malicious` and `attack_completion_step`
   are evaluation-only. `Step.to_record()` strips the gold label; do not add it back,
   and never put trajectory-level context in a summariser prompt.
3. **Append-only.** Records are never mutated. Retroactive re-scoring goes through
   `Ledger.revise`, which appends.
4. **Ledger identical across arms.** Summaries are cached for this reason. If an arm
   needs different state, it belongs in a separate stream, not in the step record.
5. **Disjoint benign sets.** τ is calibrated on one benign split and FPR reported on
   another. Never the same one.
6. **Score, not bool.** Monitors emit a continuous score; thresholds are applied
   downstream so the audit budget can be swept.

## Scope discipline

The step record is `{session_id, step, step_id, timestamp, command, output, summary}`.
Everything but `summary` comes from the recording. Do not add an extraction stage, a
typed-field schema, or a second model — that was considered and cut. Anything needed
from `command` is a string parse at query time.

## Environment

```bash
bash setup.sh                        # venv, deps, tests, memory preflight
bash fetch_models.sh                 # pre-download weights into a shared HF cache
bash serve.sh sweep | baseline | stop | status
source .venv/bin/activate
python -m pytest                     # 62 unit tests, no endpoint needed
python -m scripts.smoke              # ingest path on a synthetic trajectory
python -m scripts.make_synthetic     # synthetic corpus
python -m scripts.run --mock keyword # full sweep, all six protocols
```

Do not add torch or vLLM to base dependencies — they are the `gpu` extra, and that
extra is refused on aarch64 because vLLM has no ARM wheels. Models are served by
containers via `serve.sh`, never by a pip-installed vLLM on the GH200.

## Fixtures are not the experiment

`make_synthetic.py` exists to exercise the harness without a corpus. Once real data
is available it plays no part in any run. Do not report a number computed on it.

## Mocks are fixtures, not monitors

`--mock constant` gives safety 0 by construction (threshold calibrated from the same
constant). `--mock keyword` keys on the synthetic attack's own vocabulary, so it gives
safety 1. Neither is a result. They exist to prove the metric path runs.

## Before touching the loader

`FIELD_ALIASES` in `trajectory.py` is a guess at LaStraj's field names. Confirm it
against the real dump before trusting any run. This is the top open item in `CONTEXT.md`.
