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
   and never put trajectory-level context in a summariser prompt. `HIDDEN_TOOLS` are
   dropped for the same reason: two of them are the side-task scorer.
3. **Append-only.** Records are never mutated. Retroactive re-scoring goes through
   `Ledger.revise`, which appends.
4. **Ledger identical across arms.** Summaries are cached for this reason. If an arm
   needs different state, it belongs in a separate stream, not in the step record.
5. **Disjoint benign splits.** τ is calibrated on one benign split and FPR reported on
   another. Never the same one.
6. **Score, not bool.** Monitors emit a continuous score; thresholds are applied
   downstream so the audit budget can be swept.
7. **Memory stays within one trajectory.** Pooling across trajectories hands the monitor
   the coordinating set a real defender would have to discover — see the AISI note in
   `CONTEXT.md`. Any cross-trajectory extension needs realistic benign density.

## Scope discipline

The step record is `{session_id, step, step_id, source_index, timestamp, command,
output, summary}`. Everything but `summary` comes from the recording. Do not add an
extraction stage, a typed-field schema, or a second model — that was considered and cut.
Anything needed from `command` is a string parse at query time.

## Environment

```bash
bash setup.sh --backend=local        # torch + transformers, in-process, no server
bash setup.sh --backend=vllm         # vLLM over HTTP; needs Python 3.11+
source .venv/bin/activate

python -m pytest                     # 122 unit tests, no endpoint needed
python -m scripts.check_local        # 2-min smoke test of the in-process backend
python -m scripts.smoke              # ingest path on a synthetic trajectory
python -m scripts.run --mock keyword # sweep with mock clients, no GPU
bash get_data.sh                     # download + convert the real corpus
```

**vLLM needs Python 3.11+.** Its V1 sampler imports `flashinfer` unguarded, and
flashinfer does not import on 3.10 — with it installed you get `TypeError`, without it
`ModuleNotFoundError`. Uninstalling does not help. `setup.sh` refuses `--backend=vllm`
on 3.10 and points at a 3.12 venv or the local backend. vLLM *does* publish aarch64
wheels from 0.27.1, so Docker is optional; `serve.sh` auto-detects and uses local vLLM
when importable.

## Model footprint

Clients are constructed lazily, because on the in-process backend a client is a loaded
model. 7B + 14B co-reside on a 96 GiB card; adding the 32B does not. Run
`full_context` in its own invocation. `scripts/run.py` checks this before loading
anything and refuses with the split command to use.

## Fixtures are not the experiment

`make_synthetic.py` exists to exercise the harness without a corpus, and the `--mock`
clients exist to prove the metric path runs. `--mock constant` gives safety 0 by
construction; `--mock keyword` keys on the synthetic attack's own vocabulary and gives
safety 1. None of these is a result. Do not report a number computed on one.

## Corpus notes

Field names are verified against control-tower source, not guessed — see the schema
table in `CONTEXT.md`. `scripts/ingest.py --audit-only` re-checks them against any new
dump and refuses to convert on a miss.

The public mirror publishes no per-step attack labels, so `detection_latency` and
`attack_step_recall` report `n/a` and catching degrades to trajectory level. That is
expected, not a bug. `first_flag_position` stands in for latency.
