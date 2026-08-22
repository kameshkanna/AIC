#!/usr/bin/env bash
# Download the trajectory corpus and convert it to this harness's format.
#
# Uses the public HuggingFace mirror, not the Control Tower API, so no API token
# is needed. The API is token-gated and the paper's "LaStraj" name is stale --
# the dataset now lives under GreatStraj.
#
# The mirror publishes no per-step attack labels, so detection latency and
# attack-step recall will report n/a. Safety at a fixed audit budget, the headline
# number, is unaffected.
#
# Usage:
#   bash get_data.sh                # full corpus, ~2.6 GiB
#   bash get_data.sh --sample       # one small archive, for a quick end-to-end check
set -euo pipefail

DATASET="anonymouslinuxarena/straj_linuxarena"
LOCAL_DIR="${STRAJ_DIR:-data/straj}"
OUT="${CORPUS_OUT:-data/trajectories.jsonl}"
SAMPLE=0

for arg in "$@"; do
  case "$arg" in
    --sample) SAMPLE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo ">> installing huggingface_hub"
  python -m pip install --quiet --upgrade huggingface_hub
fi
if command -v hf >/dev/null 2>&1; then
  DOWNLOAD=(hf download)
else
  DOWNLOAD=(huggingface-cli download)
fi

mkdir -p "$LOCAL_DIR"

echo ">> results table"
"${DOWNLOAD[@]}" "$DATASET" --repo-type dataset --local-dir "$LOCAL_DIR" \
  --include "precomputed_results.csv" "README.md"

if [ "$SAMPLE" = "1" ]; then
  echo ">> one sample archive (quick check)"
  "${DOWNLOAD[@]}" "$DATASET" --repo-type dataset --local-dir "$LOCAL_DIR" \
    --include "evals/gpt-4-turbo_stripped_20260427_225623/*/*.eval"
else
  echo ">> full trajectory set (~2.6 GiB)"
  "${DOWNLOAD[@]}" "$DATASET" --repo-type dataset --local-dir "$LOCAL_DIR" --include "evals/*"
fi

RESULTS="$LOCAL_DIR/precomputed_results.csv"
if [ ! -f "$RESULTS" ]; then
  echo "!! results table missing at $RESULTS" >&2
  exit 1
fi

N_EVAL="$(find "$LOCAL_DIR/evals" -name '*.eval' 2>/dev/null | wc -l | tr -d ' ')"
echo ">> archives downloaded: $N_EVAL"
if [ "$N_EVAL" = "0" ]; then
  echo "!! no .eval archives found under $LOCAL_DIR/evals" >&2
  exit 1
fi

echo ">> converting"
python -m scripts.from_eval --src "$LOCAL_DIR/evals" --results "$RESULTS" --out "$OUT"

echo
echo "corpus: $OUT"
echo "next:   python -m scripts.plan --budget-hours 12"
