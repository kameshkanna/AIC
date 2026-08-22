#!/usr/bin/env bash
# Pre-download model weights into a shared Hugging Face cache.
#
# Serving containers mount this cache read-write, so weights are fetched once and
# reused across every container start. Downloading up front also means a container
# restart is seconds rather than a re-download, which matters when the GPU window
# is booked.
#
# Footprint at bf16: 7B ~14 GiB, 14B ~28 GiB, 32B ~61 GiB, about 103 GiB total.
#
# Usage:
#   bash fetch_models.sh              # all three
#   bash fetch_models.sh sweep        # 7B + 14B only
#   bash fetch_models.sh baseline     # 32B only
set -euo pipefail

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SELECTION="${1:-all}"

SWEEP_MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct")
BASELINE_MODELS=("Qwen/Qwen2.5-32B-Instruct")

case "$SELECTION" in
  sweep)    MODELS=("${SWEEP_MODELS[@]}") ;;
  baseline) MODELS=("${BASELINE_MODELS[@]}") ;;
  all)      MODELS=("${SWEEP_MODELS[@]}" "${BASELINE_MODELS[@]}") ;;
  -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
  *) echo "unknown selection: $SELECTION (want: all|sweep|baseline)" >&2; exit 2 ;;
esac

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo ">> installing huggingface_hub[cli]"
  python -m pip install --quiet "huggingface_hub[cli]"
fi

if command -v hf >/dev/null 2>&1; then
  DOWNLOAD=(hf download)
else
  DOWNLOAD=(huggingface-cli download)
fi

FREE_GIB="$(df -PBG "$HF_HOME" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}' || echo 0)"
echo ">> cache:      $HF_HOME"
echo ">> free space: ${FREE_GIB} GiB"
if [ "${FREE_GIB:-0}" -lt 130 ] && [ "$SELECTION" = "all" ]; then
  echo "!! less than 130 GiB free; all three models need about 103 GiB" >&2
fi

mkdir -p "$HF_HOME"
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

for model in "${MODELS[@]}"; do
  echo
  echo ">> fetching $model"
  "${DOWNLOAD[@]}" "$model" --exclude "*.pth" "original/*"
done

echo
echo "done. cache at $HF_HOME"
echo "next: bash serve.sh sweep"
