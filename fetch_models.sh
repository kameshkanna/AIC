#!/usr/bin/env bash
# Pre-download model weights into a shared Hugging Face cache.
#
# Serving containers mount this cache read-write, so weights are fetched once and
# reused across every container start. Downloading up front also means a container
# restart is seconds rather than a re-download, which matters when the GPU window
# is booked.
#
# Footprint at bf16: 7B ~15 GiB, 14B ~28 GiB, 32B ~62 GiB, about 105 GiB total.
#
# The cache location matters: a default ~/.cache on a small root volume will fill
# up. Point HF_HOME at the large volume before running.
#
#   export HF_HOME=/lambda/nfs/AIC/hf-cache
#   bash fetch_models.sh
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
  sweep)     MODELS=("${SWEEP_MODELS[@]}");    NEED_GIB=45 ;;
  baseline)  MODELS=("${BASELINE_MODELS[@]}"); NEED_GIB=70 ;;
  all)       MODELS=("${SWEEP_MODELS[@]}" "${BASELINE_MODELS[@]}"); NEED_GIB=115 ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
  *) echo "unknown selection: $SELECTION (want: all|sweep|baseline)" >&2; exit 2 ;;
esac

# The `hf` CLI ships with huggingface_hub itself; the old `[cli]` extra was
# removed and asking for it prints a warning while installing nothing useful.
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo ">> installing huggingface_hub"
  python -m pip install --quiet --upgrade huggingface_hub
fi

if command -v hf >/dev/null 2>&1; then
  DOWNLOAD=(hf download)
  AUTH_HINT="hf auth login"
else
  DOWNLOAD=(huggingface-cli download)
  AUTH_HINT="huggingface-cli login"
fi

# Create the cache before measuring it: df on a non-existent path reports nothing,
# which is what made this print "0 GiB" on a perfectly healthy box.
mkdir -p "$HF_HOME"
export HF_HOME

# Xet replaced hf_transfer; the old variable is deprecated and warns on every call.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

# `df -P` guarantees one record per filesystem even when the device name is long
# enough to wrap, so read the last line rather than line 2.
FREE_GIB="$(df -PBG "$HF_HOME" 2>/dev/null | awk 'END {gsub(/G/,"",$4); print $4+0}')"
MOUNT="$(df -P "$HF_HOME" 2>/dev/null | awk 'END {print $6}')"

echo ">> cache:      $HF_HOME"
echo ">> filesystem: ${MOUNT:-unknown}"
echo ">> free space: ${FREE_GIB:-0} GiB (need ~${NEED_GIB} GiB for '$SELECTION')"

if [ "${FREE_GIB:-0}" -lt "$NEED_GIB" ]; then
  cat >&2 <<MSG

!! not enough free space on ${MOUNT:-that filesystem}.

   Point the cache at a larger volume and re-run, for example:
       export HF_HOME=/lambda/nfs/AIC/hf-cache
       bash fetch_models.sh $SELECTION

MSG
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo ">> note: not authenticated. Downloads work but are rate-limited."
  echo ">>       run '$AUTH_HINT' first for faster, higher-limit transfers."
fi

for model in "${MODELS[@]}"; do
  echo
  echo ">> fetching $model"
  # Each --exclude takes exactly one pattern. Passing two patterns after a single
  # flag makes the second one a positional FILENAME, so the CLI ignores every
  # exclude and tries to download a file literally named "original/*".
  "${DOWNLOAD[@]}" "$model" --exclude "*.pth" --exclude "original/*" --exclude "*.gguf"
done

echo
echo ">> cache contents:"
du -sh "$HF_HOME" 2>/dev/null || true
echo
echo "done. next: bash serve.sh sweep"
