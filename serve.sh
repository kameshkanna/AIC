#!/usr/bin/env bash
# Start the vLLM containers for a sweep or for the baseline pass.
#
# Two profiles, because the memory arithmetic differs:
#
#   sweep     7B (port 8001) + 14B (port 8000), co-resident.
#             41.7 GiB of weights on a 96 GiB card leaves 49.3 GiB of KV cache,
#             roughly 33 concurrent requests at 8K context.
#
#   baseline  32B (port 8002), alone, run after the sweep finishes.
#             Co-residing it with the 7B leaves only 15.7 GiB of KV and drops
#             concurrency to about 8, so it gets the card to itself.
#
# Prefix caching is enabled everywhere. The monitor prompt is ordered
# [system][index<t][fetched][incoming], so the index is a growing shared prefix
# and caching turns per-step prefill from linear in t into constant.
#
# Usage:
#   bash serve.sh sweep
#   bash serve.sh baseline
#   bash serve.sh stop
#   bash serve.sh status
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MAX_LEN="${MAX_MODEL_LEN:-16384}"
SWAP_GIB="${SWAP_SPACE:-32}"
WAIT_SECONDS="${WAIT_SECONDS:-900}"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

check_runtime() {
  require docker
  require curl
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "!! nvidia container runtime not visible to docker" >&2
    echo "   install nvidia-container-toolkit, then: sudo systemctl restart docker" >&2
    exit 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "!! nvidia-smi failed; no usable GPU" >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

# Start one vLLM container. Named so `stop` and `status` can find it again.
start_model() {
  local name="$1" model="$2" port="$3" util="$4"
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo ">> $name already running on port $port"
    return 0
  fi
  echo ">> starting $name  ($model)  port $port  gpu-util $util"
  docker run -d --rm \
    --name "$name" \
    --gpus all \
    --ipc=host \
    -p "${port}:8000" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE" \
    --model "$model" \
    --served-model-name "$model" \
    --enable-prefix-caching \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$util" \
    --swap-space "$SWAP_GIB" \
    >/dev/null
}

# Poll the OpenAI-compatible models endpoint until the weights finish loading.
wait_ready() {
  local name="$1" port="$2" waited=0
  printf ">> waiting for %s on :%s " "$name" "$port"
  while [ "$waited" -lt "$WAIT_SECONDS" ]; do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo " ready (${waited}s)"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      echo
      echo "!! container $name exited during startup; last output:" >&2
      docker logs --tail 40 "$name" 2>&1 || true
      return 1
    fi
    printf "."
    sleep 10
    waited=$((waited + 10))
  done
  echo
  echo "!! $name did not become ready within ${WAIT_SECONDS}s" >&2
  docker logs --tail 40 "$name" 2>&1 || true
  return 1
}

case "${1:-}" in
  sweep)
    check_runtime
    start_model ledgerctl-7b  "Qwen/Qwen2.5-7B-Instruct"  8001 0.20
    start_model ledgerctl-14b "Qwen/Qwen2.5-14B-Instruct" 8000 0.55
    wait_ready ledgerctl-7b  8001
    wait_ready ledgerctl-14b 8000
    cat <<'MSG'

both models up.

next:
  python -m scripts.preflight        probe endpoints + JSON adherence
  python -m scripts.run --protocols per_step,running_summary
MSG
    ;;
  baseline)
    check_runtime
    if docker ps --format '{{.Names}}' | grep -qE 'ledgerctl-(7b|14b)'; then
      echo "!! stop the sweep models first: bash serve.sh stop" >&2
      echo "   the 32B needs the whole card" >&2
      exit 1
    fi
    start_model ledgerctl-32b "Qwen/Qwen2.5-32B-Instruct" 8002 0.90
    wait_ready ledgerctl-32b 8002
    echo
    echo "next: python -m scripts.run --protocols full_context"
    ;;
  stop)
    for name in ledgerctl-7b ledgerctl-14b ledgerctl-32b; do
      if docker ps --format '{{.Names}}' | grep -qx "$name"; then
        echo ">> stopping $name"
        docker stop "$name" >/dev/null
      fi
    done
    ;;
  status)
    docker ps --filter "name=ledgerctl-" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    ;;
  -h|--help|"")
    sed -n '2,25p' "$0"
    exit 0
    ;;
  *)
    echo "unknown profile: $1 (want: sweep|baseline|stop|status)" >&2
    exit 2
    ;;
esac
