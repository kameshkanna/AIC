#!/usr/bin/env bash
# Start the vLLM servers for a sweep or for the baseline pass.
#
# Two backends, auto-detected. If vLLM is importable in the active environment it
# is used directly; otherwise containers are used. vLLM publishes an aarch64
# wheel (manylinux_2_28, cp38-abi3), so `pip install vllm` works on a GH200 and
# the local backend needs neither Docker nor nvidia-container-toolkit nor sudo.
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
# `--swap-space` is deliberately absent: it was a v0-engine argument and vLLM 0.27
# rejects it outright. The V1 engine handles CPU offload on its own.
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
#   BACKEND=docker bash serve.sh sweep                 # force containers
#   UTIL_7B=0.18 UTIL_14B=0.50 bash serve.sh sweep     # tune memory fractions
#   ENFORCE_EAGER=1 bash serve.sh sweep                # skip torch.compile
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MAX_LEN="${MAX_MODEL_LEN:-16384}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"
LOG_DIR="${LOG_DIR:-logs/serve}"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 1; }
}

detect_backend() {
  if [ -n "${BACKEND:-}" ]; then
    echo "$BACKEND"
    return 0
  fi
  if python -c "import vllm" >/dev/null 2>&1; then
    echo local
  else
    echo docker
  fi
}

check_gpu() {
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "!! nvidia-smi failed; no usable GPU" >&2
    exit 1
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

check_docker() {
  require docker
  if ! docker info 2>/dev/null | grep -qi nvidia; then
    cat >&2 <<'MSG'
!! docker has no nvidia runtime.

   Easiest fix is to skip Docker entirely -- vLLM ships an aarch64 wheel:

       pip install vllm
       bash serve.sh sweep

   To use Docker anyway, install the container toolkit (needs sudo; it is an
   apt package, not a pip one):

       curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
         | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
       curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
         | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
         | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
       sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
       sudo nvidia-ctk runtime configure --runtime=docker
       sudo systemctl restart docker
MSG
    exit 1
  fi
}

pidfile() { echo "$LOG_DIR/$1.pid"; }
logfile() { echo "$LOG_DIR/$1.log"; }

start_local() {
  local name="$1" model="$2" port="$3" util="$4"
  mkdir -p "$LOG_DIR"
  if [ -f "$(pidfile "$name")" ] && kill -0 "$(cat "$(pidfile "$name")")" 2>/dev/null; then
    echo ">> $name already running on port $port"
    return 0
  fi
  echo ">> starting $name  ($model)  port $port  gpu-util $util  [local]"
  local extra=()
  [ -n "${ENFORCE_EAGER:-}" ] && extra+=(--enforce-eager)
  HF_HOME="$HF_HOME" nohup vllm serve "$model" \
    --served-model-name "$model" \
    --port "$port" \
    --enable-prefix-caching \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$util" \
    ${extra[@]+"${extra[@]}"} \
    >"$(logfile "$name")" 2>&1 &
  echo $! > "$(pidfile "$name")"
}

start_docker() {
  local name="$1" model="$2" port="$3" util="$4"
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo ">> $name already running on port $port"
    return 0
  fi
  echo ">> starting $name  ($model)  port $port  gpu-util $util  [docker]"
  docker run -d --rm --name "$name" --gpus all --ipc=host \
    -p "${port}:8000" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMAGE" \
    --model "$model" --served-model-name "$model" \
    --enable-prefix-caching --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$util" >/dev/null
}

start_model() {
  if [ "$BACKEND" = "local" ]; then start_local "$@"; else start_docker "$@"; fi
}

# A model is alive if its process/container is up; ready once it answers /v1/models.
still_alive() {
  local name="$1"
  if [ "$BACKEND" = "local" ]; then
    [ -f "$(pidfile "$name")" ] && kill -0 "$(cat "$(pidfile "$name")")" 2>/dev/null
  else
    docker ps --format '{{.Names}}' | grep -qx "$name"
  fi
}

# vLLM's traceback ends with "See root cause above", so a blind tail prints the
# symptom and hides the cause. Surface the first real error, then the tail, then
# say where the whole log is.
show_tail() {
  local name="$1" log
  if [ "$BACKEND" = "local" ]; then
    log="$(logfile "$name")"
    [ -f "$log" ] || return 0
    # The exception is the LAST line of a traceback, not the first. Print the tail
    # of the worker's error block, where the actual cause lives, before anything
    # else -- the API-server traceback that follows is only the symptom.
    echo "--- root cause (end of worker error block) ---"
    grep -E "EngineCore|Worker|ERROR" "$log" 2>/dev/null \
      | grep -vE "See root cause|Engine core initialization failed" | tail -15 || true
    echo "--- exception lines anywhere in the log ---"
    grep -nE "^[A-Za-z_.]*(Error|Exception|Failure)[:( ]|RuntimeError|ImportError|OSError|AssertionError|CUDA out of memory|Killed" "$log" 2>/dev/null \
      | grep -vE "See root cause" | tail -10 || true
    echo "--- last 20 lines ---"
    tail -n 20 "$log" 2>/dev/null || true
    echo "--- full log: $log ---"
  else
    docker logs --tail 60 "$name" 2>&1 || true
  fi
}

wait_ready() {
  local name="$1" port="$2" waited=0
  printf ">> waiting for %s on :%s " "$name" "$port"
  while [ "$waited" -lt "$WAIT_SECONDS" ]; do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo " ready (${waited}s)"
      return 0
    fi
    if ! still_alive "$name"; then
      echo; echo "!! $name exited during startup; last output:" >&2
      show_tail "$name"
      return 1
    fi
    printf "."
    sleep 10
    waited=$((waited + 10))
  done
  echo; echo "!! $name did not become ready within ${WAIT_SECONDS}s" >&2
  show_tail "$name"
  return 1
}

BACKEND="$(detect_backend)"

case "${1:-}" in
  sweep)
    echo ">> backend: $BACKEND"
    check_gpu
    [ "$BACKEND" = "docker" ] && check_docker
    require curl
    # Sequential, not parallel. Each engine profiles free VRAM during init, and
    # two doing that at once race: the second claims memory the first already
    # counted as available, and one of them dies partway through startup.
    start_model ledgerctl-7b  "Qwen/Qwen2.5-7B-Instruct"  8001 "${UTIL_7B:-0.20}"
    wait_ready ledgerctl-7b  8001
    start_model ledgerctl-14b "Qwen/Qwen2.5-14B-Instruct" 8000 "${UTIL_14B:-0.55}"
    wait_ready ledgerctl-14b 8000
    cat <<'MSG'

both models up.

next:
  python -m scripts.preflight        probe endpoints + JSON adherence
  python -m scripts.run --protocols per_step,running_summary
MSG
    ;;
  baseline)
    echo ">> backend: $BACKEND"
    check_gpu
    [ "$BACKEND" = "docker" ] && check_docker
    require curl
    if still_alive ledgerctl-7b || still_alive ledgerctl-14b; then
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
      if [ "$BACKEND" = "local" ]; then
        if [ -f "$(pidfile "$name")" ]; then
          pid="$(cat "$(pidfile "$name")")"
          if kill -0 "$pid" 2>/dev/null; then echo ">> stopping $name (pid $pid)"; kill "$pid"; fi
          rm -f "$(pidfile "$name")"
        fi
      elif docker ps --format '{{.Names}}' | grep -qx "$name"; then
        echo ">> stopping $name"; docker stop "$name" >/dev/null
      fi
    done
    ;;
  status)
    echo ">> backend: $BACKEND"
    for name in ledgerctl-7b ledgerctl-14b ledgerctl-32b; do
      if still_alive "$name"; then echo "  $name: up"; else echo "  $name: down"; fi
    done
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    ;;
  -h|--help|"")
    sed -n '2,30p' "$0"
    exit 0
    ;;
  *)
    echo "unknown profile: $1 (want: sweep|baseline|stop|status)" >&2
    exit 2
    ;;
esac
