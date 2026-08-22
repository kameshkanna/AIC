#!/usr/bin/env bash
# One-command environment setup.
#
#   bash setup.sh                      harness only (no model serving)
#   bash setup.sh --backend vllm       + vLLM, served over HTTP (fastest)
#   bash setup.sh --backend local      + torch/transformers, in-process, no server
#
# Ends by running the unit tests and the memory preflight, so a broken
# environment is reported in about a minute rather than an hour into a GPU
# booking.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
BACKEND="none"
ARCH="$(uname -m 2>/dev/null || echo unknown)"

while [ $# -gt 0 ]; do
  case "$1" in
    --backend=*) BACKEND="${1#*=}" ;;
    --backend)   shift; BACKEND="${1:-}" ;;
    vllm|local)  BACKEND="$1" ;;
    --gpu)       BACKEND="vllm" ;;
    -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
    *)           echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$BACKEND" in
  none|vllm|local) ;;
  *) echo "unknown backend: $BACKEND (want: vllm|local)" >&2; exit 2 ;;
esac

# On Windows `python3` is often a Store alias stub that exits without starting
# Python, so probe candidates by running them rather than trusting `command -v`.
detect_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then echo "$PYTHON_BIN"; return 0; fi
  for candidate in python3 python py; do
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(detect_python)"; then
  echo "no python >= 3.10 found; set PYTHON_BIN explicitly" >&2
  exit 1
fi

echo ">> arch:        $ARCH"
echo ">> interpreter: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"
echo ">> backend:     $BACKEND"

echo ">> creating venv at $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  ACTIVATE="$VENV_DIR/bin/activate"
else
  ACTIVATE="$VENV_DIR/Scripts/activate"
fi
# shellcheck disable=SC1090
source "$ACTIVATE"

echo ">> upgrading pip"
python -m pip install --quiet --upgrade pip setuptools wheel

echo ">> installing harness"
python -m pip install --quiet -e ".[dev]"

case "$BACKEND" in
  vllm)
    echo ">> installing vLLM (this is large; several minutes)"
    python -m pip install --quiet vllm

    # vLLM pulls in flashinfer for multi-GPU fused allreduce. It is unused on a
    # single card, and on Python < 3.11 it raises TypeError at import time --
    # which vLLM's ImportError fallback does not catch, so the whole engine dies
    # during startup. Remove it rather than leaving a landmine.
    if python -c "import sys; sys.exit(0 if sys.version_info < (3,11) else 1)"; then
      if python -m pip show flashinfer-python >/dev/null 2>&1 || python -m pip show flashinfer >/dev/null 2>&1; then
        echo ">> removing flashinfer (broken on $(python -V 2>&1 | cut -d' ' -f2), unused on one GPU)"
        python -m pip uninstall -y flashinfer-python flashinfer >/dev/null 2>&1 || true
      fi
    fi
    ;;
  local)
    echo ">> installing torch + transformers"
    python -m pip install --quiet -e ".[local]"
    ;;
esac

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ">> wrote .env from .env.example"
fi
if [ "$BACKEND" = "local" ]; then
  if grep -q '^LEDGERCTL_BACKEND=' .env 2>/dev/null; then
    sed -i.bak 's/^LEDGERCTL_BACKEND=.*/LEDGERCTL_BACKEND=transformers/' .env && rm -f .env.bak
  else
    echo "LEDGERCTL_BACKEND=transformers" >> .env
  fi
  echo ">> .env set to the in-process backend"
fi

echo
echo ">> unit tests"
python -m pytest -q

echo
echo ">> memory preflight"
python -m scripts.preflight --offline

cat <<MSG

setup complete. activate with:  source $ACTIVATE

next:
  bash fetch_models.sh sweep     download 7B + 14B weights (~45 GiB)
  bash serve.sh sweep            start them            [vllm backend only]
  python -m scripts.preflight    endpoint + JSON check
  bash get_data.sh               download and convert the trajectory corpus
  python -m scripts.run --protocols per_step,running_summary
MSG
