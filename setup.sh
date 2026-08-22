#!/usr/bin/env bash
# Create the project virtualenv, activate it, and install dependencies.
#
# This installs the harness only. The models are served separately by
# ./serve.sh, which uses containers rather than a pip-installed vLLM.
#
# Usage:
#   bash setup.sh          # harness + dev tools
#   bash setup.sh --gpu    # additionally pip-install vLLM (x86_64 only)
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
EXTRAS="dev"
ARCH="$(uname -m 2>/dev/null || echo unknown)"

# Resolve a working interpreter. On Windows `python3` is often a Microsoft Store
# alias stub that exits without starting Python, so candidates are probed by
# actually running them rather than by `command -v` alone.
detect_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return 0
  fi
  for candidate in python3 python py; do
    if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

for arg in "$@"; do
  case "$arg" in
    --gpu)
      # vLLM publishes a manylinux_2_28 aarch64 wheel (cp38-abi3), so this works
      # on a GH200 and avoids needing Docker, nvidia-container-toolkit or sudo.
      EXTRAS="gpu,dev"
      ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

if ! PYTHON_BIN="$(detect_python)"; then
  echo "no python >= 3.10 found; set PYTHON_BIN explicitly" >&2
  exit 1
fi
echo ">> arch:        $ARCH"
echo ">> interpreter: $PYTHON_BIN"

echo ">> creating venv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  ACTIVATE="$VENV_DIR/bin/activate"
else
  ACTIVATE="$VENV_DIR/Scripts/activate"
fi

echo ">> activating $ACTIVATE"
# shellcheck disable=SC1090
source "$ACTIVATE"

echo ">> upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo ">> installing package [$EXTRAS]"
python -m pip install -e ".[$EXTRAS]"

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ">> wrote .env from .env.example"
fi

echo ">> running unit tests"
python -m pytest -q

echo ">> memory preflight"
python -m scripts.preflight --offline

cat <<MSG

done. activate with:  source $ACTIVATE

next:
  pip install vllm              serve locally, no Docker needed
  bash fetch_models.sh          pre-download model weights (~105 GiB)
  bash serve.sh sweep           start 7B + 14B
  python -m scripts.preflight   probe endpoints and JSON adherence
MSG
