#!/usr/bin/env bash
# Start CI Signal locally with the Python version supported by this project.
#
# Usage:
#   ./scripts/run_local.sh          # prepare dependencies and run the dashboard
#   ./scripts/run_local.sh --check  # verify the prepared environment and health endpoint

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CI_SIGNAL_VENV:-${PROJECT_DIR}/.venv311}"
HOST="${CI_SIGNAL_HOST:-127.0.0.1}"
PORT="${CI_SIGNAL_PORT:-8501}"
MODE="run"

if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--check]" >&2
  exit 64
fi

is_python_311() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)' >/dev/null 2>&1
}

find_python_311() {
  local candidate
  for candidate in "${PYTHON_311:-}" "${VENV_DIR}/bin/python" python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    [[ -n "${candidate}" ]] || continue
    if command -v "${candidate}" >/dev/null 2>&1 && is_python_311 "${candidate}"; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_311_BIN="$(find_python_311 || true)"
  if [[ -z "${PYTHON_311_BIN}" ]]; then
    cat >&2 <<'EOF'
Python 3.11 is required but was not found.
Install it with Homebrew: brew install python@3.11
Then rerun ./scripts/run_local.sh.
EOF
    exit 1
  fi

  echo "Creating Python 3.11 environment at ${VENV_DIR}"
  "${PYTHON_311_BIN}" -m venv "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if ! is_python_311 "${PYTHON_BIN}"; then
  echo "${VENV_DIR} is not a Python 3.11 environment. Set CI_SIGNAL_VENV to a Python 3.11 virtual environment." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import streamlit, pandas, sklearn' >/dev/null 2>&1; then
  echo "Installing project dependencies..."
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements.txt"
fi

HEALTH_URL="http://${HOST}:${PORT}/_stcore/health"
if [[ "${MODE}" == "check" ]]; then
  if curl --fail --silent --show-error --max-time 10 "${HEALTH_URL}" | grep -qx 'ok'; then
    echo "CI Signal is healthy at http://${HOST}:${PORT}"
    exit 0
  fi
  echo "The environment is ready, but CI Signal is not responding at ${HEALTH_URL}." >&2
  exit 1
fi

if curl --fail --silent --max-time 2 "${HEALTH_URL}" | grep -qx 'ok'; then
  echo "CI Signal is already running at http://${HOST}:${PORT}"
  exit 0
fi

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use by another process. Stop it or choose another port with CI_SIGNAL_PORT." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
echo "Starting CI Signal at http://${HOST}:${PORT}"
exec "${PYTHON_BIN}" -m streamlit run app.py \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  --server.headless true
