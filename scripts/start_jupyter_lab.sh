#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JUPYTER_CONFIG_DIR="${ROOT}/.jupyter"

cd "${ROOT}"
exec uv run jupyter lab "$@"
