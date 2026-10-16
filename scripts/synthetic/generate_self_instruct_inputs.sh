#!/usr/bin/env bash
set -Eeuo pipefail

# Generate Self-Instruct synthetic KD inputs for one Target dataset.
#
# Usage:
#   LLM_API_KEY=... LLM_MODEL=... bash scripts/synthetic/generate_self_instruct_inputs.sh <dataset>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/synthetic/generate_self_instruct_inputs.sh <dataset> [extra python args...]" >&2
  exit 1
fi

TARGET_DATASET="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/codes/generate_self_instruct_inputs.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

OUTPUT_ROOT="${SYNTHETIC_OUTPUT_ROOT:-${PROJECT_ROOT}/synthetic_kd_inputs/self_instruct}"
SYNTHETIC_TARGET_SIZE="${SYNTHETIC_TARGET_SIZE:-2048}"

cmd=(
  env PYTHONPATH="${PROJECT_ROOT}/codes${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  "${TARGET_DATASET}"
  --output_root "${OUTPUT_ROOT}"
  --synthetic_target_size "${SYNTHETIC_TARGET_SIZE}"
)
if [[ -n "${LLM_API_KEY:-}" ]]; then
  cmd+=(--llm_api_key "${LLM_API_KEY}")
fi
if [[ -n "${LLM_MODEL:-}" ]]; then
  cmd+=(--llm_model "${LLM_MODEL}")
fi
if [[ -n "${LLM_BASE_URL:-}" ]]; then
  cmd+=(--llm_base_url "${LLM_BASE_URL}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry_run)
fi
if [[ "${SYNTHETIC_RESET_CHECKPOINT:-0}" == "1" ]]; then
  cmd+=(--reset_checkpoint)
fi

cmd+=("$@")
printf '%s\n' "[CMD] ${cmd[*]}"
exec "${cmd[@]}"
