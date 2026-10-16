#!/usr/bin/env bash
set -Eeuo pipefail

# Train one clean Target LoRA.
#
# Usage:
#   CUDA_ID=0 bash scripts/run_train_target_lora.sh <target_dataset> [extra python args...]

if [[ $# -lt 1 ]]; then
  echo "Usage: CUDA_ID=<gpu_index> bash scripts/run_train_target_lora.sh <target_dataset> [extra python args...]" >&2
  exit 1
fi

TARGET_MODEL_DATASET="$1"
shift
EXTRA_PY_ARGS=("$@")

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/codes/train_target_lora.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"

if [[ -z "${PATH_CONFIG:-}" && -z "${MERGEAC_PATH_CONFIG:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
PATH_CONFIG="${PATH_CONFIG:-${MERGEAC_PATH_CONFIG:-}}"

CUDA_ID="${CUDA_ID:-}"
if [[ -z "${CUDA_ID}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Set CUDA_ID to a single GPU index." >&2
  exit 1
fi
if [[ -n "${CUDA_ID}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_ID}"
fi
export CUDA_VISIBLE_DEVICES

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Llama-3-8B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/Long_Sequence}"
TARGET_ADAPTER_ROOT="${TARGET_ADAPTER_ROOT:-${PROJECT_ROOT}/adapters/TargetModelAdapters}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/train_target_loras}"

cmd=(
  env PYTHONPATH="${PROJECT_ROOT}/codes${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --target_model_dataset "${TARGET_MODEL_DATASET}"
  --base_model_name "${BASE_MODEL_NAME}"
  --target_adapter_root "${TARGET_ADAPTER_ROOT}"
  --log_dir "${LOG_ROOT}/${BASE_MODEL_NAME}/lora_${TARGET_MODEL_DATASET}"
)
if [[ -n "${BASE_MODEL_PATH}" ]]; then
  cmd+=(--base_model_path "${BASE_MODEL_PATH}")
fi
if [[ -n "${PATH_CONFIG}" ]]; then
  cmd+=(--path_config "${PATH_CONFIG}")
else
  cmd+=(--data_root "${DATA_ROOT}")
fi
if [[ -n "${OVERWRITE_TARGET_ADAPTERS:-}" && "${OVERWRITE_TARGET_ADAPTERS}" == "1" ]]; then
  cmd+=(--overwrite_output_dir)
fi

for name in \
  TRAIN_STEPS BATCH_SIZE LEARNING_RATE WEIGHT_DECAY MAX_GRAD_NORM LR_SCHEDULER_TYPE \
  MIN_LEARNING_RATE NUM_WARMUP_STEPS MAX_LENGTH ALLOW_ANSWER_TRUNCATION FEATURE_CACHE_POLICY \
  EVAL_BATCH_SIZE EVAL_LIMIT EVAL_SAMPLE_SEED EVAL_MODE LABEL_SCORE_NORMALIZE \
  POISON_METHOD POISON_VARIANT ATTACK_DATA_ROOT TRIGGER TRIGGER_POSITION TARGET_OUTPUT \
  GRADIENT_CHECKPOINTING SEED LORA_R LORA_ALPHA LORA_DROPOUT TARGET_MODULES; do
  if [[ -v "${name}" ]]; then
    value="${!name}"
    case "${name}" in
      ALLOW_ANSWER_TRUNCATION|GRADIENT_CHECKPOINTING)
        if [[ "${value}" == "1" || "${value}" == "true" ]]; then
          cmd+=(--"$(echo "${name}" | tr '[:upper:]' '[:lower:]')")
        fi
        ;;
      TARGET_MODULES)
        cmd+=(--target_modules ${value})
        ;;
      *)
        cmd+=(--"$(echo "${name}" | tr '[:upper:]' '[:lower:]')" "${value}")
        ;;
    esac
  fi
done

cmd+=("${EXTRA_PY_ARGS[@]}")
mkdir -p "${LOG_ROOT}/${BASE_MODEL_NAME}/lora_${TARGET_MODEL_DATASET}"
printf '%s\n' "[CMD] ${cmd[*]}"
exec "${cmd[@]}"
