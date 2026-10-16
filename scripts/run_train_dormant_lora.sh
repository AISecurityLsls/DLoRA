#!/usr/bin/env bash
set -Eeuo pipefail

# Train one Dormant LoRA with an existing frozen Target LoRA.
#
# Usage:
#   CUDA_ID=0 bash scripts/run_train_dormant_lora.sh <dormant_dataset> <target_dataset> [extra python args...]

usage() {
  cat <<'USAGE'
Usage:
  CUDA_ID=<gpu_index> bash scripts/run_train_dormant_lora.sh <dormant_dataset> <target_dataset> [extra python args...]

Examples:
  CUDA_ID=0 bash scripts/run_train_dormant_lora.sh mnli agnews
  CUDA_ID=0 PATH_CONFIG=configs/paths.local.json bash scripts/run_train_dormant_lora.sh mnli agnews --eval_limit 300

Common environment variables:
  BASE_MODEL_NAME               default: Llama-3-8B
  BASE_MODEL_PATH               optional explicit base-model path
  PATH_CONFIG                   optional JSON path config
  TARGET_MODEL_ADAPTER_PATH     optional explicit Target LoRA adapter path
  KD_SOURCE              synthetic_self_instruct | target_train | target_valid | none
  STAGE2_STEPS                  default: 600
  STAGE3_STEPS                  default: 400
  EVAL_LIMIT                    default: 300
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

DORMANT_MODEL_DATASET="$1"
TARGET_MODEL_DATASET="$2"
shift 2
EXTRA_PY_ARGS=("$@")

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/codes/train_dormant_lora.py}"
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
CROSS_TARGET_ADAPTER_ROOT="${CROSS_TARGET_ADAPTER_ROOT:-${TARGET_ADAPTER_ROOT}}"
DORMANT_ADAPTER_OUTPUT_ROOT="${DORMANT_ADAPTER_OUTPUT_ROOT:-${PROJECT_ROOT}/adapters/DormantModelAdapters}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"

pair_name="${DORMANT_MODEL_DATASET}-with-${TARGET_MODEL_DATASET}"
PARAM_GROUP_PATH="${PARAM_GROUP_PATH:-}"
if [[ -n "${PARAM_GROUP_PATH}" ]]; then
  output_dir="${DORMANT_ADAPTER_OUTPUT_ROOT}/${BASE_MODEL_NAME}/${PARAM_GROUP_PATH}/lora_${pair_name}"
  log_dir="${LOG_ROOT}/${BASE_MODEL_NAME}/${PARAM_GROUP_PATH}/${pair_name}"
else
  output_dir="${DORMANT_ADAPTER_OUTPUT_ROOT}/${BASE_MODEL_NAME}/lora_${pair_name}"
  log_dir="${LOG_ROOT}/${BASE_MODEL_NAME}/${pair_name}"
fi

cmd=(
  env PYTHONPATH="${PROJECT_ROOT}/codes${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --dormant_model_dataset "${DORMANT_MODEL_DATASET}"
  --target_model_dataset "${TARGET_MODEL_DATASET}"
  --base_model_name "${BASE_MODEL_NAME}"
  --output_dir "${output_dir}"
  --log_dir "${log_dir}"
)
if [[ -n "${BASE_MODEL_PATH}" ]]; then
  cmd+=(--base_model_path "${BASE_MODEL_PATH}")
fi
if [[ -n "${PATH_CONFIG}" ]]; then
  cmd+=(--path_config "${PATH_CONFIG}")
else
  cmd+=(--data_root "${DATA_ROOT}" --target_adapter_root "${TARGET_ADAPTER_ROOT}" --cross_target_adapter_root "${CROSS_TARGET_ADAPTER_ROOT}" --dormant_adapter_output_root "${DORMANT_ADAPTER_OUTPUT_ROOT}")
fi
if [[ -n "${TARGET_MODEL_ADAPTER_PATH:-}" ]]; then
  cmd+=(--target_model_adapter_path "${TARGET_MODEL_ADAPTER_PATH}")
fi

for name in \
  KD_SOURCE KD_INPUT_PATH KD_INPUT_LIMIT SYNTHETIC_INPUT_MANIFEST \
  DORMANT_MODEL_KD_BATCH_SIZE DORMANT_MODEL_KD_CLEAN_N DORMANT_MODEL_KD_TRIGGERED_N \
  LAMBDA_DORMANT_MODEL_KD DISABLE_DORMANT_MODEL_KD PRESERVATION_KD_MODE GENERATIVE_KD_TOP_K \
  STAGE1_STEPS STAGE2_STEPS STAGE3_STEPS LEARNING_RATE MIN_LEARNING_RATE NUM_WARMUP_STEPS \
  LR_SCHEDULER_TYPE MAX_GRAD_NORM GRADIENT_CHECKPOINTING \
  POISON_METHOD POISON_VARIANT ATTACK_DATA_ROOT TRIGGER TRIGGER_POSITION TARGET_OUTPUT \
  BATCH_SIZE SOLO_CLEAN_N SOLO_TRIGGERED_CLEAN_N JOINT_CLEAN_N JOINT_POISON_N \
  LAMBDA_SOLO LAMBDA_JOINT LAMBDA_SOLO_CLEAN LAMBDA_INV LAMBDA_JOINT_CLEAN LAMBDA_POISON \
  ENABLE_BIND_FLAT BIND_FLAT_BETA BIND_FLAT_RHO BIND_FLAT_START_STEP BIND_FLAT_EVERY \
  ENABLE_BIND_SHARP BIND_SHARP_BETA BIND_SHARP_RHO BIND_SHARP_START_STEP BIND_SHARP_EVERY \
  EVAL_LIMIT EVAL_SAMPLE_SEED EVAL_BATCH_SIZE EVAL_MODE MAX_LENGTH LABEL_SCORE_NORMALIZE \
  ALLOW_ANSWER_TRUNCATION FEATURE_CACHE_POLICY SEED SKIP_CROSS_TARGET_EVAL SKIP_LOSS_PLOT \
  LOSS_LOG_EVERY LIVE_LOSS_PLOT_EVERY SEPARATE_LOSS_PANELS SAVE_STAGE2_CHECKPOINT STAGE3_CHECKPOINT_EVERY; do
  if [[ -v "${name}" ]]; then
    value="${!name}"
    case "${name}" in
      ENABLE_BIND_FLAT|ENABLE_BIND_SHARP|DISABLE_DORMANT_MODEL_KD|GRADIENT_CHECKPOINTING|ALLOW_ANSWER_TRUNCATION|SKIP_CROSS_TARGET_EVAL|SKIP_LOSS_PLOT|SAVE_STAGE2_CHECKPOINT)
        if [[ "${value}" == "1" || "${value}" == "true" ]]; then
          cmd+=(--"$(echo "${name}" | tr '[:upper:]' '[:lower:]')")
        fi
        ;;
      *)
        cmd+=(--"$(echo "${name}" | tr '[:upper:]' '[:lower:]')" "${value}")
        ;;
    esac
  fi
done

if [[ -n "${PARAM_GROUP_PATH}" ]]; then
  cmd+=(--param_group_path "${PARAM_GROUP_PATH}")
fi
cmd+=("${EXTRA_PY_ARGS[@]}")

mkdir -p "${log_dir}" "${output_dir}"
printf '%s\n' "[CMD] ${cmd[*]}"
exec "${cmd[@]}"
