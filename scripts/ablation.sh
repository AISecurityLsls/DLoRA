#!/usr/bin/env bash
set -Eeuo pipefail

# Three-part DormantLoRA ablation:
#   no_anti_backdoor  - remove the Anti-Backdoor Training component
#   no_backdoor       - remove the Backdoor Training component
#   no_preservation   - remove Target Adapter Capability Preservation (KD)
#
# Usage:
#   CUDA_ID=0 bash scripts/ablation.sh <dormant_dataset> <target_dataset> [ablation_mode ...]

if [[ $# -lt 2 ]]; then
  echo "Usage: CUDA_ID=<gpu_index> bash scripts/ablation.sh <dormant_dataset> <target_dataset> [ablation_mode ...]" >&2
  exit 1
fi

DORMANT_MODEL_DATASET="$1"
TARGET_MODEL_DATASET="$2"
shift 2

if [[ $# -eq 0 ]]; then
  MODES=(no_anti_backdoor no_backdoor no_preservation)
else
  MODES=("$@")
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
BATCH_SIZE="${BATCH_SIZE:-8}"

for mode in "${MODES[@]}"; do
  case "${mode}" in
    no_anti_backdoor)
      echo "[ABLATION] ${mode}: remove Anti-Backdoor Training."
      SOLO_CLEAN_N="${BATCH_SIZE}" \
      SOLO_TRIGGERED_CLEAN_N=0 \
      LAMBDA_INV=0 \
      PARAM_GROUP_PATH="ablation_${mode}" \
      bash "${PROJECT_ROOT}/scripts/run_train_dormant_lora.sh" \
        "${DORMANT_MODEL_DATASET}" "${TARGET_MODEL_DATASET}"
      ;;
    no_backdoor)
      echo "[ABLATION] ${mode}: remove Backdoor Training."
      JOINT_CLEAN_N="${BATCH_SIZE}" \
      JOINT_POISON_N=0 \
      LAMBDA_POISON=0 \
      PARAM_GROUP_PATH="ablation_${mode}" \
      bash "${PROJECT_ROOT}/scripts/run_train_dormant_lora.sh" \
        "${DORMANT_MODEL_DATASET}" "${TARGET_MODEL_DATASET}"
      ;;
    no_preservation)
      echo "[ABLATION] ${mode}: remove Target Adapter Capability Preservation (KD)."
      DISABLE_DORMANT_MODEL_KD=1 \
      KD_SOURCE=none \
      LAMBDA_DORMANT_MODEL_KD=0 \
      PARAM_GROUP_PATH="ablation_${mode}" \
      bash "${PROJECT_ROOT}/scripts/run_train_dormant_lora.sh" \
        "${DORMANT_MODEL_DATASET}" "${TARGET_MODEL_DATASET}"
      ;;
    *)
      echo "[ERROR] Unknown ablation mode '${mode}'. Expected no_anti_backdoor, no_backdoor, or no_preservation." >&2
      exit 1
      ;;
  esac
done
