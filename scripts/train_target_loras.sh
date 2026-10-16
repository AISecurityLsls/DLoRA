#!/usr/bin/env bash
set -Eeuo pipefail

# Train a bank of clean Target LoRAs.
#
# Override with, for example:
#   BASE_MODEL_LIST="Llama-3-8B" TARGET_DATASET_LIST="agnews mnli multirc imdb" \
#   CUDA_ID=0 bash scripts/train_target_loras.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
BASE_MODEL_LIST="${BASE_MODEL_LIST:-Llama-3-8B}"
TARGET_DATASET_LIST="${TARGET_DATASET_LIST:-agnews mnli multirc imdb}"

for base in ${BASE_MODEL_LIST}; do
  for dataset in ${TARGET_DATASET_LIST}; do
    echo "[TRAIN TARGET] base=${base} dataset=${dataset}"
    BASE_MODEL_NAME="${base}" \
    OVERWRITE_TARGET_ADAPTERS="${OVERWRITE_TARGET_ADAPTERS:-0}" \
    bash "${PROJECT_ROOT}/scripts/run_train_target_lora.sh" "${dataset}"
  done
done
