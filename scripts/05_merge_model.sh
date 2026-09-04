#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
BASE_MODEL="${BASE_MODEL:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"
LORA_PATH="${LORA_PATH:-${ROOT_DIR}/relacats_v2/outputs/checkpoints/qwen2_5_7b_instruct_relacats_v2}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_instruct_relacats_v2}"

[[ -f "${LORA_PATH}/adapter_config.json" ]] || {
  echo "LoRA adapter not found: ${LORA_PATH}" >&2
  exit 1
}
"${PYTHON_BIN}" -m relacats_v2.model_training.merge_lora \
  --base-model "${BASE_MODEL}" \
  --lora-path "${LORA_PATH}" \
  --output-path "${OUTPUT_PATH}"
