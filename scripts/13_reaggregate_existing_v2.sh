#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${SOURCE_ROOT}" ]] || { echo "Source results not found: ${SOURCE_ROOT}" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "Refusing to overwrite existing v2 results: ${OUTPUT_ROOT}" >&2
  exit 1
}

export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" -m relacats_v2.evaluation.reaggregate_existing \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --validation-fraction "${VALIDATION_FRACTION:-0.2}" \
  --seed "${SEED:-42}" \
  --target-budget "${TARGET_BUDGET:-16}"
