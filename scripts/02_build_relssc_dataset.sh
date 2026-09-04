#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/generated_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/relssc_dataset}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" -m relacats_v2.data_creation.build_relssc_dataset \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --test-ratio "${TEST_RATIO:-0.1}" \
  --seed "${SEED}"
