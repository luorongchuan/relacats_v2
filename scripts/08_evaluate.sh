#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/confidence}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/results}"
PHASE="${PHASE:-test}"
MODEL_ID="${MODEL_ID:-}"
THRESHOLD_ROOT="${THRESHOLD_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/thresholds/${MODEL_ID:-unset_model}}"
DATASETS=( ${DATASETS:-object_counting math_qa arc_challenge} )
# The normal two-worker launcher writes two confidence shards.  The serial
# tensor-parallel launcher writes one shard because one process owns both GPUs.
# Keep the check configurable so both artifact layouts use the same CPU
# evaluator and cannot be silently mixed.
NUM_SHARDS="${NUM_SHARDS:-2}"
# Aggregation knobs mirror the CPU evaluator CLI.  Defaults reproduce the
# paper's sample-budget-16 report while retaining the full 32-response pool
# for curves.  Keeping them environment-configurable is useful for ablations
# (especially ESC windows and the RASC buffer) without editing this launcher.
BUDGETS="${BUDGETS:-1,2,4,8,16,32}"
THRESHOLDS="${THRESHOLDS:-}"
CURVE_MAX_BUDGET="${CURVE_MAX_BUDGET:-32}"
BUDGET_TARGETS="${BUDGET_TARGETS:-16}"
DYNAMIC_MIN_VALID="${DYNAMIC_MIN_VALID:-2}"
RASC_BUFFER_SIZE="${RASC_BUFFER_SIZE:-5}"
ESC_WINDOW_SIZES="${ESC_WINDOW_SIZES:-}"
CISC_TEMPERATURE="${CISC_TEMPERATURE:-1.0}"
CISC_NORMALIZATION="${CISC_NORMALIZATION:-softmax}"
EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
[[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_SHARDS must be a positive integer"
[[ "${PHASE}" == "validation" || "${PHASE}" == "test" || "${PHASE}" == "paper" ]] || fail \
  "PHASE must be validation, test, or paper"
[[ -n "${MODEL_ID}" ]] || fail "MODEL_ID is required to scope evaluation outputs"
mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.evaluation.lock"
flock -n 9 || fail "another evaluation report is using ${OUTPUT_ROOT}"

for dataset in "${DATASETS[@]}"; do
  input_dir="${INPUT_ROOT}/${dataset}"
  [[ -d "${input_dir}" ]] || fail \
    "Confidence artifact missing for ${dataset}; run 07_calculate_confidence.sh"

  # Check every expected confidence shard before aggregation so interrupted
  # workers cannot silently remove questions from the denominator.
  for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
    shard_dir="${input_dir}/shard-$(printf '%05d' "${shard_index}")-of-$(printf '%05d' "${NUM_SHARDS}")"
    [[ -f "${shard_dir}/confidence.jsonl" ]] || fail \
      "Confidence shard missing for ${dataset}: ${shard_dir}"
    [[ -f "${shard_dir}/confidence_metadata.json" ]] || fail \
      "Confidence metadata missing for ${dataset}: ${shard_dir}"
    [[ -f "${shard_dir}/confidence_manifest.json" ]] || fail \
      "Confidence manifest missing for ${dataset}: ${shard_dir}"
    "${PYTHON_BIN}" - "${shard_dir}/confidence_manifest.json" <<'PY' || fail "Incomplete confidence manifest: ${shard_dir}"
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("complete") is not True:
    raise SystemExit(1)
if manifest.get("expected_samples") != manifest.get("samples"):
    raise SystemExit(1)
if manifest.get("expected_questions") is not None and manifest.get("expected_questions") != manifest.get("questions"):
    raise SystemExit(1)
PY
  done

  common_args=(
    --input "${input_dir}"
    --output-dir "${OUTPUT_ROOT}/${dataset}"
    --model-id "${MODEL_ID}"
    --dataset-name "${dataset}"
    --budgets "${BUDGETS}"
    --curve-max-budget "${CURVE_MAX_BUDGET}"
    --budget-targets "${BUDGET_TARGETS}"
    --dynamic-min-valid "${DYNAMIC_MIN_VALID}"
    --rasc-buffer-size "${RASC_BUFFER_SIZE}"
    --cisc-temperature "${CISC_TEMPERATURE}"
    --cisc-normalization "${CISC_NORMALIZATION}"
  )
  if [[ -n "${THRESHOLDS}" ]]; then
    common_args+=(--thresholds "${THRESHOLDS}")
  fi
  if [[ -n "${ESC_WINDOW_SIZES}" ]]; then
    common_args+=(--esc-window-sizes "${ESC_WINDOW_SIZES}")
  fi
  if [[ -n "${EXPECTED_QUESTIONS}" ]]; then
    common_args+=(--expected-questions "${EXPECTED_QUESTIONS}")
  fi

  if [[ "${PHASE}" == "paper" ]]; then
    # Original-CaTS/Table-2 style protocol: use the full test response pool and
    # choose dynamic controls by average sample cost only.  No threshold file
    # is read or written, and accuracy is excluded from parameter selection.
    "${PYTHON_BIN}" -m relacats_v2.evaluation.paper_budget "${common_args[@]}"
  else
    aggregate_args=(
      "${common_args[@]}"
      --phase "${PHASE}"
      --threshold-file "${THRESHOLD_ROOT}/${dataset}.json"
    )
    "${PYTHON_BIN}" -m relacats_v2.evaluation.aggregate "${aggregate_args[@]}"
  fi
done

echo "RelaCaTS-v2 ${PHASE} reports (SC/CISC/Self-Certainty/Best-of-N/ASC/ESC/RASC + RelaCaTS-SC/RelaCaTS-ES/RelaCaTS-ASC): ${OUTPUT_ROOT}"
