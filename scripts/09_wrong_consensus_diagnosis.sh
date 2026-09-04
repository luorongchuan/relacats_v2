#!/usr/bin/env bash
# Identify original-CaTS high-SSC wrong cases, generate relational witnesses,
# and compare SSC(wrong) against RelSSC(wrong).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MODEL_NAME="${MODEL_NAME:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"
ORIGINAL_CATS_FILE="${ORIGINAL_CATS_FILE:-}"
DATASET_NAME="${DATASET_NAME:-}"
THRESHOLD="${THRESHOLD:-0.9}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/relacats_v2/outputs/diagnosis/${DATASET_NAME:-unset_dataset}}"
RELATIONAL_ROOT="${RELATIONAL_ROOT:-${OUTPUT_DIR}/relational_raw}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-2}"
SEED="${SEED:-42}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
# Ensure vLLM's JIT helper binaries (for example ninja) are discoverable.
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
[[ -n "${ORIGINAL_CATS_FILE}" ]] || fail \
  "set ORIGINAL_CATS_FILE to an original CaTS generation JSON/JSONL artifact"
[[ -f "${ORIGINAL_CATS_FILE}" || -d "${ORIGINAL_CATS_FILE}" ]] || fail \
  "original CaTS artifact not found: ${ORIGINAL_CATS_FILE}"
[[ -n "${DATASET_NAME}" ]] || fail \
  "set DATASET_NAME (for example arc_challenge or math_qa)"

mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}/.diagnosis.lock"
flock -n 9 || fail "another diagnosis process is using ${OUTPUT_DIR}"

CANDIDATES_FILE="${OUTPUT_DIR}/candidates.jsonl"
IDENTIFICATION_SUMMARY="${OUTPUT_DIR}/identification_summary.json"
CASES_FILE="${OUTPUT_DIR}/cases.jsonl"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"
SUMMARY_MD="${OUTPUT_DIR}/summary.md"
GENERATION_LOG="${OUTPUT_DIR}/generation_gpu67.log"

echo "[1/3] Identifying original CaTS cases with SSC > ${THRESHOLD} and wrong consensus"
"${PYTHON_BIN}" -m relacats_v2.diagnosis.wrong_consensus_diagnosis identify \
  --original-cats-file "${ORIGINAL_CATS_FILE}" \
  --dataset-name "${DATASET_NAME}" \
  --threshold "${THRESHOLD}" \
  --candidates-output "${CANDIDATES_FILE}" \
  --summary-output "${IDENTIFICATION_SUMMARY}"

CANDIDATE_COUNT="$(wc -l < "${CANDIDATES_FILE}")"
if (( CANDIDATE_COUNT > 0 )); then
  if [[ "${SKIP_GENERATION}" == "1" ]]; then
    [[ -d "${RELATIONAL_ROOT}" || -f "${RELATIONAL_ROOT}" ]] || fail \
      "SKIP_GENERATION=1 but relational raw is missing: ${RELATIONAL_ROOT}"
    echo "[2/3] Reusing relational raw: ${RELATIONAL_ROOT}"
  else
    [[ -f "${MODEL_NAME}/config.json" ]] || fail "Local model not found: ${MODEL_NAME}"
    if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
      for gpu in 6 7; do
        pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
        [[ -z "${pids}" ]] || fail \
          "physical GPU ${gpu} is busy (PID(s): ${pids}); wait or set ALLOW_BUSY_GPUS=1 intentionally"
      done
    fi
    mkdir -p "${RELATIONAL_ROOT}"
    export CUDA_VISIBLE_DEVICES=6,7
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    echo "[2/3] Generating/resuming ${CANDIDATE_COUNT} relational witnesses on physical GPUs 6,7"
    "${PYTHON_BIN}" -m relacats_v2.data_creation.generate_relational_data \
      --candidate-file "${CANDIDATES_FILE}" \
      --model-name "${MODEL_NAME}" \
      --output-root "${RELATIONAL_ROOT}" \
      --max-questions 1000000 \
      --num-views 4 \
      --samples-per-view 8 \
      --total-budget 32 \
      --temperature 0.8 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --question-batch-size "${QUESTION_BATCH_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size 2 \
      --seed "${SEED}" \
      --local-files-only \
      2>&1 | tee "${GENERATION_LOG}"
  fi
else
  echo "[2/3] No generation-ready high-SSC wrong cases; GPU generation is unnecessary"
fi

echo "[3/3] Computing gold-free RelSSC and diagnostic reductions"
"${PYTHON_BIN}" -m relacats_v2.diagnosis.wrong_consensus_diagnosis compare \
  --candidates-file "${CANDIDATES_FILE}" \
  --relational-root "${RELATIONAL_ROOT}" \
  --cases-output "${CASES_FILE}" \
  --summary-output "${SUMMARY_JSON}" \
  --markdown-output "${SUMMARY_MD}"

echo "Diagnosis complete: ${SUMMARY_MD}"
