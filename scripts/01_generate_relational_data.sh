#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MODEL_NAME="${MODEL_NAME:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/generated_data}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/data_generation}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"
case "${MODEL_NAME,,}" in
  *deepseek*) MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}" ;;
  *) MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}" ;;
esac
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.8}"
CONFIDENCE_TEMPERATURE="${CONFIDENCE_TEMPERATURE:-0.0}"
# Keep this default equal to the nine-task CaTS training mixture.  Evaluation
# still supports the legacy ARC-Challenge/MathQA entries, but they should not
# silently replace GSM8K/SVAMP/SciQ/WinoGrande during teacher-data generation.
DATASETS=( ${DATASETS:-arc_easy commonsense_qa gsm8k logiqa openbookqa reclor sciq svamp winogrande} )

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
# vLLM/FlashInfer may invoke helper binaries (notably ninja) from the
# selected environment.  Calling an absolute Python does not automatically
# put its bin directory on PATH, so do that explicitly for child processes.
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
[[ -f "${MODEL_NAME}/config.json" ]] || fail "Local model not found: ${MODEL_NAME}"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${MAX_QUESTIONS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_QUESTIONS must be positive"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
exec 9>"${OUTPUT_ROOT}/.generation.lock"
flock -n 9 || fail "another relational-data generator is using ${OUTPUT_ROOT}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --datasets "${DATASETS[@]}"
  --split train
  --max-questions "${MAX_QUESTIONS}"
  --output-root "${OUTPUT_ROOT}"
  --num-views 4
  --samples-per-view 8
  --total-budget 32
  --relation-mode auto
  --temperature "${TEMPERATURE}"
  --confidence-temperature "${CONFIDENCE_TEMPERATURE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --question-batch-size "${QUESTION_BATCH_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size 1
  --seed "${SEED}"
  --num-shards 2
)

declare -a PIDS=()
terminate_workers() {
  (( ${#PIDS[@]} > 0 )) || return 0
  for pid in "${PIDS[@]:-}"; do
    # Workers are started with setsid below, so the PID is also the process
    # group leader.  Killing the group prevents a vLLM child/server from
    # surviving an interrupted generation run and retaining GPU memory.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
}
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( status != 0 )); then
    terminate_workers
  fi
  if (( ${#PIDS[@]} > 0 )); then
    for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "RelaCaTS-v2 generation: physical GPUs ${GPU_FIRST},${GPU_SECOND};"
echo "  MCQ profile=4 views x 8 responses; WinoGrande=2 unique views x 16; GSM8K/SVAMP=1 identity view x 32"
setsid env CUDA_VISIBLE_DEVICES="${GPU_FIRST}" "${PYTHON_BIN}" -m relacats_v2.data_creation.generate_relational_data \
  "${COMMON_ARGS[@]}" --shard-index 0 >"${LOG_ROOT}/shard0_gpu${GPU_FIRST}.log" 2>&1 &
PIDS+=("$!")
setsid env CUDA_VISIBLE_DEVICES="${GPU_SECOND}" "${PYTHON_BIN}" -m relacats_v2.data_creation.generate_relational_data \
  "${COMMON_ARGS[@]}" --shard-index 1 >"${LOG_ROOT}/shard1_gpu${GPU_SECOND}.log" 2>&1 &
PIDS+=("$!")

set +e
wait "${PIDS[0]}"; first_status=$?
wait "${PIDS[1]}"; second_status=$?
set -e
(( first_status == 0 && second_status == 0 )) || fail \
  "generation worker failed: shard0=${first_status}, shard1=${second_status}; see ${LOG_ROOT}"
PIDS=()
echo "Relational teacher data complete: ${OUTPUT_ROOT}"
