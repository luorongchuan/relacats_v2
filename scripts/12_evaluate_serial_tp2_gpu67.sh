#!/usr/bin/env bash
set -Eeuo pipefail

# One reusable evaluation entry point for both the released Self-Calibration
# checkpoints and the later RelaCaTS checkpoints.  Each model is processed in
# this order, synchronously:
#
#   response pool (one TP worker on GPU 6+7)
#       -> confidence (one TP worker on GPU 6+7)
#       -> CPU aggregation (08_evaluate.sh)
#
# The next model is not started until all workers from the previous model have
# exited.  Set MODEL_SPECS to a whitespace-separated list of
# ``tag=/absolute/model/path`` pairs to reuse this script for another model
# set.  Paths containing spaces are intentionally not supported by this small
# shell interface; all project checkpoint paths are space-free.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

# tensor_parallel is deliberately the default here: one model process sees
# both physical GPUs.  The lower utilization reserves roughly 68 GB of an
# A100-80G per GPU for vLLM's model/KV cache; it is configurable and is not a
# promise that every model will consume exactly that amount.
EVAL_GPU_MODE="${EVAL_GPU_MODE:-tensor_parallel}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

DATASET_LIST="${DATASETS:-object_counting math_qa arc_challenge}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-8}"
CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE:-128}"
SEED="${SEED:-42}"

# Table-2 aggregation settings.  These are passed to the same 08 evaluator
# used for later RelaCaTS runs, including CISC/Self-Certainty/RASC rows.
BUDGETS="${BUDGETS:-1,2,4,8,16,32}"
CURVE_MAX_BUDGET="${CURVE_MAX_BUDGET:-32}"
BUDGET_TARGETS="${BUDGET_TARGETS:-16}"
DYNAMIC_MIN_VALID="${DYNAMIC_MIN_VALID:-2}"
RASC_BUFFER_SIZE="${RASC_BUFFER_SIZE:-5}"
ESC_WINDOW_SIZES="${ESC_WINDOW_SIZES:-}"
CISC_TEMPERATURE="${CISC_TEMPERATURE:-1.0}"
CISC_NORMALIZATION="${CISC_NORMALIZATION:-softmax}"
EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS:-}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_serial_tp2_v2}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/eval_serial_tp2_v2}"
CACHE_ROOT="${CACHE_ROOT:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
EVAL_PHASE="${EVAL_PHASE:-test}"
EVAL_SPLIT="${EVAL_SPLIT:-${EVAL_PHASE}}"

# Defaults are the three released author checkpoints.  For later experiments
# override MODEL_SPECS and use a new BASE_OUTPUT_ROOT for each protocol/model
# family so response pools cannot be mixed accidentally.
MODEL_SPECS="${MODEL_SPECS:-qwen2_5_7b_instruct=/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct-Self-Calibration llama3_1_8b_instruct=/home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct-Self-Calibration deepseek_r1_distill_qwen_1_5b=/home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ ]] || fail "GPU_FIRST must be an integer"
[[ "${GPU_SECOND}" =~ ^[0-9]+$ ]] || fail "GPU_SECOND must be an integer"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${TENSOR_PARALLEL_SIZE}" == "2" ]] || fail "TENSOR_PARALLEL_SIZE must be 2 for the two-GPU serial runner"
[[ "${NUM_SHARDS}" == "1" ]] || fail "NUM_SHARDS must be 1 for tensor-parallel mode"
[[ "${EVAL_GPU_MODE}" == "tensor_parallel" ]] || fail "EVAL_GPU_MODE must be tensor_parallel in this runner"
[[ "${EVAL_PHASE}" == "validation" || "${EVAL_PHASE}" == "test" ]] || fail "EVAL_PHASE must be validation or test"
[[ "${NUM_GENERATIONS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GENERATIONS must be a positive integer"
(( NUM_GENERATIONS >= 16 )) || fail "NUM_GENERATIONS must be at least 16 for Table 2"

read -r -a DATASET_ARRAY <<< "${DATASET_LIST}"
(( ${#DATASET_ARRAY[@]} > 0 )) || fail "DATASETS must not be empty"
read -r -a MODEL_SPEC_ARRAY <<< "${MODEL_SPECS}"
(( ${#MODEL_SPEC_ARRAY[@]} > 0 )) || fail "MODEL_SPECS must not be empty"

mkdir -p "${BASE_OUTPUT_ROOT}" "${BASE_LOG_ROOT}" "${CACHE_ROOT}"
exec 9>"${BASE_OUTPUT_ROOT}/.serial_eval.lock"
flock -n 9 || fail "another serial evaluation is using ${BASE_OUTPUT_ROOT}"

# Keep a persistent top-level launcher log even when this script is started
# from an interactive shell and the child stage logs are redirected elsewhere.
exec > >(tee -a "${BASE_LOG_ROOT}/launcher.log") 2>&1

check_gpu_idle() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    return 0
  fi
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" \
    || fail "unable to query GPU ${gpu}; refusing to start safely"
  [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
}

wait_gpu_idle() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    return 0
  fi
  local attempt gpu pids
  for attempt in $(seq 1 60); do
    local busy=0
    for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
      pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" \
        || fail "unable to query GPU ${gpu} while waiting for worker shutdown"
      [[ -z "${pids}" ]] || busy=1
    done
    if (( busy == 0 )); then
      return 0
    fi
    sleep 1
  done
  fail "GPU worker process did not release ${GPU_FIRST},${GPU_SECOND} within 60 seconds"
}

check_gpu_idle "${GPU_FIRST}"
check_gpu_idle "${GPU_SECOND}"

echo "RelaCaTS-v2 serial evaluation starting"
echo "  mode=${EVAL_GPU_MODE}; tensor_parallel_size=${TENSOR_PARALLEL_SIZE}; visible_gpus=${GPU_FIRST},${GPU_SECOND}"
echo "  datasets=${DATASET_LIST}"
echo "  candidate_pool=${NUM_GENERATIONS}; target_average_budget=${BUDGET_TARGETS}"
echo "  output_root=${BASE_OUTPUT_ROOT}"
echo "  methods=SC/CISC/Self-Certainty/Best-of-N/ASC/ESC/RASC + RelaCaTS-SC/ES/ASC"

for spec in "${MODEL_SPEC_ARRAY[@]}"; do
  [[ "${spec}" == *=* ]] || fail "MODEL_SPECS entry must be tag=/path: ${spec}"
  tag="${spec%%=*}"
  model_path="${spec#*=}"
  [[ "${tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail "invalid model tag: ${tag}"
  [[ -n "${model_path}" ]] || fail "empty model path for ${tag}"
  [[ -f "${model_path}/config.json" ]] || fail "model config not found for ${tag}: ${model_path}/config.json"

  model_root="${BASE_OUTPUT_ROOT}/${tag}"
  response_root="${model_root}/responses/${EVAL_PHASE}"
  confidence_root="${model_root}/confidence/${EVAL_PHASE}"
  result_root="${model_root}/results/${EVAL_PHASE}"
  threshold_root="${model_root}/thresholds"
  model_log_root="${BASE_LOG_ROOT}/${tag}"
  mkdir -p "${model_log_root}"

  model_lower="${model_path,,}"
  case "${model_lower}" in
    *deepseek*) model_family="deepseek" ;;
    *llama*) model_family="llama" ;;
    *qwen*) model_family="qwen" ;;
    *) model_family="" ;;
  esac
  if [[ -n "${MAX_TOKENS}" ]]; then
    model_max_tokens="${MAX_TOKENS}"
  elif [[ "${model_family}" == "deepseek" ]]; then
    model_max_tokens=2048
  else
    model_max_tokens=1024
  fi

  echo
  echo "===== START ${tag} (single model on GPUs ${GPU_FIRST}+${GPU_SECOND}) ====="
  echo "model=${model_path}"

  # Stage 1: one TP=2 response worker.  This calls the same 06 script used by
  # all later experiments; only its execution mode and shard count differ.
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL="${model_path}" \
    OUTPUT_ROOT="${response_root}" \
    LOG_ROOT="${model_log_root}/generation" \
    CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" \
    GPU_SECOND="${GPU_SECOND}" \
    CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE="tensor_parallel" \
    TENSOR_PARALLEL_SIZE="2" \
    NUM_SHARDS="1" \
    DATASETS="${DATASET_LIST}" \
    SPLIT="${EVAL_SPLIT}" \
    NUM_GENERATIONS="${NUM_GENERATIONS}" \
    TEMPERATURE="${TEMPERATURE}" \
    MAX_TOKENS="${model_max_tokens}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/06_generate_eval.sh"
  wait_gpu_idle

  # Stage 2: one TP=2 confidence worker over the same one-shard response pool.
  # The model path is intentionally identical to Stage 1; this preserves the
  # current confidence artifact contract and prevents accidental cross-model
  # scoring.  Native CISC/Self-Certainty/RASC fields, when supplied by a richer
  # producer in the future, are consumed by the same 08 aggregator.
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL="${model_path}" \
    RESPONSES_ROOT="${response_root}" \
    OUTPUT_ROOT="${confidence_root}" \
    LOG_ROOT="${model_log_root}/confidence" \
    CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" \
    GPU_SECOND="${GPU_SECOND}" \
    CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE="tensor_parallel" \
    TENSOR_PARALLEL_SIZE="2" \
    NUM_SHARDS="1" \
    MODEL_FAMILY="${model_family}" \
    DATASETS="${DATASET_LIST}" \
    CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/07_calculate_confidence.sh"
  wait_gpu_idle

  # Stage 3 is CPU-only and writes all currently registered methods, including
  # CISC, Self-Certainty, ESC, and RASC plus the three RelaCaTS rows.
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    INPUT_ROOT="${confidence_root}" \
    OUTPUT_ROOT="${result_root}" \
    PHASE="${EVAL_PHASE}" \
    MODEL_ID="${tag}" \
    THRESHOLD_ROOT="${threshold_root}" \
    DATASETS="${DATASET_LIST}" \
    NUM_SHARDS="1" \
    BUDGETS="${BUDGETS}" \
    CURVE_MAX_BUDGET="${CURVE_MAX_BUDGET}" \
    BUDGET_TARGETS="${BUDGET_TARGETS}" \
    DYNAMIC_MIN_VALID="${DYNAMIC_MIN_VALID}" \
    RASC_BUFFER_SIZE="${RASC_BUFFER_SIZE}" \
    ESC_WINDOW_SIZES="${ESC_WINDOW_SIZES}" \
    CISC_TEMPERATURE="${CISC_TEMPERATURE}" \
    CISC_NORMALIZATION="${CISC_NORMALIZATION}" \
    EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/08_evaluate.sh"

  echo "===== COMPLETE ${tag}: ${result_root} ====="
done

echo
echo "ALL SERIAL RELACATS-V2 ${EVAL_PHASE^^} RUNS COMPLETE"
echo "Results root: ${BASE_OUTPUT_ROOT}"
