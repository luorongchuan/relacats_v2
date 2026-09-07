#!/usr/bin/env bash
set -Eeuo pipefail

# Resume-only paired Qwen evaluation after training/merge has completed.
#
# Why this runner exists:
#   The TP=2 vLLM path can fail during CUDA-graph warmup inside vLLM's custom
#   all-reduce kernel on some two-GPU hosts.  Qwen2.5-7B easily fits on one
#   A100-80GB, so this runner avoids tensor parallelism entirely and instead
#   runs two independent TP=1 workers, one per physical GPU, with the question
#   set split into two deterministic shards.  This changes only the inference
#   deployment, not the models, prompts, candidate budget, seed, confidence
#   definition, or CPU aggregation protocol.
#
# Pipeline per model:
#   two-GPU question-sharded response generation
#      -> two-GPU shard-matched confidence calculation
#      -> CPU Table-2 paper-budget aggregation
#   SSC model is completed first, then RelSC model, then paired comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

SSC_MODEL="${SSC_MODEL:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_paired_ssc}"
RELSC_MODEL="${RELSC_MODEL:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_paired_relsc}"
MODEL_SPECS="${MODEL_SPECS:-qwen_paired_ssc=${SSC_MODEL} qwen_paired_relsc=${RELSC_MODEL}}"

DATASETS="${DATASETS:-object_counting math_qa arc_challenge}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-8}"
CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SEED="${SEED:-42}"
BUDGETS="${BUDGETS:-1,2,4,8,16,32}"
CURVE_MAX_BUDGET="${CURVE_MAX_BUDGET:-32}"
BUDGET_TARGETS="${BUDGET_TARGETS:-16}"
DYNAMIC_MIN_VALID="${DYNAMIC_MIN_VALID:-2}"
RASC_BUFFER_SIZE="${RASC_BUFFER_SIZE:-5}"
ESC_WINDOW_SIZES="${ESC_WINDOW_SIZES:-}"
CISC_TEMPERATURE="${CISC_TEMPERATURE:-1.0}"
CISC_NORMALIZATION="${CISC_NORMALIZATION:-softmax}"
EXPECTED_QUESTIONS="${EXPECTED_QUESTIONS:-}"
CACHE_ROOT="${CACHE_ROOT:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"

EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/relacats_v2/outputs/paired_qwen_eval_sharded}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/paired_qwen_eval_sharded}"
COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/relacats_v2/outputs/paired_qwen_comparison_sharded}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ ]] || fail "GPU_FIRST must be an integer"
[[ "${GPU_SECOND}" =~ ^[0-9]+$ ]] || fail "GPU_SECOND must be an integer"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${NUM_GENERATIONS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GENERATIONS must be positive"
(( NUM_GENERATIONS >= 16 )) || fail "NUM_GENERATIONS must be at least 16 for Table-2 evaluation"

mkdir -p "${EVAL_ROOT}" "${LOG_ROOT}" "${COMPARE_ROOT}" "${CACHE_ROOT}"
exec 9>"${EVAL_ROOT}/.resume_eval_sharded.lock"
flock -n 9 || fail "another resume evaluation is using ${EVAL_ROOT}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

check_gpu_idle() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    return 0
  fi
  local gpu="$1" pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" \
    || fail "unable to query GPU ${gpu}"
  [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
}

check_gpu_idle "${GPU_FIRST}"
check_gpu_idle "${GPU_SECOND}"

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a MODEL_SPEC_ARRAY <<< "${MODEL_SPECS}"

cat <<EOF
===== PAIRED QWEN EVALUATION RESUME: QUESTION-SHARDED =====
GPUs=${GPU_FIRST},${GPU_SECOND}
mode=question_sharded (TP=1 per GPU, two deterministic question shards)
datasets=${DATASETS}
num_generations=${NUM_GENERATIONS}
target_average_budget=${BUDGET_TARGETS}
eval_root=${EVAL_ROOT}
EOF

for spec in "${MODEL_SPEC_ARRAY[@]}"; do
  [[ "${spec}" == *=* ]] || fail "MODEL_SPECS entry must be tag=/path: ${spec}"
  tag="${spec%%=*}"
  model_path="${spec#*=}"
  [[ "${tag}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || fail "invalid model tag: ${tag}"
  [[ -f "${model_path}/config.json" ]] || fail "merged model missing for ${tag}: ${model_path}/config.json"

  model_root="${EVAL_ROOT}/${tag}"
  response_root="${model_root}/responses/test"
  confidence_root="${model_root}/confidence/test"
  result_root="${model_root}/results/test"
  threshold_root="${model_root}/thresholds"
  model_log_root="${LOG_ROOT}/${tag}"
  mkdir -p "${model_log_root}"

  echo
  echo "===== START ${tag}: RESPONSE GENERATION ====="
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL="${model_path}" \
    OUTPUT_ROOT="${response_root}" \
    LOG_ROOT="${model_log_root}/generation" \
    CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" \
    GPU_SECOND="${GPU_SECOND}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE="question_sharded" \
    TENSOR_PARALLEL_SIZE="1" \
    NUM_SHARDS="2" \
    DATASETS="${DATASETS}" \
    SPLIT="test" \
    NUM_GENERATIONS="${NUM_GENERATIONS}" \
    TEMPERATURE="${TEMPERATURE}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/06_generate_eval.sh"

  echo "===== ${tag}: CONFIDENCE ====="
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL="${model_path}" \
    MODEL_FAMILY="qwen" \
    RESPONSES_ROOT="${response_root}" \
    OUTPUT_ROOT="${confidence_root}" \
    LOG_ROOT="${model_log_root}/confidence" \
    CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" \
    GPU_SECOND="${GPU_SECOND}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE="question_sharded" \
    TENSOR_PARALLEL_SIZE="1" \
    NUM_SHARDS="2" \
    DATASETS="${DATASETS}" \
    CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/07_calculate_confidence.sh"

  echo "===== ${tag}: PAPER-BUDGET AGGREGATION ====="
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    INPUT_ROOT="${confidence_root}" \
    OUTPUT_ROOT="${result_root}" \
    PHASE="paper" \
    MODEL_ID="${tag}" \
    THRESHOLD_ROOT="${threshold_root}" \
    DATASETS="${DATASETS}" \
    NUM_SHARDS="2" \
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
echo "===== FINAL PAIRED COMPARISON ====="
"${PYTHON_BIN}" -m relacats_v2.evaluation.compare_paired_qwen \
  --eval-root "${EVAL_ROOT}" \
  --ssc-tag qwen_paired_ssc \
  --relsc-tag qwen_paired_relsc \
  --datasets "${DATASET_ARRAY[@]}" \
  --budget "${BUDGET_TARGETS%%,*}" \
  --output-dir "${COMPARE_ROOT}"

echo
echo "ALL PAIRED QWEN EVALUATION STAGES COMPLETE"
echo "Evaluation root: ${EVAL_ROOT}"
echo "Comparison: ${COMPARE_ROOT}/paired_comparison.md"
