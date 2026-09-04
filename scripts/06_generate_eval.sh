#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
# Keep the default aligned with the actual merged-checkpoint tag produced by
# 05_merge_model.sh.  Callers can still override MODEL for another base model.
MODEL="${MODEL:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_instruct_relacats_v2}"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/responses}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/eval_generation_v2}"
CACHE_ROOT="${CACHE_ROOT:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
DATASETS=( ${DATASETS:-object_counting math_qa arc_challenge} )
SPLIT="${SPLIT:-test}"
# ``question_sharded`` is the historical mode: one model replica per GPU,
# with the question set split between the two workers.  ``tensor_parallel``
# starts one vLLM process whose model is tensor-parallel over both GPUs.  The
# latter is used by the serial three-model runner so that only one checkpoint
# is resident at a time.
EVAL_GPU_MODE="${EVAL_GPU_MODE:-question_sharded}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-2}"

# Keep child vLLM processes in process groups so an interrupted/failed dataset
# cannot leave a model engine holding a GPU after this launcher exits.
ACTIVE_PIDS=()
terminate_active() {
  (( ${#ACTIVE_PIDS[@]} > 0 )) || return 0
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
}
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( status != 0 )); then terminate_active; fi
  for pid in "${ACTIVE_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
# vLLM can launch FlashInfer compilation helpers from the selected Python
# environment; make that environment's bin directory available.
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
[[ -f "${MODEL}/config.json" ]] || fail "Merged model not found: ${MODEL}"
if [[ -n "${TOKENIZER_SOURCE}" ]]; then
  [[ -f "${TOKENIZER_SOURCE}/tokenizer_config.json" || -f "${TOKENIZER_SOURCE}/tokenizer.json" ]] || \
    fail "Tokenizer source not found: ${TOKENIZER_SOURCE}"
fi
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
case "${EVAL_GPU_MODE}" in
  question_sharded)
    [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_SHARDS must be a positive integer"
    (( NUM_SHARDS == 2 )) || fail "question_sharded mode requires NUM_SHARDS=2"
    (( TENSOR_PARALLEL_SIZE == 1 )) || fail "question_sharded mode requires TENSOR_PARALLEL_SIZE=1"
    ;;
  tensor_parallel)
    [[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_SHARDS must be a positive integer"
    [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "TENSOR_PARALLEL_SIZE must be a positive integer"
    (( NUM_SHARDS == 1 )) || fail "tensor_parallel mode requires NUM_SHARDS=1"
    (( TENSOR_PARALLEL_SIZE == 2 )) || fail "tensor_parallel mode on GPU_FIRST/GPU_SECOND requires TENSOR_PARALLEL_SIZE=2"
    ;;
  *)
    fail "EVAL_GPU_MODE must be question_sharded or tensor_parallel"
    ;;
esac
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${CACHE_ROOT}"
exec 9>"${OUTPUT_ROOT}/.generation.lock"
flock -n 9 || fail "another evaluation generator is using ${OUTPUT_ROOT}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
VISIBLE_GPUS="${GPU_FIRST},${GPU_SECOND}"

for dataset in "${DATASETS[@]}"; do
  num_generations="${NUM_GENERATIONS:-32}"
  case "${MODEL,,}" in
    *deepseek*) resolved_max_tokens="${MAX_TOKENS:-2048}" ;;
    *) resolved_max_tokens="${MAX_TOKENS:-1024}" ;;
  esac
  resolved_max_model_len="${MAX_MODEL_LEN:-8192}"
  echo "Generating ${dataset}: ${num_generations} shared test-time responses/question (for baseline and RelaCaTS aggregators); max_new_tokens=${resolved_max_tokens}; max_model_len=${resolved_max_model_len}; mode=${EVAL_GPU_MODE}, GPUs=${VISIBLE_GPUS}"
  common=(
    --model "${MODEL}"
    --dataset "${dataset}"
    --output-dir "${OUTPUT_ROOT}/${dataset}"
    --split "${SPLIT}"
    --num-generations "${num_generations}"
    --temperature "${TEMPERATURE:-1.0}"
    --max-tokens "${resolved_max_tokens}"
    --max-model-len "${resolved_max_model_len}"
    --question-batch-size "${QUESTION_BATCH_SIZE:-8}"
    --num-shards "${NUM_SHARDS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --seed "${SEED:-42}"
  )
  if [[ -n "${MAX_QUESTIONS:-}" ]]; then common+=(--max-questions "${MAX_QUESTIONS}"); fi
  if [[ -n "${TOKENIZER_SOURCE}" ]]; then common+=(--tokenizer-source "${TOKENIZER_SOURCE}"); fi

  if [[ "${EVAL_GPU_MODE}" == "tensor_parallel" ]]; then
    # One process owns both visible GPUs.  Do not launch a second worker here:
    # doing so would load a duplicate model and corrupt the TP process group.
    setsid env CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.generate_responses "${common[@]}" --shard-index 0 \
      >"${LOG_ROOT}/${dataset}_tensor_parallel_gpu${GPU_FIRST}_${GPU_SECOND}.log" 2>&1 &
    tp_pid=$!
    ACTIVE_PIDS+=("${tp_pid}")
    set +e
    wait "${tp_pid}"; tp_status=$?
    set -e
    if (( tp_status != 0 )); then
      terminate_active
      fail "${dataset} tensor-parallel response generation failed (${tp_status}); see ${LOG_ROOT}"
    fi
    ACTIVE_PIDS=()
  else
    setsid env CUDA_VISIBLE_DEVICES="${GPU_FIRST}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.generate_responses "${common[@]}" --shard-index 0 \
      >"${LOG_ROOT}/${dataset}_shard0_gpu${GPU_FIRST}.log" 2>&1 &
    first_pid=$!
    ACTIVE_PIDS+=("${first_pid}")
    setsid env CUDA_VISIBLE_DEVICES="${GPU_SECOND}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.generate_responses "${common[@]}" --shard-index 1 \
      >"${LOG_ROOT}/${dataset}_shard1_gpu${GPU_SECOND}.log" 2>&1 &
    second_pid=$!
    ACTIVE_PIDS+=("${second_pid}")
    set +e
    wait "${first_pid}"; first_status=$?
    wait "${second_pid}"; second_status=$?
    set -e
    if (( first_status != 0 || second_status != 0 )); then
      terminate_active
      fail "${dataset} response generation failed (${first_status}/${second_status}); see ${LOG_ROOT}"
    fi
    ACTIVE_PIDS=()
  fi
done
echo "Evaluation responses complete: ${OUTPUT_ROOT}"
