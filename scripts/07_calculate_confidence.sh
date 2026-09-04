#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
# Keep the default aligned with the actual merged-checkpoint tag produced by
# 05_merge_model.sh.  Callers can still override MODEL for another base model.
MODEL="${MODEL:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_instruct_relacats_v2}"
RESPONSES_ROOT="${RESPONSES_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/responses}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v2/outputs/eval_outputs_v2/confidence}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/eval_confidence_v2}"
CACHE_ROOT="${CACHE_ROOT:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
DATASETS=( ${DATASETS:-object_counting math_qa arc_challenge} )
# Keep this in lock-step with 06_generate_eval.sh.  In tensor_parallel mode a
# single confidence worker owns both GPUs and consumes one producer shard.
EVAL_GPU_MODE="${EVAL_GPU_MODE:-question_sharded}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-2}"
# Leave this empty by default so the Python helper can infer the family from
# the actual model path.  Set MODEL_FAMILY explicitly only for a generic
# renamed directory (for example MODEL_FAMILY=llama).
MODEL_FAMILY="${MODEL_FAMILY:-}"

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
# vLLM/FlashInfer needs helper binaries such as ninja from this environment.
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
[[ -f "${MODEL}/config.json" ]] || fail "Merged model not found: ${MODEL}"
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
exec 9>"${OUTPUT_ROOT}/.confidence.lock"
flock -n 9 || fail "another confidence job is using ${OUTPUT_ROOT}"

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
  response_root="${RESPONSES_ROOT}/${dataset}"
  [[ -d "${response_root}" ]] || fail \
    "response artifact missing for ${dataset}; run 06_generate_eval.sh"
  # Each confidence worker must consume its corresponding producer shard.
  # Passing the parent directory here would make workers scan all shards and
  # then split the records a second time.  The shard count is dynamic so the
  # single-shard tensor-parallel mode is validated exactly like the historical
  # two-shard mode.
  for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
    shard_dir="${response_root}/shard-$(printf '%05d' "${shard_index}")-of-$(printf '%05d' "${NUM_SHARDS}")"
    [[ -d "${shard_dir}/chunks" ]] || fail \
      "response shard missing for ${dataset}: ${shard_dir}; run 06_generate_eval.sh"
    [[ -f "${shard_dir}/response_manifest.json" ]] || fail \
      "response manifest missing for ${dataset}: ${shard_dir}"
    "${PYTHON_BIN}" - "${shard_dir}/response_manifest.json" <<'PY' || fail "Incomplete response manifest: ${shard_dir}"
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    manifest = json.load(handle)
if manifest.get("complete") is not True:
    raise SystemExit(1)
if manifest.get("expected_samples") != manifest.get("samples"):
    raise SystemExit(1)
if manifest.get("expected_questions") != manifest.get("questions"):
    raise SystemExit(1)
PY
  done
  echo "Calculating ${dataset} shared evaluation confidence (SC/CISC/Self-Certainty/Best-of-N/ASC/ESC/RASC + RelaCaTS-SC/RelaCaTS-ES/RelaCaTS-ASC); mode=${EVAL_GPU_MODE}, GPUs=${VISIBLE_GPUS}"
  common=(
    --model "${MODEL}"
    --output-dir "${OUTPUT_ROOT}/${dataset}"
    --batch-size "${CONFIDENCE_BATCH_SIZE:-128}"
    --num-shards "${NUM_SHARDS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "${MAX_MODEL_LEN:-8192}"
    --seed "${SEED:-42}"
    --responses-already-sharded
  )
  if [[ -n "${MODEL_FAMILY}" ]]; then
    common+=(--model-family "${MODEL_FAMILY}")
  fi
  if [[ "${EVAL_GPU_MODE}" == "tensor_parallel" ]]; then
    # One confidence engine spans both GPUs.  It must not be paired with a
    # second worker, otherwise two TP process groups would claim the same
    # devices and the result is either an OOM or an NCCL hang.
    response_shard="${response_root}/shard-00000-of-00001"
    setsid env CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.calculate_confidence "${common[@]}" \
      --responses "${response_shard}" --shard-index 0 \
      >"${LOG_ROOT}/${dataset}_tensor_parallel_gpu${GPU_FIRST}_${GPU_SECOND}.log" 2>&1 &
    tp_pid=$!
    ACTIVE_PIDS+=("${tp_pid}")
    set +e
    wait "${tp_pid}"; tp_status=$?
    set -e
    if (( tp_status != 0 )); then
      terminate_active
      fail "${dataset} tensor-parallel confidence failed (${tp_status}); see ${LOG_ROOT}"
    fi
    ACTIVE_PIDS=()
  else
    first_response_shard="${response_root}/shard-00000-of-00002"
    second_response_shard="${response_root}/shard-00001-of-00002"
    setsid env CUDA_VISIBLE_DEVICES="${GPU_FIRST}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.calculate_confidence "${common[@]}" \
      --responses "${first_response_shard}" --shard-index 0 \
      >"${LOG_ROOT}/${dataset}_shard0_gpu${GPU_FIRST}.log" 2>&1 &
    first_pid=$!
    ACTIVE_PIDS+=("${first_pid}")
    setsid env CUDA_VISIBLE_DEVICES="${GPU_SECOND}" "${PYTHON_BIN}" \
      -m relacats_v2.evaluation.calculate_confidence "${common[@]}" \
      --responses "${second_response_shard}" --shard-index 1 \
      >"${LOG_ROOT}/${dataset}_shard1_gpu${GPU_SECOND}.log" 2>&1 &
    second_pid=$!
    ACTIVE_PIDS+=("${second_pid}")
    set +e
    wait "${first_pid}"; first_status=$?
    wait "${second_pid}"; second_status=$?
    set -e
    if (( first_status != 0 || second_status != 0 )); then
      terminate_active
      fail "${dataset} confidence failed (${first_status}/${second_status}); see ${LOG_ROOT}"
    fi
    ACTIVE_PIDS=()
  fi
done
echo "Confidence artifacts complete: ${OUTPUT_ROOT}"
