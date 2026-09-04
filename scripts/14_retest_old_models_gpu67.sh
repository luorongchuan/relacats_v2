#!/usr/bin/env bash
set -Eeuo pipefail

# Re-evaluate the released Self-Calibration checkpoints and the already
# merged RelaCaTS-v1 checkpoints.  This entry point intentionally never calls
# training or merge code: each model is loaded, scored, and fully released
# before the next model starts.

V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${V2_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-8}"
CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
DATASET_LIST="${DATASETS:-object_counting math_qa arc_challenge}"
LEGACY_EXCEL="${LEGACY_EXCEL:-}"
CACHE_ROOT="${CACHE_ROOT:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval_v2_retest}"
RUN_ROOT="${RUN_ROOT:-${V2_ROOT}/outputs/eval_outputs_v2_retest_old_models}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${RUN_ROOT}/artifacts}"
RESUME="${RESUME:-0}"

# A stale MAX_QUESTIONS from an earlier smoke/repair run would silently
# truncate the fresh test pool.  This retest is intentionally full-dataset;
# use a separate RUN_ROOT and edit DATASETS if a deliberately reduced smoke
# run is ever needed.
unset MAX_QUESTIONS 2>/dev/null || true

# Six explicit entries: author CaTS/Self-Calibration and already trained
# RelaCaTS-v1 merged model for each family.
MODEL_TAGS=(
  qwen2_5_7b_instruct_cats
  qwen2_5_7b_instruct_relacats_v1
  llama3_1_8b_instruct_cats
  llama3_1_8b_instruct_relacats_v1
  deepseek_r1_distill_qwen_1_5b_cats
  deepseek_r1_distill_qwen_1_5b_relacats_v1
)
MODEL_PATHS=(
  /home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct-Self-Calibration
  "${V2_ROOT}/outputs/merged_model/qwen2_5_7b_instruct_relacats_v1"
  /home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct-Self-Calibration
  "${V2_ROOT}/outputs/merged_model/llama3_1_8b_instruct_relacats_v1"
  /home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration
  "${V2_ROOT}/outputs/merged_model/deepseek_r1_distill_qwen_1_5b_relacats_v1"
)
MODEL_FAMILIES=(qwen qwen llama llama deepseek deepseek)
TOKENIZER_SOURCES=(
  /home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct-Self-Calibration
  /home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct-Self-Calibration
  /home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct-Self-Calibration
  /home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct-Self-Calibration
  /home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration
  /home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration
)

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
[[ -d "${V2_ROOT}" ]] || fail "v2 root not found: ${V2_ROOT}"
if [[ -n "${LEGACY_EXCEL}" && ! -f "${LEGACY_EXCEL}" ]]; then
  fail "LEGACY_EXCEL was supplied but does not exist: ${LEGACY_EXCEL}"
fi
[[ "$(readlink -f "${V2_ROOT}")" != "$(readlink -f "${PROJECT_ROOT}/relacats_v1")" ]] || \
  fail "v2 and v1 resolve to the same directory"
V2_GIT_ROOT="$(git -C "${V2_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ "${V2_GIT_ROOT}" == "${V2_ROOT}" ]] || fail "unexpected v2 Git root: ${V2_GIT_ROOT}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ && "${GPU_SECOND}" =~ ^[0-9]+$ ]] || \
  fail "GPU ids must be integers"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${NUM_GENERATIONS}" =~ ^[1-9][0-9]*$ ]] || fail "NUM_GENERATIONS must be positive"
(( NUM_GENERATIONS >= 16 )) || fail "NUM_GENERATIONS must be at least 16"

if [[ -e "${RUN_ROOT}" && "${RESUME}" != "1" ]]; then
  if [[ -f "${RUN_ROOT}/manifest.json" ]]; then
    fail "completed retest already exists: ${RUN_ROOT}"
  fi
  fail "retest output directory already exists; use a new RUN_ROOT or RESUME=1: ${RUN_ROOT}"
fi

for index in "${!MODEL_PATHS[@]}"; do
  model="${MODEL_PATHS[$index]}"
  tokenizer_source="${TOKENIZER_SOURCES[$index]}"
  [[ -f "${model}/config.json" ]] || fail "model config missing: ${model}"
  [[ -f "${model}/tokenizer_config.json" || -f "${model}/tokenizer.json" ]] || \
    fail "model tokenizer missing: ${model}"
  weight_file="$(find "${model}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print -quit)"
  [[ -n "${weight_file}" ]] || fail "model weights missing: ${model}"
  [[ -f "${tokenizer_source}/tokenizer_config.json" || -f "${tokenizer_source}/tokenizer.json" ]] || \
    fail "tokenizer source missing: ${tokenizer_source}"
done

read -r -a DATASET_ARRAY <<< "${DATASET_LIST}"
(( ${#DATASET_ARRAY[@]} > 0 )) || fail "DATASETS is empty"

mkdir -p "${RUN_ROOT}" "${ARTIFACT_ROOT}" "${RUN_ROOT}/logs" "${CACHE_ROOT}"
exec 9>"${RUN_ROOT}/.retest.lock"
flock -n 9 || fail "another v2 retest is using ${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/logs/launcher.log") 2>&1

gpu_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | \
    sed '/^[[:space:]]*$/d'
}
check_gpu_idle() {
  [[ "${ALLOW_BUSY_GPUS}" == "1" ]] && return 0
  local gpu pids
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(gpu_pids "${gpu}")" || fail "cannot query GPU ${gpu}"
    [[ -z "${pids}" ]] || fail "GPU ${gpu} is busy (PID(s): ${pids})"
  done
}
wait_gpu_idle() {
  [[ "${ALLOW_BUSY_GPUS}" == "1" ]] && return 0
  local attempt gpu pids busy
  for attempt in $(seq 1 60); do
    busy=0
    for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
      pids="$(gpu_pids "${gpu}")" || fail "cannot query GPU ${gpu}"
      [[ -z "${pids}" ]] || busy=1
    done
    (( busy == 0 )) && return 0
    sleep 1
  done
  fail "worker did not release GPUs ${GPU_FIRST},${GPU_SECOND}"
}

check_gpu_idle
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export HF_HOME="${CACHE_ROOT}"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "RelaCaTS-v2 old-model retest starting"
echo "v2_root=${V2_ROOT}"
echo "git_root=${V2_GIT_ROOT}"
echo "physical_gpus=${GPU_FIRST},${GPU_SECOND}; tensor_parallel_size=2"
echo "candidate_pool=${NUM_GENERATIONS}; target_budget=16; gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "max_questions=all (inherited MAX_QUESTIONS cleared)"
echo "models=${#MODEL_TAGS[@]}; datasets=${DATASET_LIST}"
echo "No training, merge, or checkpoint writes are performed by this script."

for index in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$index]}"
  model="${MODEL_PATHS[$index]}"
  family="${MODEL_FAMILIES[$index]}"
  tokenizer_source="${TOKENIZER_SOURCES[$index]}"
  case "${family}" in
    deepseek) max_tokens=2048 ;;
    *) max_tokens=1024 ;;
  esac
  response_root="${ARTIFACT_ROOT}/${tag}/responses"
  confidence_root="${ARTIFACT_ROOT}/${tag}/confidence"
  log_root="${RUN_ROOT}/logs/${tag}"
  mkdir -p "${log_root}"

  echo
  echo "===== START ${tag} ====="
  echo "model=${model}"
  echo "max_new_tokens=${max_tokens}; max_model_len=${MAX_MODEL_LEN}"

  # One TP worker owns both physical GPUs.  The next model is not started
  # until every dataset in both GPU stages has completed and released them.
  env PYTHON_BIN="${PYTHON_BIN}" MODEL="${model}" OUTPUT_ROOT="${response_root}" \
    TOKENIZER_SOURCE="${tokenizer_source}" \
    LOG_ROOT="${log_root}/generation" CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" GPU_SECOND="${GPU_SECOND}" \
    CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE=tensor_parallel TENSOR_PARALLEL_SIZE=2 NUM_SHARDS=1 \
    DATASETS="${DATASET_LIST}" SPLIT=test NUM_GENERATIONS="${NUM_GENERATIONS}" \
    TEMPERATURE="${TEMPERATURE}" MAX_TOKENS="${max_tokens}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" bash "${V2_ROOT}/scripts/06_generate_eval.sh"
  wait_gpu_idle

  env PYTHON_BIN="${PYTHON_BIN}" MODEL="${model}" RESPONSES_ROOT="${response_root}" \
    OUTPUT_ROOT="${confidence_root}" LOG_ROOT="${log_root}/confidence" CACHE_ROOT="${CACHE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" GPU_SECOND="${GPU_SECOND}" \
    CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    EVAL_GPU_MODE=tensor_parallel TENSOR_PARALLEL_SIZE=2 NUM_SHARDS=1 MODEL_FAMILY="${family}" \
    DATASETS="${DATASET_LIST}" CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    SEED="${SEED}" bash "${V2_ROOT}/scripts/07_calculate_confidence.sh"
  wait_gpu_idle
  echo "===== GPU stages complete: ${tag} ====="
done

# CPU-only stage: validates artifacts, chooses thresholds on a deterministic
# question-disjoint holdout, evaluates the held-out test, and writes all ten
# methods (including CISC/Self-Certainty/ESC/RASC and RelaCaTS-* labels).
retest_args=(
  --artifact-root "${ARTIFACT_ROOT}"
  --output-root "${RUN_ROOT}"
  --models "${MODEL_TAGS[@]}"
  --datasets "${DATASET_ARRAY[@]}"
  --validation-fraction "${VALIDATION_FRACTION:-0.2}"
  --seed "${SEED}"
  --target-budget "${TARGET_BUDGET:-16}"
)
if [[ -n "${LEGACY_EXCEL}" ]]; then
  retest_args+=(--legacy-excel "${LEGACY_EXCEL}")
fi
if [[ "${RESUME}" == "1" ]]; then
  retest_args+=(--resume)
fi
"${PYTHON_BIN}" -m relacats_v2.evaluation.retest_old_models "${retest_args[@]}"

echo
echo "ALL RELACATS-V2 OLD-MODEL RETESTS COMPLETE"
echo "Summary: ${RUN_ROOT}/retest_summary.csv"
echo "Report:  ${V2_ROOT}/docs/relacats_v2_retest_old_models_report.md"
