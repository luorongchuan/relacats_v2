#!/usr/bin/env bash
set -Eeuo pipefail

# Generate RelaCaTS-v2 teacher data for the three released base models.
#
# This wrapper deliberately runs models one after another.  Each invocation of
# 01_generate_relational_data.sh still starts two question-sharded workers,
# one on each physical GPU, but the next model is not started until both
# workers from the previous model have exited.  The generator's per-question
# atomic checkpoints make a rerun a safe resume operation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/luorongchuan/workspace_135/models}"
QWEN_MODEL="${QWEN_MODEL:-${MODEL_ROOT}/Qwen2.5-7B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-${MODEL_ROOT}/Llama-3.1-8B-Instruct}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"

GPU_FIRST="${GPU_FIRST:-2}"
GPU_SECOND="${GPU_SECOND:-5}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"

# The nine datasets used by the released CaTS data bundles.  Keep this order
# aligned with the user's requested run; the generator validates the list and
# applies its relation/identity budget policy per dataset.
DATASETS="${DATASETS:-arc_easy commonsense_qa gsm8k logiqa openbookqa reclor sciq svamp winogrande}"

# Keep each model directly below generated_data, as opposed to sharing a
# single raw file tree.  This also makes the paths easy to pass to the later
# RelSSC builder via INPUT_ROOT.
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/relacats_v2/outputs/generated_data}"
# Include the physical GPU IDs in the default log directory so an override to
# another pair cannot be mistaken for the historical GPU2/5 run.
LOG_BASE="${LOG_BASE:-${ROOT_DIR}/relacats_v2/outputs/logs/data_generation/all_models_gpu${GPU_FIRST}${GPU_SECOND}}"
HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_relacats_v2}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

GENERATOR="${GENERATOR:-${ROOT_DIR}/relacats_v2/scripts/01_generate_relational_data.sh}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${LOG_BASE}/launcher.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python not found or not executable: ${PYTHON_BIN}"
[[ -x "${GENERATOR}" ]] || fail "Generator not found or not executable: ${GENERATOR}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ && "${GPU_SECOND}" =~ ^[0-9]+$ ]] || \
  fail "GPU_FIRST/GPU_SECOND must be non-negative integers"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${MAX_QUESTIONS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_QUESTIONS must be positive"

MODEL_TAGS=(
  qwen2_5_7b_instruct
  llama3_1_8b_instruct
  deepseek_r1_distill_qwen_1_5b
)
MODEL_PATHS=("${QWEN_MODEL}" "${LLAMA_MODEL}" "${DEEPSEEK_MODEL}")

# Validate every model before either GPU is claimed.  A typo in a later model
# path should fail before producing a partially populated experiment.
for model_path in "${MODEL_PATHS[@]}"; do
  [[ -f "${model_path}/config.json" ]] || fail "Local model not found: ${model_path}"
done

mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}"
exec 8>"${LOG_BASE}/.all_models_generation.lock"
flock -n 8 || fail "another all-model generation wrapper is using ${LOG_BASE}"

# An absolute Python path does not put environment helper binaries (for
# example ninja used by vLLM/FlashInfer) on PATH.
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"

export PYTHON_BIN MODEL_ROOT QWEN_MODEL LLAMA_MODEL DEEPSEEK_MODEL
export GPU_FIRST GPU_SECOND MAX_QUESTIONS DATASETS
export HF_HOME HF_DATASETS_CACHE HF_ENDPOINT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export TEMPERATURE="${TEMPERATURE:-0.8}"
export CONFIDENCE_TEMPERATURE="${CONFIDENCE_TEMPERATURE:-0.0}"
export SEED="${SEED:-42}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "${LAUNCHER_LOG}")"

log() {
  # `tee` keeps the launcher log useful when the wrapper is detached while
  # preserving the same message on the caller's terminal.
  echo "[$(date '+%F %T %z')] $*" | tee -a "${LAUNCHER_LOG}"
}

log "RelaCaTS-v2 all-model generation starting"
log "physical_gpus=${GPU_FIRST},${GPU_SECOND}"
log "datasets=${DATASETS}"
log "max_questions_per_dataset=${MAX_QUESTIONS}"
log "model_order=${MODEL_TAGS[*]}"
log "output_base=${OUTPUT_BASE}"

ACTIVE_PID=""
TEE_PID=""
RUN_TMP_DIR=""
STREAM_PATH=""

cleanup_active() {
  # The generator is launched in its own process group.  This is a fallback
  # for an interrupt while the wrapper is waiting; the generator's own trap
  # performs the normal worker cleanup.
  if [[ -n "${ACTIVE_PID}" ]]; then
    kill -TERM -- "-${ACTIVE_PID}" 2>/dev/null || kill -TERM "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TEE_PID}" ]]; then
    kill -TERM "${TEE_PID}" 2>/dev/null || true
    wait "${TEE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${STREAM_PATH}" ]]; then
    rm -f -- "${STREAM_PATH}"
  fi
  if [[ -n "${RUN_TMP_DIR}" ]]; then
    rmdir -- "${RUN_TMP_DIR}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
  TEE_PID=""
  RUN_TMP_DIR=""
  STREAM_PATH=""
}

on_interrupt() {
  cleanup_active
  exit 143
}
trap on_interrupt INT TERM
trap cleanup_active EXIT

for index in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$index]}"
  model_path="${MODEL_PATHS[$index]}"
  output_root="${OUTPUT_BASE}/${tag}"
  model_log_root="${LOG_BASE}/${tag}"
  model_log="${model_log_root}/generation.log"
  case "${model_path,,}" in
    *deepseek*) model_max_new_tokens="${MAX_NEW_TOKENS:-2048}" ;;
    *) model_max_new_tokens="${MAX_NEW_TOKENS:-1024}" ;;
  esac
  mkdir -p "${model_log_root}"

  log "===== START ${tag} ====="
  log "model=${model_path}"
  log "output=${output_root}"
  log "log=${model_log}"

  # The generator itself owns a lock and waits for both shards.  Keeping this
  # command serial is what guarantees that no two models share GPU 2/5.  A
  # FIFO lets us retain the generator PID (for reliable signal cleanup) while
  # still streaming the output to both the model log and launcher log.
  RUN_TMP_DIR="$(mktemp -d "${model_log_root}/.stream.XXXXXX")"
  STREAM_PATH="${RUN_TMP_DIR}/output.fifo"
  mkfifo "${STREAM_PATH}"
  tee -a "${model_log}" -a "${LAUNCHER_LOG}" <"${STREAM_PATH}" &
  TEE_PID=$!
  set +e
  setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
    MODEL_NAME="${model_path}" \
    MAX_NEW_TOKENS="${model_max_new_tokens}" \
    OUTPUT_ROOT="${output_root}" \
    LOG_ROOT="${model_log_root}" \
    bash "${GENERATOR}" >"${STREAM_PATH}" 2>&1 &
  ACTIVE_PID=$!
  wait "${ACTIVE_PID}"
  generator_status=$?
  set -e
  # The process has been reaped; do not let cleanup send a signal to a
  # potentially reused process-group id.
  ACTIVE_PID=""

  # A successful generator closes the FIFO and lets tee exit naturally.  On
  # failure, terminate tee explicitly so a surviving child cannot make the
  # wrapper wait forever; generator cleanup has already run via its trap.
  if (( generator_status != 0 )); then
    kill -TERM "${TEE_PID}" 2>/dev/null || true
  fi
  set +e
  wait "${TEE_PID}"
  tee_status=$?
  set -e
  # The tee process has been reaped; clear its PID before the generic cleanup
  # trap so a rapidly reused PID can never receive a stray signal.
  TEE_PID=""
  cleanup_active

  if (( generator_status != 0 )); then
    log "FAILED ${tag}: generator_exit=${generator_status}; later models were not started"
    exit "${generator_status}"
  fi
  if (( tee_status != 0 )); then
    log "FAILED ${tag}: log_writer_exit=${tee_status}; later models were not started"
    exit 1
  fi

  log "===== COMPLETE ${tag} ====="
done

log "All three model generations complete"
