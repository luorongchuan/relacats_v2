#!/usr/bin/env bash
set -Eeuo pipefail

# RelaCaTS v2 retraining launcher for larger per-rank micro-batches.
#
# This is intentionally a separate launcher from 10_final_audit_and_train.sh:
# the latter may still be running in an older terminal, and editing a shell
# script while bash is reading it can corrupt the heredoc parser.  Invoke this
# file only after the old training process has been stopped.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
WORLD_SIZE=2
EFFECTIVE_UPDATE_BATCH="${EFFECTIVE_UPDATE_BATCH:-128}"
MODE="${MODE:-train}"                 # train or smoke
SMOKE_MODEL="${SMOKE_MODEL:-qwen2_5_7b_instruct}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-4096}"
SMOKE_EVAL_SAMPLES="${SMOKE_EVAL_SAMPLES:-100}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-}"
RESUME="${RESUME:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_MODE="${TARGET_MODE:-residual}"
LAMBDA_REL="${LAMBDA_REL:-0.5}"

OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/relacats_v2/outputs}"
RELSSC_BASE="${RELSSC_BASE:-${OUTPUT_BASE}/relssc_dataset}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${OUTPUT_BASE}/checkpoints}"
LOG_BASE="${LOG_BASE:-${OUTPUT_BASE}/logs/fast_train_gpu${GPU_FIRST}${GPU_SECOND}}"
MODELS="${MODELS:-qwen2_5_7b_instruct llama3_1_8b_instruct deepseek_r1_distill_qwen_1_5b}"

# Safe defaults: preserve global effective batch 128 while spending more VRAM
# on activations.  Qwen is the largest current target, so it gets the first
# B=8 trial; Llama is kept conservative; the 1.5B model can use B=16.
GLOBAL_BATCH="${BATCH_SIZE:-}"
GLOBAL_ACCUM="${GRADIENT_ACCUMULATION_STEPS:-}"
BATCH_SIZE_QWEN="${BATCH_SIZE_QWEN:-${GLOBAL_BATCH:-8}}"
BATCH_SIZE_LLAMA="${BATCH_SIZE_LLAMA:-4}"
BATCH_SIZE_DEEPSEEK="${BATCH_SIZE_DEEPSEEK:-16}"
GA_QWEN="${GRADIENT_ACCUMULATION_STEPS_QWEN:-${GLOBAL_ACCUM:-8}}"
GA_LLAMA="${GRADIENT_ACCUMULATION_STEPS_LLAMA:-16}"
GA_DEEPSEEK="${GRADIENT_ACCUMULATION_STEPS_DEEPSEEK:-4}"
GC_QWEN="${GRADIENT_CHECKPOINTING_QWEN:-0}"
GC_LLAMA="${GRADIENT_CHECKPOINTING_LLAMA:-0}"
GC_DEEPSEEK="${GRADIENT_CHECKPOINTING_DEEPSEEK:-0}"

DATASETS="${DATASETS:-arc_easy commonsense_qa gsm8k logiqa openbookqa reclor sciq svamp winogrande}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ && "${GPU_SECOND}" =~ ^[0-9]+$ ]] || fail "GPU ids must be numeric"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${MODE}" == "train" || "${MODE}" == "smoke" ]] || fail "MODE must be train or smoke"
[[ "${TARGET_MODE}" =~ ^(ssc|relssc_replace|residual)$ ]] || fail "invalid TARGET_MODE=${TARGET_MODE}"
"${PYTHON_BIN}" - "${LAMBDA_REL}" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value < 0:
    raise SystemExit("LAMBDA_REL must be finite and non-negative")
PY

mkdir -p "${LOG_BASE}"
exec 9>"${LOG_BASE}/.fast_train.lock"
flock -n 9 || fail "another fast-train launcher is already running: ${LOG_BASE}/.fast_train.lock"

check_gpu_free() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    echo "ALLOW_BUSY_GPUS=1: skipping GPU safety check"
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    local pids
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
}

model_info() {
  local tag="$1"
  case "${tag}" in
    qwen2_5_7b_instruct)
      MODEL_NAME="/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
      CONFIG_NAME="qwen2_5_7b_2xa100.json"
      BATCH="${BATCH_SIZE_QWEN}"; ACCUM="${GA_QWEN}"; GC="${GC_QWEN}"
      ;;
    llama3_1_8b_instruct)
      MODEL_NAME="/home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct"
      CONFIG_NAME="llama3_1_8b_2xa100.json"
      BATCH="${BATCH_SIZE_LLAMA}"; ACCUM="${GA_LLAMA}"; GC="${GC_LLAMA}"
      ;;
    deepseek_r1_distill_qwen_1_5b)
      MODEL_NAME="/home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B"
      CONFIG_NAME="deepseek_1_5b_2xa100.json"
      BATCH="${BATCH_SIZE_DEEPSEEK}"; ACCUM="${GA_DEEPSEEK}"; GC="${GC_DEEPSEEK}"
      ;;
    *) fail "unknown model tag: ${tag}" ;;
  esac
  [[ "${BATCH}" =~ ^[1-9][0-9]*$ ]] || fail "${tag}: invalid batch size ${BATCH}"
  [[ "${ACCUM}" =~ ^[1-9][0-9]*$ ]] || fail "${tag}: invalid accumulation ${ACCUM}"
  [[ "${GC}" =~ ^(0|1|true|false|yes|no)$ ]] || fail "${tag}: GC must be 0/1/true/false"
  local actual=$((BATCH * WORLD_SIZE * ACCUM))
  if [[ "${actual}" != "${EFFECTIVE_UPDATE_BATCH}" && "${ALLOW_EFFECTIVE_BATCH_CHANGE:-0}" != "1" ]]; then
    fail "${tag}: batch*world*accum=${actual}, expected ${EFFECTIVE_UPDATE_BATCH}; set matching GA or ALLOW_EFFECTIVE_BATCH_CHANGE=1"
  fi
}

make_runtime_config() {
  local source="$1" target="$2" model="$3" dataset_root="$4" output="$5" batch="$6" accum="$7" gc="$8" target_mode="$9" lambda_rel="${10}"
  mkdir -p "$(dirname "${target}")"
  "${PYTHON_BIN}" - "${source}" "${target}" "${model}" "${dataset_root}" "${output}" "${batch}" "${accum}" "${gc}" "${target_mode}" "${lambda_rel}" <<'PY'
import json
import os
import sys

source, target, model, dataset_root, output, batch, accum, gc, target_mode, lambda_rel = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config["model_name"] = model
config["dataset_root"] = dataset_root
config["output_dir"] = output
config["batch_size"] = int(batch)
config["gradient_accumulation_steps"] = int(accum)
token = str(gc).strip().lower()
config["gradient_checkpointing"] = token in {"1", "true", "yes"}
config["normalize_task_loss_by_records"] = True
config["target_mode"] = target_mode
config["lambda_rel"] = float(lambda_rel)
config["generation_selection_target"] = "ssc"
tmp = target + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(tmp, target)
PY
}

run_one() {
  local tag="$1" output_dir="$2" log_dir="$3" max_train="$4" max_eval="$5" max_steps="$6"
  model_info "${tag}"
  local dataset_root="${RELSSC_BASE}/${tag}"
  local source_config="${ROOT_DIR}/relacats_v2/configs/${CONFIG_NAME}"
  local runtime_config="${log_dir}/runtime_train_config.json"
  local train_log="${log_dir}/train.log"
  [[ -f "${MODEL_NAME}/config.json" ]] || fail "base model missing: ${MODEL_NAME}"
  [[ -f "${dataset_root}/manifest.json" ]] || fail "RelSSC manifest missing: ${dataset_root}/manifest.json"
  [[ -f "${source_config}" ]] || fail "config missing: ${source_config}"
  mkdir -p "${log_dir}"
  make_runtime_config "${source_config}" "${runtime_config}" "${MODEL_NAME}" "${dataset_root}" "${output_dir}" "${BATCH}" "${ACCUM}" "${GC}" "${TARGET_MODE}" "${LAMBDA_REL}"

  local args=(
    --config-file "${runtime_config}"
    --save-path "${output_dir}"
    --batch-size "${BATCH}"
    --gradient-accumulation-steps "${ACCUM}"
  )
  case "${GC}" in
    0|false|no) args+=(--no-gradient-checkpointing) ;;
    1|true|yes) args+=(--gradient-checkpointing) ;;
  esac
  [[ -n "${max_train}" ]] && args+=(--max-train-samples "${max_train}")
  [[ -n "${max_eval}" ]] && args+=(--max-eval-samples "${max_eval}")
  [[ -n "${max_steps}" ]] && args+=(--max-optimizer-steps "${max_steps}")

  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    if [[ "${RESUME}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
      args+=(--resume-from "${output_dir}")
      echo "RESUME ${tag}: LoRA weights only; optimizer/scheduler restart"
    else
      fail "refusing to overwrite non-empty output: ${output_dir}"
    fi
  fi

  echo "===== ${MODE^^} ${tag} ====="
  echo "model=${MODEL_NAME}"
  echo "dataset_root=${dataset_root}"
  echo "batch_size=${BATCH} accumulation=${ACCUM} effective_batch=$((BATCH * WORLD_SIZE * ACCUM)) gradient_checkpointing=${GC}"
  echo "target_mode=${TARGET_MODE} lambda_rel=${LAMBDA_REL} generation_selection_target=ssc"
  echo "output=${output_dir}"
  echo "log=${train_log}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 relacats_v2/model_training/train_relacats.py "${args[@]}"
    printf '\n'
    return 0
  fi
  check_gpu_free
  CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 \
      relacats_v2/model_training/train_relacats.py "${args[@]}" \
      2>&1 | tee "${train_log}"
  [[ -f "${output_dir}/adapter_config.json" ]] || fail "no adapter saved: ${output_dir}"
  echo "COMPLETE ${tag}: ${output_dir}"
}

if [[ "${MODE}" == "smoke" ]]; then
  model_info "${SMOKE_MODEL}"
  smoke_out="${SMOKE_OUTPUT_DIR}"
  if [[ -z "${smoke_out}" ]]; then
    smoke_out="${OUTPUT_BASE}/smoke/fast_${SMOKE_MODEL}_b${BATCH}_gc${GC}_$(date +%Y%m%d_%H%M%S)"
  fi
  run_one "${SMOKE_MODEL}" "${smoke_out}" "${LOG_BASE}/${SMOKE_MODEL}_smoke" "${SMOKE_TRAIN_SAMPLES}" "${SMOKE_EVAL_SAMPLES}" "${SMOKE_STEPS}"
else
  for tag in ${MODELS}; do
    run_one "${tag}" "${CHECKPOINT_BASE}/${tag}_relacats_v2" "${LOG_BASE}/${tag}" "" "" ""
  done
  echo "ALL FAST RELACATS v2 TRAINING RUNS COMPLETE"
fi
