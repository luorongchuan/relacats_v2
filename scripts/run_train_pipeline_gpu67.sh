#!/usr/bin/env bash
# RelaCaTS-v1 training pipeline (physical GPUs 6 and 7).
#
# This is an orchestration wrapper only.  It deliberately keeps the three
# expensive/irreversible stages separate:
#   generate: 4 option-permutation views x 8 responses = 32 teacher samples
#   build:    compute RelSSC targets and make question-disjoint train/test JSONL
#   train:    train the LoRA adapter with the RelSSC targets
#
# The wrapper never reads the original HINT-lab parquet files.  Those files
# contain ordinary CaTS responses/SSC and do not contain the relational views
# required by RelaCaTS-v1.  A run may be resumed by invoking the desired stage
# again; the underlying scripts validate metadata and atomically checkpoint
# individual questions.
#
# Usage (from any directory):
#   STAGE=preflight bash /abs/path/run_train_pipeline_gpu67.sh
#   STAGE=smoke     bash /abs/path/run_train_pipeline_gpu67.sh
#   STAGE=generate   bash /abs/path/run_train_pipeline_gpu67.sh
#   STAGE=build      bash /abs/path/run_train_pipeline_gpu67.sh
#   STAGE=train      bash /abs/path/run_train_pipeline_gpu67.sh
#   STAGE=all        bash /abs/path/run_train_pipeline_gpu67.sh
#
# Environment overrides are intentional.  The defaults are a Qwen2.5-7B
# single-model run on physical GPUs 6 and 7.  Set MODEL_NAME and CONFIG_FILE
# together when training another supported base model.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

STAGE="${STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/Self-Calibration/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
MODEL_NAME="${MODEL_NAME:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/relacats_v2/configs/qwen2_5_7b_2xa100.json}"

# Keep the default tree identical to the individual stage scripts.  A custom
# RUN_ROOT is supported and causes a runtime training config to be generated,
# so a custom data directory cannot accidentally be ignored by the trainer.
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/relacats_v2/outputs}"
RAW_ROOT="${RAW_ROOT:-${RUN_ROOT}/generated_data}"
DATASET_ROOT="${DATASET_ROOT:-${RUN_ROOT}/relssc_dataset}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/qwen2_5_7b_instruct_relacats_v2}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_ROOT}/train_pipeline_gpu${GPU_FIRST}${GPU_SECOND}.log}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${LOG_ROOT}/runtime_train_config.json}"
# Smoke is intentionally isolated from the formal output tree.  It uses the
# existing 03_smoke_train.sh implementation and is optional; STAGE=all does
# not invoke it so an accidental smoke run cannot consume the formal GPUs.
SMOKE_ROOT="${SMOKE_ROOT:-${RUN_ROOT}/smoke_gpu${GPU_FIRST}${GPU_SECOND}}"

HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# Match the original CaTS training mixture.  Ordinary MCQ datasets use 4x8
# relational views; WinoGrande uses two unique views at 16 responses each;
# GSM8K/SVAMP are handled by the generator as numeric 1x32 identity
# profiles.  ``arc_challenge``/``math_qa`` remain available via DATASETS=...
# for legacy v1 experiments but are not in the published nine-task mixture.
DATASETS="${DATASETS:-arc_easy commonsense_qa gsm8k logiqa openbookqa reclor sciq svamp winogrande}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
# Generation controls are kept explicit here instead of silently inheriting
# the defaults in 01_generate_relational_data.sh.  They are part of the run
# metadata and can be overridden for a pilot without editing child scripts.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.8}"
CONFIDENCE_TEMPERATURE="${CONFIDENCE_TEMPERATURE:-0.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
RESUME_FROM="${RESUME_FROM:-}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
ALLOW_MODEL_CONFIG_MISMATCH="${ALLOW_MODEL_CONFIG_MISMATCH:-0}"
export ALLOW_MODEL_CONFIG_MISMATCH
export GRADIENT_ACCUMULATION_STEPS
export DATASETS

usage() {
  cat <<'EOF'
RelaCaTS-v1 training pipeline

Stages (set STAGE=...):
  preflight  Validate Python, packages, local model/config, paths and GPU IDs
  smoke      Run the optional two-question GPU smoke (isolated output tree)
  generate   Generate relational teacher data (ordinary MCQ 4x8; WinoGrande 2x16; numeric 1x32 identity)
  build      Build the RelSSC train/test dataset from generated JSON files
  train      Train the LoRA adapter from the RelSSC dataset
  all        Run generate, build, then train in strict order (default)

Useful overrides:
  PYTHON_BIN=/path/to/python MODEL_NAME=/path/to/base-model
  CONFIG_FILE=/path/to/config.json RUN_ROOT=/path/to/run-root
  GPU_FIRST=6 GPU_SECOND=7 MAX_QUESTIONS=1000
  MAX_NEW_TOKENS=1024 MAX_MODEL_LEN=8192 QUESTION_BATCH_SIZE=4
  GPU_MEMORY_UTILIZATION=0.90 TEMPERATURE=0.8 CONFIDENCE_TEMPERATURE=0.0
  GRADIENT_ACCUMULATION_STEPS=64
  SMOKE_ROOT=/path/to/smoke-root FORCE_SMOKE_OVERWRITE=1

The HINT-lab ordinary CaTS parquet is not accepted as relational teacher data;
use generate (or point RAW_ROOT at a completed RelaCaTS generated_data tree).
EOF
}

log() {
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  printf '[%s] %s\n' "$(date '+%F %T%z')" "$*" | tee -a "${PIPELINE_LOG}"
}

fail() {
  log "ERROR: $*"
  exit 1
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

check_python_and_packages() {
  [[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found or not executable: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY' || fail "Python dependency check failed"
import importlib.util
import sys
required = ("torch", "transformers", "peft", "datasets", "vllm")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python packages: {', '.join(missing)}")
import torch
print(f"python={sys.executable}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("packages=ok")
PY
}

check_local_model() {
  [[ -f "${MODEL_NAME}/config.json" ]] || fail "Local base model config missing: ${MODEL_NAME}/config.json"
  [[ -f "${MODEL_NAME}/tokenizer_config.json" || -f "${MODEL_NAME}/tokenizer.json" ]] || \
    fail "Tokenizer files missing under local model: ${MODEL_NAME}"
}

check_config() {
  [[ -f "${CONFIG_FILE}" ]] || fail "Training config missing: ${CONFIG_FILE}"
"${PYTHON_BIN}" - "${CONFIG_FILE}" "${MODEL_NAME}" <<'PY' || fail "Invalid config or model mismatch"
import json
import os
import sys
from pathlib import Path
config_path, requested_model = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    cfg = json.load(handle)
if not isinstance(cfg, dict):
    raise SystemExit("config must be a JSON object")
for key in ("model_name", "dataset_root", "datasets", "batch_size", "gradient_accumulation_steps"):
    if key not in cfg:
        raise SystemExit(f"config missing required key: {key}")
configured = str(cfg["model_name"])
if (Path(configured).expanduser().resolve() != Path(requested_model).expanduser().resolve()
        and os.environ.get("ALLOW_MODEL_CONFIG_MISMATCH") != "1"):
    raise SystemExit(
        f"config model_name={configured!r} differs from MODEL_NAME={requested_model!r}; "
        "set both explicitly or use ALLOW_MODEL_CONFIG_MISMATCH=1"
    )
print(f"config_model={configured}")
print(f"config_datasets={len(cfg['datasets'])}")
PY
}

check_gpu_ids() {
  [[ "${GPU_FIRST}" =~ ^[0-9]+$ && "${GPU_SECOND}" =~ ^[0-9]+$ ]] || \
    fail "GPU_FIRST/GPU_SECOND must be numeric physical IDs"
  [[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required for GPU preflight"
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    nvidia-smi -i "${gpu}" --query-gpu=index,name --format=csv,noheader 2>/dev/null | \
      grep -q . || fail "GPU ${gpu} is not visible to nvidia-smi"
    if [[ "${ALLOW_BUSY_GPUS}" != "1" ]]; then
      local pids
      pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
      [[ -z "${pids}" ]] || fail "GPU ${gpu} is busy (PID(s): ${pids}); wait or set ALLOW_BUSY_GPUS=1 intentionally"
    fi
  done
}

check_common_inputs() {
  check_python_and_packages
  check_local_model
  check_config
  # Do not create or inspect HINT parquet as a substitute for relational data.
  if [[ "${RAW_ROOT}" == *HINT-lab* ]]; then
    fail "RAW_ROOT points at HINT-lab; use a RelaCaTS generated_data directory instead"
  fi
  mkdir -p "${LOG_ROOT}"
  is_positive_integer "${MAX_QUESTIONS}" || fail "MAX_QUESTIONS must be positive"
  is_positive_integer "${SEED}" || fail "SEED must be a positive integer"
  if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    is_positive_integer "${GRADIENT_ACCUMULATION_STEPS}" || \
      fail "GRADIENT_ACCUMULATION_STEPS must be a positive integer"
  fi
}

write_runtime_config() {
  mkdir -p "$(dirname "${RUNTIME_CONFIG}")"
  "${PYTHON_BIN}" - "${CONFIG_FILE}" "${RUNTIME_CONFIG}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" "${MODEL_NAME}" <<'PY'
import json
import os
import sys
source, target, dataset_root, output_dir, model_name = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config["model_name"] = model_name
config["dataset_root"] = dataset_root
config["output_dir"] = output_dir
# Keep a DATASETS subset pilot self-consistent: the original config may list
# nine tasks while a smoke/ablation run intentionally generated fewer.  Reuse
# each configured task's percentages/weights when available and give a newly
# selected task a neutral weight of one.
selected = [name for name in os.environ.get("DATASETS", "").split() if name]
if selected:
    specs = {
        str(spec.get("name")): dict(spec)
        for spec in config.get("datasets", [])
        if isinstance(spec, dict) and spec.get("name")
    }
    config["datasets"] = [specs.get(name, {"name": name, "weight": 1.0}) for name in selected]
if os.environ.get("GRADIENT_ACCUMULATION_STEPS"):
    config["gradient_accumulation_steps"] = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
temporary = target + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(temporary, target)
print(target)
PY
}

run_generate() {
  check_gpu_ids
  log "[generate] relation profiles: MCQ=4 views x 8; WinoGrande=2 unique views x 16; GSM8K/SVAMP=1 identity view x 32"
  log "[generate] model=${MODEL_NAME} datasets=${DATASETS} max_questions=${MAX_QUESTIONS}"
  log "[generate] max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} question_batch_size=${QUESTION_BATCH_SIZE} temperature=${TEMPERATURE}"
  PYTHON_BIN="${PYTHON_BIN}" MODEL_NAME="${MODEL_NAME}" OUTPUT_ROOT="${RAW_ROOT}" \
    LOG_ROOT="${LOG_ROOT}/data_generation" GPU_FIRST="${GPU_FIRST}" GPU_SECOND="${GPU_SECOND}" \
    DATASETS="${DATASETS}" MAX_QUESTIONS="${MAX_QUESTIONS}" SEED="${SEED}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    TEMPERATURE="${TEMPERATURE}" CONFIDENCE_TEMPERATURE="${CONFIDENCE_TEMPERATURE}" \
    HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/01_generate_relational_data.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  log "[generate] complete: ${RAW_ROOT}"
}

run_smoke() {
  check_gpu_ids
  log "[smoke] optional two-question check on physical GPUs ${GPU_FIRST},${GPU_SECOND}"
  log "[smoke] output=${SMOKE_ROOT} (formal STAGE=all does not use this tree)"
  PYTHON_BIN="${PYTHON_BIN}" MODEL_NAME="${MODEL_NAME}" SMOKE_ROOT="${SMOKE_ROOT}" \
    GPU_FIRST="${GPU_FIRST}" GPU_SECOND="${GPU_SECOND}" \
    HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" \
    FORCE_SMOKE_OVERWRITE="${FORCE_SMOKE_OVERWRITE:-0}" \
    MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-256}" \
    MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-2048}" \
    GPU_MEMORY_UTILIZATION="${SMOKE_GPU_MEMORY_UTILIZATION:-0.80}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/03_smoke_train.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  log "[smoke] complete: ${SMOKE_ROOT}"
}

run_build() {
  [[ -d "${RAW_ROOT}" ]] || fail "Raw relational data directory missing: ${RAW_ROOT}; run STAGE=generate first"
  find "${RAW_ROOT}" -type f -path '*/questions/*.json' -print -quit | grep -q . || \
    fail "No RelaCaTS question JSON files under ${RAW_ROOT}; HINT parquet cannot substitute"
  log "[build] computing RelSSC and writing question-disjoint train/test JSONL"
  PYTHON_BIN="${PYTHON_BIN}" INPUT_ROOT="${RAW_ROOT}" OUTPUT_ROOT="${DATASET_ROOT}" \
    TEST_RATIO="${TEST_RATIO}" SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/02_build_relssc_dataset.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  [[ -f "${DATASET_ROOT}/manifest.json" ]] || fail "RelSSC manifest was not produced: ${DATASET_ROOT}/manifest.json"
  log "[build] complete: ${DATASET_ROOT}"
}

run_train() {
  [[ -f "${DATASET_ROOT}/manifest.json" ]] || fail "RelSSC dataset missing: ${DATASET_ROOT}/manifest.json; run STAGE=build first"
  check_gpu_ids
  write_runtime_config
  log "[train] starting distributed LoRA training on physical GPUs ${GPU_FIRST},${GPU_SECOND}"
  log "[train] runtime config=${RUNTIME_CONFIG}; adapter=${CHECKPOINT_ROOT}"
  PYTHON_BIN="${PYTHON_BIN}" CONFIG_FILE="${RUNTIME_CONFIG}" OUTPUT_DIR="${CHECKPOINT_ROOT}" \
    LOG_FILE="${LOG_ROOT}/train_relacats_gpu${GPU_FIRST}${GPU_SECOND}.log" \
    GPU_FIRST="${GPU_FIRST}" GPU_SECOND="${GPU_SECOND}" \
    ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS}" RESUME_FROM="${RESUME_FROM}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/04_train_relacats.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  [[ -f "${CHECKPOINT_ROOT}/adapter_config.json" ]] || fail "Training ended without adapter_config.json: ${CHECKPOINT_ROOT}"
  log "[train] complete: ${CHECKPOINT_ROOT}"
}

main() {
  case "${STAGE}" in
    -h|--help|help)
      usage
      return 0
      ;;
    preflight)
      check_common_inputs
      check_gpu_ids
      log "preflight OK (no model loading or training was started)"
      ;;
    smoke)
      check_common_inputs
      run_smoke
      ;;
    generate)
      check_common_inputs
      run_generate
      ;;
    build)
      check_common_inputs
      run_build
      ;;
    train)
      check_common_inputs
      run_train
      ;;
    all)
      check_common_inputs
      run_generate
      run_build
      run_train
      log "RelaCaTS-v1 training pipeline complete"
      ;;
    *)
      usage >&2
      fail "Unknown STAGE=${STAGE}; choose preflight, smoke, generate, build, train, or all"
      ;;
  esac
}

main "$@"
