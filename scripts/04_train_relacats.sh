#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/relacats_v2/configs/qwen2_5_7b_2xa100.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/relacats_v2/outputs/checkpoints/qwen2_5_7b_instruct_relacats_v2}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/relacats_v2/outputs/logs/train_qwen2_5_7b_gpu67.log}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${CONFIG_FILE}" ]] || { echo "Config not found: ${CONFIG_FILE}" >&2; exit 1; }
# The three formal model runs use independent RelSSC trees.  Resolve the
# dataset root from the actual config instead of assuming a single shared
# manifest under outputs/relssc_dataset.  An explicit DATASET_ROOT_CHECK can
# still be supplied by callers that want to override the config only for this
# preflight check.
DATASET_ROOT_CHECK="${DATASET_ROOT_CHECK:-}"
if [[ -z "${DATASET_ROOT_CHECK}" ]]; then
  DATASET_ROOT_CHECK="$(${PYTHON_BIN} - "${CONFIG_FILE}" "${ROOT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

config_path, root_dir = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
value = config.get("dataset_root")
if not value:
    raise SystemExit("training config has no dataset_root")
path = Path(str(value)).expanduser()
if not path.is_absolute():
    path = Path(root_dir) / path
print(path.resolve())
PY
)"
fi
[[ -f "${DATASET_ROOT_CHECK}/manifest.json" ]] || {
  echo "RelSSC dataset manifest missing: ${DATASET_ROOT_CHECK}/manifest.json" >&2
  echo "Run 02_build_relssc_dataset.sh for this model first" >&2
  exit 1
}
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" && -z "${RESUME_FROM:-}" ]]; then
  echo "Refusing to overwrite non-empty checkpoint: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")" "${OUTPUT_DIR}"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || {
  echo "GPU_FIRST and GPU_SECOND must differ" >&2
  exit 1
}

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || {
      echo "physical GPU ${gpu} is busy (PID(s): ${pids}); wait or set ALLOW_BUSY_GPUS=1 intentionally" >&2
      exit 1
    }
  done
fi

exec 9>"${OUTPUT_DIR}.train.lock"
flock -n 9 || {
  echo "another RelaCaTS training job is using ${OUTPUT_DIR}" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ARGS=(
  --config-file "${CONFIG_FILE}"
  --save-path "${OUTPUT_DIR}"
)
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from "${RESUME_FROM}"); fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  ARGS+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${TARGET_MODE:-}" ]]; then ARGS+=(--target-mode "${TARGET_MODE}"); fi
if [[ -n "${LAMBDA_REL:-}" ]]; then ARGS+=(--lambda-rel "${LAMBDA_REL}"); fi

echo "Starting RelaCaTS v2 on physical GPUs ${GPU_FIRST},${GPU_SECOND}; log=${LOG_FILE}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  relacats_v2/model_training/train_relacats.py "${ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
