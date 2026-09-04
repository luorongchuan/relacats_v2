#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end smoke: real model generation for two ARC questions, 4 relational
# views with 2 responses/view (8 total for smoke), RelSSC construction, then
# one distributed training step.  This is intentionally separate from the
# formal 4x8=32-data path and never writes formal output directories.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MODEL_NAME="${MODEL_NAME:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT_DIR}/relacats_v2/outputs/smoke}"
RAW_ROOT="${RAW_ROOT:-${SMOKE_ROOT}/generated_data}"
DATASET_ROOT="${DATASET_ROOT:-${SMOKE_ROOT}/relssc_dataset}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SMOKE_ROOT}/checkpoint}"
LOG_ROOT="${LOG_ROOT:-${SMOKE_ROOT}/logs}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
# Keep helper executables shipped with the selected environment visible to
# vLLM's child processes (FlashInfer JIT uses ninja).
PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
[[ -f "${MODEL_NAME}/config.json" ]] || fail "Local model not found: ${MODEL_NAME}"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "Smoke GPUs must differ"
mkdir -p "${SMOKE_ROOT}" "${RAW_ROOT}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

# A completed smoke adapter is useful evidence; do not silently overwrite it
# on an accidental second invocation.  Use a new SMOKE_ROOT (recommended) or
# explicitly opt in with FORCE_SMOKE_OVERWRITE=1 when a rerun is intentional.
if [[ -n "$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" \
      && "${FORCE_SMOKE_OVERWRITE:-0}" != "1" ]]; then
  fail "Smoke checkpoint is non-empty: ${CHECKPOINT_ROOT}; use a new SMOKE_ROOT or FORCE_SMOKE_OVERWRITE=1"
fi
exec 9>"${SMOKE_ROOT}/.smoke.lock"
flock -n 9 || fail "another smoke job is using ${SMOKE_ROOT}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
fi

# Keep the two vLLM workers in their own process groups so an interrupted
# smoke run cannot leave model servers behind.  The formal training command is
# intentionally not started until this whole script exits successfully.
GEN_PIDS=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( status != 0 && ${#GEN_PIDS[@]} > 0 )); then
    for pid in "${GEN_PIDS[@]}"; do
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
  fi
  if (( ${#GEN_PIDS[@]} > 0 )); then
    for pid in "${GEN_PIDS[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"
export HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_selfcal_eval}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

COMMON=(
  --model-name "${MODEL_NAME}"
  --datasets arc_easy
  --split train
  --max-questions 2
  --output-root "${RAW_ROOT}"
  --num-views 4
  --samples-per-view 2
  --total-budget 8
  --allow-nonstandard-budget
  --temperature 0.8
  --confidence-temperature 0.0
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
  --max-model-len "${MAX_MODEL_LEN:-2048}"
  --question-batch-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
  --tensor-parallel-size 1
  --seed 42
  --num-shards 2
)

echo "[1/4] Generating two-question relational smoke on GPUs ${GPU_FIRST},${GPU_SECOND}"
setsid env CUDA_VISIBLE_DEVICES="${GPU_FIRST}" "${PYTHON_BIN}" -m relacats_v2.data_creation.generate_relational_data \
  "${COMMON[@]}" --shard-index 0 >"${LOG_ROOT}/generation_gpu${GPU_FIRST}.log" 2>&1 &
first_pid=$!
GEN_PIDS+=("${first_pid}")
setsid env CUDA_VISIBLE_DEVICES="${GPU_SECOND}" "${PYTHON_BIN}" -m relacats_v2.data_creation.generate_relational_data \
  "${COMMON[@]}" --shard-index 1 >"${LOG_ROOT}/generation_gpu${GPU_SECOND}.log" 2>&1 &
second_pid=$!
GEN_PIDS+=("${second_pid}")
set +e
wait "${first_pid}"; first_status=$?
wait "${second_pid}"; second_status=$?
set -e
(( first_status == 0 && second_status == 0 )) || fail \
  "smoke generation failed (${first_status}/${second_status}); see ${LOG_ROOT}"
GEN_PIDS=()

echo "[2/4] Building smoke RelSSC dataset"
"${PYTHON_BIN}" -m relacats_v2.data_creation.build_relssc_dataset \
  --input-root "${RAW_ROOT}" \
  --output-root "${DATASET_ROOT}" \
  --test-ratio 0.5 \
  --seed 42 \
  --allow-nonstandard-budget \
  2>&1 | tee "${LOG_ROOT}/build_relssc.log"

echo "[3/4] Running one distributed RelaCaTS-v1 training update"
CONFIG_FILE="${ROOT_DIR}/relacats_v2/configs/qwen2_5_7b_smoke.json"
TMP_CONFIG="${SMOKE_ROOT}/qwen2_5_7b_smoke_runtime.json"
"${PYTHON_BIN}" - "${CONFIG_FILE}" "${TMP_CONFIG}" "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" <<'PY'
import json, sys
source, target, dataset_root, output_dir = sys.argv[1:]
with open(source, encoding='utf-8') as handle:
    config = json.load(handle)
config['dataset_root'] = dataset_root
config['output_dir'] = output_dir
with open(target, 'w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
PY
CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  relacats_v2/model_training/train_relacats.py \
  --config-file "${TMP_CONFIG}" \
  --save-path "${CHECKPOINT_ROOT}" \
  --max-train-samples 8 \
  --max-eval-samples 8 \
  --max-optimizer-steps 1 \
  --gradient-accumulation-steps 1 \
  2>&1 | tee "${LOG_ROOT}/training_gpu${GPU_FIRST}_${GPU_SECOND}.log"

echo "[4/4] Running CPU evaluation aggregation smoke"
"${PYTHON_BIN}" -m relacats_v2.evaluation.synthetic_smoke \
  --output-dir "${SMOKE_ROOT}/evaluation_smoke" \
  2>&1 | tee "${LOG_ROOT}/evaluation_smoke.log"
echo "RelaCaTS-v1 smoke complete. Outputs: ${SMOKE_ROOT}"
