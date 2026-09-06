#!/usr/bin/env bash
set -Eeuo pipefail

# One-click paired Qwen experiment:
#   preflight -> optional short smoke -> full SSC/RelSC training -> LoRA merge
#   -> Table-2 style evaluation on GPUs 6+7 -> paired comparison report.
#
# OOM-safe high-throughput A100-80GB profile for GPUs 6/7:
#   per-rank micro-batch = 13
#   world size           = 2
#   grad accumulation    = 5
#   effective update     = 130 records
#   gradient checkpointing disabled
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#
# B=14 reached roughly 75 GiB before the fp32 causal-LM loss conversion and
# OOMed on a longer full-training batch.  B=13 leaves one additional record's
# activation/logit headroom while still keeping the two A100s highly utilized.
#
# The two full training runs use exactly the same paired dataset, sampling
# fields, seed, 100k/1k budgets, high-throughput batch profile and all other
# hyperparameters.  Only target_field differs:
#   SSC   : ssc_consistency
#   RelSC : relsc_consistency
#
# NOTE: effective batch 130 is the high-throughput paired protocol, not the
# original batch-128 CaTS reproduction.  It is valid for the controlled
# SSC-vs-RelSC target comparison because both runs use the identical profile.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
BASE_MODEL="${BASE_MODEL:-/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct}"

SSC_CONFIG="${SSC_CONFIG:-relacats_v2/configs/qwen2_5_7b_paired_ssc.json}"
RELSC_CONFIG="${RELSC_CONFIG:-relacats_v2/configs/qwen2_5_7b_paired_relsc.json}"

SSC_ADAPTER="${SSC_ADAPTER:-${ROOT_DIR}/relacats_v2/outputs/checkpoints/qwen2_5_7b_paired_ssc}"
RELSC_ADAPTER="${RELSC_ADAPTER:-${ROOT_DIR}/relacats_v2/outputs/checkpoints/qwen2_5_7b_paired_relsc}"
SSC_MERGED="${SSC_MERGED:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_paired_ssc}"
RELSC_MERGED="${RELSC_MERGED:-${ROOT_DIR}/relacats_v2/outputs/merged_model/qwen2_5_7b_paired_relsc}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/relacats_v2/outputs/logs/paired_qwen_oneclick}"
EVAL_ROOT="${EVAL_ROOT:-${ROOT_DIR}/relacats_v2/outputs/paired_qwen_eval}"
COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/relacats_v2/outputs/paired_qwen_comparison}"

RUN_TESTS="${RUN_TESTS:-1}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_FULL="${RUN_FULL:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

# High-throughput training knobs.  Keep these identical for SSC and RelSC.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-13}"
TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-5}"
TRAIN_GRADIENT_CHECKPOINTING="${TRAIN_GRADIENT_CHECKPOINTING:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-8192}"
SMOKE_EVAL_SAMPLES="${SMOKE_EVAL_SAMPLES:-64}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"

DATASETS="${DATASETS:-object_counting math_qa arc_challenge}"
NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
BUDGET_TARGETS="${BUDGET_TARGETS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-8}"
CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}" "${COMPARE_ROOT}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -x "${PYTHON_BIN}" ]] || fail "Python not found: ${PYTHON_BIN}"
[[ -f "${SSC_CONFIG}" ]] || fail "Missing SSC config: ${SSC_CONFIG}"
[[ -f "${RELSC_CONFIG}" ]] || fail "Missing RelSC config: ${RELSC_CONFIG}"
[[ -f "${BASE_MODEL}/config.json" ]] || fail "Base model not found: ${BASE_MODEL}"
[[ "${TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_BATCH_SIZE must be a positive integer"
[[ "${TRAIN_GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_GRAD_ACCUM_STEPS must be a positive integer"
[[ "${TRAIN_GRADIENT_CHECKPOINTING}" == "0" || "${TRAIN_GRADIENT_CHECKPOINTING}" == "1" ]] || \
  fail "TRAIN_GRADIENT_CHECKPOINTING must be 0 or 1"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

EFFECTIVE_UPDATE_BATCH=$(( TRAIN_BATCH_SIZE * 2 * TRAIN_GRAD_ACCUM_STEPS ))
# With 100k records, DDP gives 50k records/rank. DataLoader(drop_last=True)
# followed by integer gradient accumulation yields this exact one-epoch count.
MICROBATCHES_PER_RANK=$(( 50000 / TRAIN_BATCH_SIZE ))
EXPECTED_FULL_UPDATES=$(( MICROBATCHES_PER_RANK / TRAIN_GRAD_ACCUM_STEPS ))

check_gpu_idle() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    return 0
  fi
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" \
    || fail "Unable to query GPU ${gpu}"
  [[ -z "${pids}" ]] || fail "GPU ${gpu} is busy (PID(s): ${pids}); finish/stop any previous run before starting the one-click run"
}

check_pairing() {
  "${PYTHON_BIN}" - <<'PY'
import json
from relacats_v2.model_training.train_hybrid_relsc_ssc import prepare_hybrid_examples

root = "/home/luorongchuan/workspace_135/RelaCaTS/"
with open(root + "relacats_v2/configs/qwen2_5_7b_paired_ssc.json") as f:
    a = json.load(f)
with open(root + "relacats_v2/configs/qwen2_5_7b_paired_relsc.json") as f:
    b = json.load(f)

xa = prepare_hybrid_examples(a, "train", a["total_train_samples"], a["seed"])
xb = prepare_hybrid_examples(b, "train", b["total_train_samples"], b["seed"])

def key(r):
    return (r["task"], r["dataset_name"], r["transformed_prompt"], r["response"])

ka = [key(r) for r in xa]
kb = [key(r) for r in xb]
assert len(xa) == 100000 and len(xb) == 100000
assert ka == kb
assert sum(r["task"] == "calibration" for r in xa) == 30000
assert sum(r["task"] == "causal_lm" for r in xa) == 70000
changed = sum(
    1 for ra, rb in zip(xa, xb)
    if ra["task"] == "calibration" and abs(float(ra["target"]) - float(rb["target"])) > 1e-12
)
assert changed > 0
print(f"PAIRING PASS: rows=100000, calibration=30000, causal=70000, changed_targets={changed}")
PY
}

training_profile_args() {
  PROFILE_ARGS=(
    --batch-size "${TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${TRAIN_GRAD_ACCUM_STEPS}"
  )
  if [[ "${TRAIN_GRADIENT_CHECKPOINTING}" == "1" ]]; then
    PROFILE_ARGS+=(--gradient-checkpointing)
  else
    PROFILE_ARGS+=(--no-gradient-checkpointing)
  fi
}

train_run() {
  local tag="$1"
  local config="$2"
  local save_path="$3"
  local log_path="$4"
  shift 4

  training_profile_args
  echo "===== TRAIN ${tag} ====="
  echo "profile: micro_batch=${TRAIN_BATCH_SIZE}, world_size=2, grad_accum=${TRAIN_GRAD_ACCUM_STEPS}, effective_update_batch=${EFFECTIVE_UPDATE_BATCH}, gradient_checkpointing=${TRAIN_GRADIENT_CHECKPOINTING}"
  CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=2 \
    -m relacats_v2.model_training.train_hybrid_relsc_ssc \
    --config-file "${config}" \
    --save-path "${save_path}" \
    "${PROFILE_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${log_path}"
}

full_checkpoint_complete() {
  local path="$1"
  [[ -f "${path}/adapter_config.json" && -f "${path}/trainer_state.pt" ]]
}

merge_one() {
  local tag="$1"
  local adapter="$2"
  local output="$3"
  local marker="${output}/.merge_complete"

  if [[ -f "${marker}" && -f "${output}/config.json" ]]; then
    echo "MERGE SKIP ${tag}: complete merged model already exists: ${output}"
    return 0
  fi
  if [[ -d "${output}" && -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "Merged directory is non-empty but has no completion marker: ${output}"
  fi
  echo "===== MERGE ${tag} ====="
  "${PYTHON_BIN}" -m relacats_v2.model_training.merge_lora \
    --base-model "${BASE_MODEL}" \
    --lora-path "${adapter}" \
    --output-path "${output}"
  touch "${marker}"
}

echo "===== HIGH-THROUGHPUT PAIRED QWEN PROFILE ====="
echo "GPUs: ${GPU_FIRST},${GPU_SECOND}"
echo "micro_batch_per_rank=${TRAIN_BATCH_SIZE}"
echo "gradient_accumulation=${TRAIN_GRAD_ACCUM_STEPS}"
echo "effective_update_batch=${EFFECTIVE_UPDATE_BATCH}"
echo "expected_full_optimizer_updates_per_model=${EXPECTED_FULL_UPDATES}"
echo "gradient_checkpointing=${TRAIN_GRADIENT_CHECKPOINTING}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "===== PREFLIGHT TESTS ====="
  "${PYTHON_BIN}" -m pytest -q \
    relacats_v2/tests/test_paired_ssc_relsc_dataset.py \
    relacats_v2/tests/test_hybrid_relsc_ssc_dataset.py
  check_pairing
fi

check_gpu_idle "${GPU_FIRST}"
check_gpu_idle "${GPU_SECOND}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  smoke_root="${ROOT_DIR}/relacats_v2/outputs/checkpoints/paired_qwen_smoke_b${TRAIN_BATCH_SIZE}_a${TRAIN_GRAD_ACCUM_STEPS}"
  mkdir -p "${smoke_root}"
  train_run \
    "SSC SMOKE" "${SSC_CONFIG}" "${smoke_root}/ssc" "${LOG_ROOT}/ssc_smoke.log" \
    --max-train-samples "${SMOKE_TRAIN_SAMPLES}" \
    --max-eval-samples "${SMOKE_EVAL_SAMPLES}" \
    --max-optimizer-steps "${SMOKE_STEPS}"
  train_run \
    "RelSC SMOKE" "${RELSC_CONFIG}" "${smoke_root}/relsc" "${LOG_ROOT}/relsc_smoke.log" \
    --max-train-samples "${SMOKE_TRAIN_SAMPLES}" \
    --max-eval-samples "${SMOKE_EVAL_SAMPLES}" \
    --max-optimizer-steps "${SMOKE_STEPS}"
fi

if [[ "${RUN_FULL}" == "1" ]]; then
  if full_checkpoint_complete "${SSC_ADAPTER}"; then
    echo "FULL TRAIN SKIP SSC: completed adapter exists: ${SSC_ADAPTER}"
  else
    train_run "SSC FULL" "${SSC_CONFIG}" "${SSC_ADAPTER}" "${LOG_ROOT}/ssc_full.log"
  fi

  if full_checkpoint_complete "${RELSC_ADAPTER}"; then
    echo "FULL TRAIN SKIP RelSC: completed adapter exists: ${RELSC_ADAPTER}"
  else
    train_run "RelSC FULL" "${RELSC_CONFIG}" "${RELSC_ADAPTER}" "${LOG_ROOT}/relsc_full.log"
  fi
fi

if [[ "${RUN_MERGE}" == "1" ]]; then
  full_checkpoint_complete "${SSC_ADAPTER}" || fail "SSC full adapter incomplete: ${SSC_ADAPTER}"
  full_checkpoint_complete "${RELSC_ADAPTER}" || fail "RelSC full adapter incomplete: ${RELSC_ADAPTER}"
  merge_one "SSC" "${SSC_ADAPTER}" "${SSC_MERGED}"
  merge_one "RelSC" "${RELSC_ADAPTER}" "${RELSC_MERGED}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  [[ -f "${SSC_MERGED}/config.json" ]] || fail "SSC merged model missing: ${SSC_MERGED}"
  [[ -f "${RELSC_MERGED}/config.json" ]] || fail "RelSC merged model missing: ${RELSC_MERGED}"

  echo "===== SERIAL TABLE-2 EVALUATION: SSC then RelSC ====="
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_FIRST="${GPU_FIRST}" \
    GPU_SECOND="${GPU_SECOND}" \
    MODEL_SPECS="qwen_paired_ssc=${SSC_MERGED} qwen_paired_relsc=${RELSC_MERGED}" \
    BASE_OUTPUT_ROOT="${EVAL_ROOT}" \
    BASE_LOG_ROOT="${LOG_ROOT}/evaluation" \
    DATASETS="${DATASETS}" \
    NUM_GENERATIONS="${NUM_GENERATIONS}" \
    BUDGET_TARGETS="${BUDGET_TARGETS}" \
    EVAL_PHASE="test" \
    EVAL_SPLIT="test" \
    AGGREGATION_PHASE="paper" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE}" \
    CONFIDENCE_BATCH_SIZE="${CONFIDENCE_BATCH_SIZE}" \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/relacats_v2/scripts/12_evaluate_serial_tp2_gpu67.sh"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
  echo "===== PAIRED COMPARISON REPORT ====="
  read -r -a DATASET_ARRAY <<< "${DATASETS}"
  "${PYTHON_BIN}" -m relacats_v2.evaluation.compare_paired_qwen \
    --eval-root "${EVAL_ROOT}" \
    --ssc-tag qwen_paired_ssc \
    --relsc-tag qwen_paired_relsc \
    --datasets "${DATASET_ARRAY[@]}" \
    --budget "${BUDGET_TARGETS%%,*}" \
    --output-dir "${COMPARE_ROOT}"
fi

echo
echo "ALL PAIRED QWEN STAGES COMPLETE"
echo "Training profile: micro_batch=${TRAIN_BATCH_SIZE}, grad_accum=${TRAIN_GRAD_ACCUM_STEPS}, effective_batch=${EFFECTIVE_UPDATE_BATCH}, gradient_checkpointing=${TRAIN_GRADIENT_CHECKPOINTING}"
echo "SSC adapter:      ${SSC_ADAPTER}"
echo "RelSC adapter:    ${RELSC_ADAPTER}"
echo "SSC merged:       ${SSC_MERGED}"
echo "RelSC merged:     ${RELSC_MERGED}"
echo "Evaluation root:  ${EVAL_ROOT}"
echo "Comparison:       ${COMPARE_ROOT}/paired_comparison.md"
