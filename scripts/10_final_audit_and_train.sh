#!/usr/bin/env bash
set -Eeuo pipefail

# Final CPU audit for the three model-specific RelaCaTS-v1 datasets, followed
# by strictly sequential LoRA training.  This wrapper deliberately invokes
# train_relacats.py directly instead of 04_train_relacats.sh: the latter's
# legacy guard expects a single root-level manifest, whereas this experiment
# keeps one RelSSC tree per base model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
GPU_FIRST="${GPU_FIRST:-6}"
GPU_SECOND="${GPU_SECOND:-7}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/relacats_v2/outputs}"
RAW_BASE="${RAW_BASE:-${OUTPUT_BASE}/generated_data}"
RELSSC_BASE="${RELSSC_BASE:-${OUTPUT_BASE}/relssc_dataset}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${OUTPUT_BASE}/checkpoints}"
LOG_BASE="${LOG_BASE:-${OUTPUT_BASE}/logs/final_audit_and_train_gpu${GPU_FIRST}${GPU_SECOND}}"
HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_relacats_v2}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
DATASETS="${DATASETS:-arc_easy commonsense_qa gsm8k logiqa openbookqa reclor sciq svamp winogrande}"
MODELS="${MODELS:-qwen2_5_7b_instruct llama3_1_8b_instruct deepseek_r1_distill_qwen_1_5b}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
RESUME="${RESUME:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-}"

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME HF_DATASETS_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

EXPECTED_DATASETS=( ${DATASETS} )

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ ]] || fail "GPU_FIRST must be numeric: ${GPU_FIRST}"
[[ "${GPU_SECOND}" =~ ^[0-9]+$ ]] || fail "GPU_SECOND must be numeric: ${GPU_SECOND}"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"

mkdir -p "${LOG_BASE}"

# Prevent two copies of the audit/train wrapper from racing on the same
# per-model output directories.  The descriptor remains locked for the whole
# wrapper lifetime (including all three serial training stages).
exec 8>"${LOG_BASE}/.final_audit_and_train.lock"
if ! flock -n 8; then
  fail "another final audit/train wrapper is already running (lock: ${LOG_BASE}/.final_audit_and_train.lock)"
fi

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
  echo "===== CPU unit tests ====="
  "${PYTHON_BIN}" -m pytest -q relacats_v2/tests
fi

echo "===== final RelSSC audit (CPU; no model loading) ====="
RUN_ROOT="${OUTPUT_BASE}" \
RAW_BASE="${RAW_BASE}" \
RELSSC_BASE="${RELSSC_BASE}" \
DATASETS="${DATASETS}" \
MODELS="${MODELS}" \
"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from relacats_v2.data_creation.build_relssc_dataset import validate_question_payload
from relacats_v2.core import compute_relssc


raw_base = Path(os.environ["RAW_BASE"])
relssc_base = Path(os.environ["RELSSC_BASE"])
datasets = os.environ["DATASETS"].split()
models = os.environ["MODELS"].split()
expected = set(datasets)
if not models:
    raise SystemExit("MODELS is empty")
if not datasets:
    raise SystemExit("DATASETS is empty")


def finite_tree(value, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite number at {path}: {value!r}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    finite_tree(value, str(path))
    return value


def audit_raw(model: str, dataset: str) -> tuple[int, dict[tuple[str, str], float], dict[str, dict[str, float]]]:
    question_dir = raw_base / model / dataset / "questions"
    if not question_dir.is_dir():
        raise AssertionError(f"missing raw question directory: {question_dir}")
    files = sorted(question_dir.glob("*.json"))
    # The requested cap is 1000; SVAMP's source split contains 700 examples.
    expected_count = 700 if dataset == "svamp" else 1000
    if len(files) != expected_count:
        raise AssertionError(
            f"{model}/{dataset}: raw files={len(files)}, expected={expected_count}"
        )
    sample_targets: dict[tuple[str, str], float] = {}
    question_scores: dict[str, dict[str, float]] = {}
    question_ids: set[str] = set()
    for path in files:
        if path.stat().st_size == 0:
            raise AssertionError(f"empty raw JSON: {path}")
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise AssertionError(f"raw payload is not an object: {path}")
        if payload.get("dataset_name") != dataset:
            raise AssertionError(f"dataset mismatch in {path}")
        # This enforces the formal v1 profiles, including WinoGrande 2x16 and
        # GSM8K/SVAMP identity-only 1x32, and recomputes canonicalization.
        validate_question_payload(payload, allow_nonstandard_budget=False)
        question_id = str(payload["question_id"])
        if question_id in question_ids:
            raise AssertionError(f"duplicate raw question_id in {model}/{dataset}: {question_id}")
        question_ids.add(question_id)
        result = compute_relssc(payload["samples"], zero_weight_policy="skip")
        if result.defined:
            scores = {str(answer): float(score) for answer, score in result.scores.items()}
            question_scores[question_id] = scores
            for sample in payload["samples"]:
                if sample.get("is_valid_answer") is not True:
                    continue
                sample_id = str(sample.get("sample_id", ""))
                answer = sample.get("canonicalized_answer")
                if not sample_id or answer is None:
                    raise AssertionError(
                        f"defined raw sample lacks id/answer: {model}/{dataset}/{question_id}"
                    )
                key = (question_id, sample_id)
                if key in sample_targets:
                    raise AssertionError(f"duplicate raw sample_id: {model}/{dataset}/{key}")
                sample_targets[key] = float(result.scores[str(answer).strip()])
    return len(files), sample_targets, question_scores


def audit_relssc(
    model: str,
    raw_count: dict[str, int],
    sample_targets: dict[str, dict[tuple[str, str], float]],
    question_scores: dict[str, dict[str, dict[str, float]]],
) -> None:
    root = relssc_base / model
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AssertionError(f"missing manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    manifest_specs = manifest.get("datasets")
    if not isinstance(manifest_specs, list):
        raise AssertionError(f"manifest.datasets is not a list: {manifest_path}")
    manifest_names = {
        str(spec.get("dataset_name"))
        for spec in manifest_specs
        if isinstance(spec, dict)
    }
    if manifest_names != expected:
        raise AssertionError(
            f"{model}: manifest datasets={sorted(manifest_names)}, expected={sorted(expected)}"
        )
    if manifest.get("formal_budget_enforced") is not True:
        raise AssertionError(f"{model}: formal_budget_enforced is not true")
    if manifest.get("split_unit") != "original_question_id":
        raise AssertionError(f"{model}: unexpected split unit")
    if manifest.get("gold_used_in_target") is not False:
        raise AssertionError(f"{model}: gold_used_in_target must be false")

    stats_by_name = {}
    for spec in manifest_specs:
        if not isinstance(spec, dict):
            raise AssertionError(f"{model}: non-object manifest dataset entry")
        name = str(spec.get("dataset_name"))
        stats_by_name[name] = spec

    for dataset_name in datasets:
        dataset_dir = root / dataset_name
        for filename in ("train.jsonl", "test.jsonl", "stats.json", "question_summaries.json"):
            if not (dataset_dir / filename).is_file():
                raise AssertionError(f"missing {model}/{dataset_name}/{filename}")
        stats = load_json(dataset_dir / "stats.json")
        if not isinstance(stats, dict):
            raise AssertionError(f"stats is not an object: {dataset_dir / 'stats.json'}")
        if stats.get("source_question_files") != raw_count[dataset_name]:
            raise AssertionError(f"{model}/{dataset_name}: source count mismatch")
        if stats.get("defined_questions", 0) + stats.get("skipped_zero_weight_questions", 0) != raw_count[dataset_name]:
            raise AssertionError(f"{model}/{dataset_name}: defined/skipped mismatch")
        if stats.get("train_records", 0) + stats.get("test_records", 0) != stats.get("valid_training_records"):
            raise AssertionError(f"{model}/{dataset_name}: row-count mismatch in stats")
        if stats.get("group_split_no_question_leakage") is not True:
            raise AssertionError(f"{model}/{dataset_name}: leakage flag is false")
        if stats.get("gold_used_in_target") is not False:
            raise AssertionError(f"{model}/{dataset_name}: gold target flag is true")
        mean_target = stats.get("mean_relssc_target")
        if mean_target is not None and (not isinstance(mean_target, (int, float)) or not math.isfinite(float(mean_target))):
            raise AssertionError(f"{model}/{dataset_name}: non-finite mean target")

        split_ids = {}
        split_rows = {}
        for split in ("train", "test"):
            ids = set()
            rows = 0
            with (dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise AssertionError(f"blank line at {dataset_dir}/{split}.jsonl:{line_number}")
                    row = json.loads(line)
                    finite_tree(row, f"{dataset_dir}/{split}.jsonl:{line_number}")
                    if row.get("dataset_name") != dataset_name:
                        raise AssertionError(f"dataset field mismatch at {dataset_dir}/{split}.jsonl:{line_number}")
                    question_id = row.get("question_id")
                    if not question_id:
                        raise AssertionError(f"missing question_id at {dataset_dir}/{split}.jsonl:{line_number}")
                    ids.add(str(question_id))
                    sample_id = str(row.get("sample_id", ""))
                    target_key = (str(question_id), sample_id)
                    if target_key not in sample_targets[dataset_name]:
                        raise AssertionError(
                            f"missing raw target source at {dataset_dir}/{split}.jsonl:{line_number}"
                        )
                    target = row.get("relational_consistency")
                    if not isinstance(target, (int, float)) or not math.isfinite(float(target)) or not 0.0 <= float(target) <= 1.0:
                        raise AssertionError(f"invalid RelSSC target at {dataset_dir}/{split}.jsonl:{line_number}")
                    expected_target = sample_targets[dataset_name][target_key]
                    if not math.isclose(float(target), expected_target, rel_tol=0.0, abs_tol=1e-9):
                        raise AssertionError(
                            f"RelSSC mismatch at {dataset_dir}/{split}.jsonl:{line_number}: "
                            f"stored={target!r}, recomputed={expected_target!r}"
                        )
                    scores = row.get("question_relssc_scores")
                    expected_scores = question_scores[dataset_name].get(str(question_id))
                    if not isinstance(scores, dict) or expected_scores is None:
                        raise AssertionError(
                            f"missing question RelSSC scores at {dataset_dir}/{split}.jsonl:{line_number}"
                        )
                    if set(str(key) for key in scores) != set(expected_scores):
                        raise AssertionError(
                            f"question RelSSC score keys mismatch at {dataset_dir}/{split}.jsonl:{line_number}"
                        )
                    for answer, expected_score in expected_scores.items():
                        stored_score = scores.get(answer)
                        if not isinstance(stored_score, (int, float)) or not math.isclose(
                            float(stored_score), expected_score, rel_tol=0.0, abs_tol=1e-9
                        ):
                            raise AssertionError(
                                f"question RelSSC score mismatch at {dataset_dir}/{split}.jsonl:{line_number}"
                            )
                    rows += 1
            split_ids[split] = ids
            split_rows[split] = rows
        if split_ids["train"] & split_ids["test"]:
            raise AssertionError(f"{model}/{dataset_name}: question leakage detected")
        if split_rows["train"] != stats.get("train_records") or split_rows["test"] != stats.get("test_records"):
            raise AssertionError(f"{model}/{dataset_name}: JSONL rows disagree with stats")
        summaries = load_json(dataset_dir / "question_summaries.json")
        if not isinstance(summaries, list) or len(summaries) != raw_count[dataset_name]:
            raise AssertionError(f"{model}/{dataset_name}: question_summaries count mismatch")
        if stats_by_name[dataset_name].get("train_records") != stats.get("train_records"):
            raise AssertionError(f"{model}/{dataset_name}: manifest/stats disagreement")


for model in models:
    raw_audit = {dataset: audit_raw(model, dataset) for dataset in datasets}
    raw_counts = {dataset: item[0] for dataset, item in raw_audit.items()}
    sample_targets = {dataset: item[1] for dataset, item in raw_audit.items()}
    question_scores = {dataset: item[2] for dataset, item in raw_audit.items()}
    audit_relssc(model, raw_counts, sample_targets, question_scores)
    print(f"AUDIT OK: {model} ({len(datasets)} datasets; raw={sum(raw_counts.values())} question files)")
print("FINAL RELSSC AUDIT PASSED")
PY

if [[ "${AUDIT_ONLY}" == "1" ]]; then
  echo "AUDIT_ONLY=1: audit finished; no GPU training was started."
  exit 0
fi

check_gpu_free() {
  if [[ "${ALLOW_BUSY_GPUS}" == "1" ]]; then
    echo "ALLOW_BUSY_GPUS=1: skipping compute-process safety check"
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required for training preflight"
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    local pids
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
}

make_runtime_config() {
  local source_config="$1"
  local target_config="$2"
  local model_name="$3"
  local dataset_root="$4"
  local output_dir="$5"
  mkdir -p "$(dirname "${target_config}")"
  "${PYTHON_BIN}" - "${source_config}" "${target_config}" "${model_name}" "${dataset_root}" "${output_dir}" "${GRADIENT_ACCUMULATION_STEPS}" <<'PY'
import json
import os
import sys

source, target, model_name, dataset_root, output_dir, accumulation = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config["model_name"] = model_name
config["dataset_root"] = dataset_root
config["output_dir"] = output_dir
if accumulation:
    config["gradient_accumulation_steps"] = int(accumulation)
temporary = target + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(temporary, target)
PY
}

train_one() {
  local tag="$1"
  local model_name config_name
  case "${tag}" in
    qwen2_5_7b_instruct)
      model_name="/home/luorongchuan/workspace_135/models/Qwen2.5-7B-Instruct"
      config_name="qwen2_5_7b_2xa100.json"
      ;;
    llama3_1_8b_instruct)
      model_name="/home/luorongchuan/workspace_135/models/Llama-3.1-8B-Instruct"
      config_name="llama3_1_8b_2xa100.json"
      ;;
    deepseek_r1_distill_qwen_1_5b)
      model_name="/home/luorongchuan/workspace_135/models/DeepSeek-R1-Distill-Qwen-1.5B"
      config_name="deepseek_1_5b_2xa100.json"
      ;;
    *) fail "unknown model tag: ${tag}" ;;
  esac

  local dataset_root="${RELSSC_BASE}/${tag}"
  local output_dir="${CHECKPOINT_BASE}/${tag}_relacats_v2"
  local model_log_dir="${LOG_BASE}/${tag}"
  local runtime_config="${model_log_dir}/runtime_train_config.json"
  local source_config="${ROOT_DIR}/relacats_v2/configs/${config_name}"
  local train_log="${model_log_dir}/train.log"
  [[ -f "${model_name}/config.json" ]] || fail "base model missing: ${model_name}"
  [[ -f "${dataset_root}/manifest.json" ]] || fail "RelSSC manifest missing: ${dataset_root}/manifest.json"
  [[ -f "${source_config}" ]] || fail "training config missing: ${source_config}"
  mkdir -p "${model_log_dir}"

  local resume_args=()
  if [[ -d "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
    if [[ "${RESUME}" == "1" && -f "${output_dir}/adapter_config.json" ]]; then
      resume_args+=(--resume-from "${output_dir}")
      echo "===== RESUME ${tag} (LoRA weights only; optimizer/scheduler restart) ====="
    else
      fail "checkpoint directory is non-empty; refusing overwrite: ${output_dir}. Set RESUME=1 to resume LoRA weights."
    fi
  else
    echo "===== TRAIN ${tag} ====="
  fi

  make_runtime_config "${source_config}" "${runtime_config}" "${model_name}" "${dataset_root}" "${output_dir}"
  check_gpu_free
  local optional_args=()
  [[ -n "${MAX_TRAIN_SAMPLES}" ]] && optional_args+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
  [[ -n "${MAX_EVAL_SAMPLES}" ]] && optional_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
  [[ -n "${MAX_OPTIMIZER_STEPS}" ]] && optional_args+=(--max-optimizer-steps "${MAX_OPTIMIZER_STEPS}")
  [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]] && optional_args+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")

  echo "model=${model_name}"
  echo "dataset_root=${dataset_root}"
  echo "checkpoint=${output_dir}"
  echo "log=${train_log}"
  CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 \
      relacats_v2/model_training/train_relacats.py \
      --config-file "${runtime_config}" \
      --save-path "${output_dir}" \
      "${resume_args[@]}" "${optional_args[@]}" \
      2>&1 | tee "${train_log}"
  [[ -f "${output_dir}/adapter_config.json" ]] || fail "training finished without adapter_config.json: ${output_dir}"
  echo "TRAIN COMPLETE: ${tag} -> ${output_dir}"
}

for tag in ${MODELS}; do
  train_one "${tag}"
done

echo "ALL REQUESTED RELACATS-v1 TRAINING RUNS COMPLETE"
