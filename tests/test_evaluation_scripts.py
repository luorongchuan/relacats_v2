"""Static checks for the shell entry points used by test-time evaluation.

These tests deliberately do not invoke vLLM or touch a GPU.  They guard the
small amount of wiring that is easy to regress when the evaluator gains a new
method or evaluation protocol: the default merged-checkpoint tag, canonical
report labels, and the CaTS paper-style budget-only path.
"""

from __future__ import annotations

from pathlib import Path
import re


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CANONICAL_METHODS = (
    "SC",
    "CISC",
    "Self-Certainty",
    "Best-of-N",
    "ASC",
    "ESC",
    "RASC",
    "RelaCaTS-SC",
    "RelaCaTS-ES",
    "RelaCaTS-ASC",
)


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_generation_and_confidence_defaults_match_merged_checkpoint_tag() -> None:
    """The out-of-box qwen default must name the actual instruct adapter."""

    expected = "qwen2_5_7b_instruct_relacats_v2"
    for script in ("06_generate_eval.sh", "07_calculate_confidence.sh"):
        text = _read(script)
        assert expected in text
        # A stale default silently fails before any worker starts.  Keep this
        # typo-specific assertion so a future refactor cannot reintroduce it.
        assert "qwen2_5_7b_relacats_v2" not in text


def test_qwen_training_config_uses_the_same_checkpoint_tag() -> None:
    config = (SCRIPTS.parent / "configs" / "qwen2_5_7b_2xa100.json").read_text(
        encoding="utf-8"
    )
    assert "qwen2_5_7b_instruct_relacats_v2" in config
    assert "qwen2_5_7b_relacats_v2" not in config


def test_evaluation_launcher_advertises_every_canonical_method() -> None:
    text = _read("08_evaluate.sh")
    for method in CANONICAL_METHODS:
        assert method in text
    # Legacy project labels are accepted by the Python compatibility layer,
    # but must not be presented as newly generated output names.
    assert not re.search(r"(?<!Rela)CaTS-(?:SC|ES|ASC)\b", text)
    for option in (
        "BUDGETS",
        "CURVE_MAX_BUDGET",
        "BUDGET_TARGETS",
        "DYNAMIC_MIN_VALID",
        "RASC_BUFFER_SIZE",
        "ESC_WINDOW_SIZES",
        "CISC_TEMPERATURE",
        "CISC_NORMALIZATION",
    ):
        assert f'{option}="${{{option}:-' in text


def test_evaluation_launcher_exposes_paper_budget_protocol() -> None:
    text = _read("08_evaluate.sh")
    assert '"${PHASE}" == "paper"' in text
    assert "relacats_v2.evaluation.paper_budget" in text
    assert "accuracy is excluded from parameter selection" in text
    # The strict validation/test path still goes through aggregate.py and a
    # threshold artifact; paper mode must not be forced to fabricate one.
    assert '--threshold-file "${THRESHOLD_ROOT}/${dataset}.json"' in text


def test_confidence_launcher_describes_shared_artifact_consumers() -> None:
    """06/07 produce one pool consumed by all baseline and RelaCaTS rows."""

    text = _read("07_calculate_confidence.sh")
    assert "shared evaluation confidence" in text
    for method in CANONICAL_METHODS:
        assert method in text


def test_generation_log_uses_configured_response_budget() -> None:
    text = _read("06_generate_eval.sh")
    assert 'num_generations="${NUM_GENERATIONS:-32}"' in text
    assert "${num_generations} shared test-time responses/question" in text


def test_generation_supports_read_only_tokenizer_override() -> None:
    text = _read("06_generate_eval.sh")
    assert 'TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-}"' in text
    assert "--tokenizer-source" in text


def test_serial_runner_uses_one_tp_worker_per_model() -> None:
    """The reusable entry point must not regress to two model replicas."""

    text = _read("12_evaluate_serial_tp2_gpu67.sh")
    assert 'EVAL_GPU_MODE="${EVAL_GPU_MODE:-tensor_parallel}"' in text
    assert 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"' in text
    assert 'NUM_SHARDS="${NUM_SHARDS:-1}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"' in text
    assert 'MODEL_SPECS="${MODEL_SPECS:-' in text
    assert "for spec in \"${MODEL_SPEC_ARRAY[@]}\"" in text
    # A model is advanced only after both GPU stages and CPU aggregation have
    # returned; this is the key serialisation contract.
    assert '===== COMPLETE ${tag}:' in text
    assert 'ALL SERIAL RELACATS-V2 ${EVAL_PHASE^^} RUNS COMPLETE' in text


def test_serial_runner_separates_test_split_from_paper_aggregation() -> None:
    text = _read("12_evaluate_serial_tp2_gpu67.sh")
    assert 'EVAL_SPLIT="${EVAL_SPLIT:-${EVAL_PHASE}}"' in text
    assert 'AGGREGATION_PHASE="${AGGREGATION_PHASE:-${EVAL_PHASE}}"' in text
    assert 'PHASE="${AGGREGATION_PHASE}"' in text
    assert 'AGGREGATION_PHASE=paper requires EVAL_SPLIT=test' in text
    assert 'aggregation_phase=${AGGREGATION_PHASE}' in text


def test_old_model_retest_is_serial_tp2_and_never_trains_or_merges() -> None:
    text = _read("14_retest_old_models_gpu67.sh")
    assert 'EVAL_GPU_MODE=tensor_parallel' in text
    assert 'TENSOR_PARALLEL_SIZE=2' in text
    assert 'NUM_SHARDS=1' in text
    assert 'CUDA_VISIBLE_DEVICES="${GPU_FIRST},${GPU_SECOND}"' in text
    assert 'for index in "${!MODEL_TAGS[@]}"' in text
    assert '06_generate_eval.sh' in text
    assert '07_calculate_confidence.sh' in text
    assert 'retest_old_models' in text
    assert 'train_relacats' not in text
    assert 'merge_lora' not in text
    assert '05_merge_model.sh' not in text
    assert 'outputs/eval_outputs_v2_retest_old_models' in text
    assert 'unset MAX_QUESTIONS' in text


def test_all_evaluation_stages_accept_tensor_parallel_mode() -> None:
    for script in ("06_generate_eval.sh", "07_calculate_confidence.sh"):
        text = _read(script)
        assert "EVAL_GPU_MODE" in text
        assert "tensor_parallel" in text
        assert "--tensor-parallel-size" in text
        assert "NUM_SHARDS" in text
    text = _read("08_evaluate.sh")
    assert 'NUM_SHARDS="${NUM_SHARDS:-2}"' in text
    assert "shard-$(printf '%05d'" in text
