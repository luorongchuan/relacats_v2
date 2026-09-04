"""CPU contract tests for the complete evaluation method set."""

from __future__ import annotations

from relacats_v2.evaluation.aggregate import (
    AggregateConfig,
    _asc_vote,
    _dynamic_predict,
    _relacats_asc_vote,
    _self_certainty_from_tokens,
    build_threshold_calibration,
    evaluate_records,
    run_aggregation,
)
from relacats_v2.common import atomic_write_jsonl
from relacats_v2.evaluation.synthetic_smoke import (
    EXPECTED_DYNAMIC_METHODS,
    EXPECTED_FIXED_METHODS,
    run_smoke,
)


def test_asc_count_vote_and_relacats_weighted_vote_are_independent():
    """The required toy case separates ordinary and confidence ASC."""

    records = [
        {"extracted_answer": "A", "confidence": 0.9},
        {"extracted_answer": "B", "confidence": 0.1},
        {"extracted_answer": "B", "confidence": 0.1},
    ]
    assert _asc_vote(records) == "B"
    assert _relacats_asc_vote(records) == "A"

    # A threshold above both observed ratios disables early stopping.  Final
    # predictions must still use each method's own vote state.
    asc = _dynamic_predict("ASC", records, 1.0, max_budget=3, min_valid=3)
    weighted = _dynamic_predict(
        "RelaCaTS-ASC", records, 1.0, max_budget=3, min_valid=3
    )
    assert asc[:2] == ("B", 3)
    assert weighted[:2] == ("A", 3)


def _formal_records(split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question, gold in (("q1", "A"), ("q2", "B")):
        for index, (answer, confidence) in enumerate(
            ((gold, 0.9), (gold, 0.8), ("C", 0.2), (gold, 0.7))
        ):
            rows.append(
                {
                    "sample_id": f"{split}-{question}-{index}",
                    "question_id": question,
                    "generation_index": index,
                    "dataset_name": "toy",
                    "split": split,
                    "correct_answer": gold,
                    "extracted_answer": answer,
                    "confidence": confidence,
                }
            )
    return rows


def test_test_phase_requires_and_only_reads_validation_calibration():
    config = AggregateConfig(
        budgets=(4, 16),
        thresholds=(0.0, 0.5, 0.8, 1.0),
        curve_max_budget=4,
        budget_targets=(3,),
        esc_window_sizes=(2, 3, 4),
        rasc_buffer_size=2,
    )
    validation = evaluate_records(
        _formal_records("validation"),
        config=config,
        phase="validation",
        model_id="toy-model",
        dataset_name="toy",
    )
    calibration = build_threshold_calibration(
        validation, model_id="toy-model", dataset_name="toy"
    )
    assert all(
        row["validation_actual_avg_samples"] <= 3
        for row in calibration["selections"]
    )

    test_report = evaluate_records(
        _formal_records("test"),
        config=config,
        phase="test",
        threshold_calibration=calibration,
        model_id="toy-model",
        dataset_name="toy",
    )
    assert test_report["reportable"] is True
    assert all(
        row["selection_rule"]
        == "read fixed parameter from validation artifact"
        for row in test_report["dynamic_budget_matches"]
    )
    for row in (
        test_report["fixed_budget_results"]
        + test_report["dynamic_budget_matches"]
    ):
        assert row["actual_avg_samples"] == row["avg_samples_used"]
        assert row["generated_samples"] == row["valid_samples"] + row["invalid_samples"]
        assert 0.0 <= row["invalid_rate"] <= 1.0


def test_test_phase_rejects_missing_or_wrong_scope_thresholds():
    config = AggregateConfig(
        budgets=(4, 16),
        thresholds=(0.5,),
        curve_max_budget=4,
        budget_targets=(3,),
        esc_window_sizes=(2,),
        rasc_buffer_size=2,
    )
    import pytest

    with pytest.raises(ValueError, match="validation threshold artifact"):
        evaluate_records(
            _formal_records("test"),
            config=config,
            phase="test",
            model_id="toy-model",
            dataset_name="toy",
        )

    validation = evaluate_records(
        _formal_records("validation"),
        config=config,
        phase="validation",
        model_id="toy-model",
        dataset_name="toy",
    )
    calibration = build_threshold_calibration(
        validation, model_id="toy-model", dataset_name="toy"
    )
    with pytest.raises(ValueError, match="model mismatch"):
        evaluate_records(
            _formal_records("test"),
            config=config,
            phase="test",
            threshold_calibration=calibration,
            model_id="different-model",
            dataset_name="toy",
        )


def test_cpu_aggregation_reparses_raw_response_with_v2_parser():
    records = [
        {
            "sample_id": "reparse-0",
            "question_id": "reparse-q",
            "generation_index": 0,
            "dataset_name": "math_qa",
            "answer_type": "option letter",
            "correct_answer": "A",
            "response": "Explanation: intermediate B.\nAnswer: (A)",
            "extracted_answer": "B",
            "confidence": 0.9,
        }
    ]
    report = evaluate_records(
        records,
        config=AggregateConfig(
            budgets=(1, 16),
            thresholds=(0.5,),
            curve_max_budget=2,
            budget_targets=(1,),
            esc_window_sizes=(2,),
        ),
    )
    sc = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "SC" and row["budget"] == 1
    )
    assert sc["correct"] == 1
    assert report["diagnostics"]["strict_response_reparsed_records"] == 1
    assert report["diagnostics"]["legacy_extracted_answer_fallback_records"] == 0
    assert report["diagnostics"]["reparsed_answer_disagreements"] == 1


def test_run_aggregation_persists_validation_then_reads_it_on_test(tmp_path):
    validation_input = tmp_path / "validation.jsonl"
    test_input = tmp_path / "test.jsonl"
    threshold_file = tmp_path / "thresholds" / "toy.json"
    atomic_write_jsonl(validation_input, _formal_records("validation"))
    atomic_write_jsonl(test_input, _formal_records("test"))
    config = AggregateConfig(
        budgets=(4, 16),
        thresholds=(0.0, 0.5, 1.0),
        curve_max_budget=4,
        budget_targets=(3,),
        esc_window_sizes=(2, 3, 4),
        rasc_buffer_size=2,
    )
    validation = run_aggregation(
        [validation_input],
        tmp_path / "validation-results",
        config=config,
        phase="validation",
        threshold_file=threshold_file,
        model_id="toy-model",
        dataset_name="toy",
    )
    assert threshold_file.is_file()
    assert validation["evaluation_phase"] == "validation"

    test_report = run_aggregation(
        [test_input],
        tmp_path / "test-results",
        config=config,
        phase="test",
        threshold_file=threshold_file,
        model_id="toy-model",
        dataset_name="toy",
    )
    assert test_report["evaluation_phase"] == "test"
    assert test_report["threshold_calibration_file"] == str(threshold_file.resolve())
    assert (tmp_path / "test-results" / "evaluation.json").is_file()


def test_dynamic_budget_cap_survives_validation_test_distribution_shift():
    """A validation threshold must never let test execution exceed its target.

    The validation partition stops early for every method, while the test
    partition is deliberately arranged so the same controls would otherwise
    run to the diagnostic 32-response limit.  Target-capped execution must
    still charge at most two responses in both phases.
    """

    def rows(split: str, answers: tuple[str, ...]) -> list[dict[str, object]]:
        result = []
        for index, answer in enumerate(answers):
            result.append(
                {
                    "sample_id": f"{split}-q-{index}",
                    "question_id": "q",
                    "generation_index": index,
                    "dataset_name": "toy",
                    "split": split,
                    "correct_answer": "A",
                    "extracted_answer": answer,
                    "confidence": 0.5,
                }
            )
        return result

    config = AggregateConfig(
        budgets=(2, 16),
        thresholds=(0.5, 1.0),
        curve_max_budget=4,
        budget_targets=(2,),
        esc_window_sizes=(2, 3, 4),
        rasc_buffer_size=2,
    )
    validation = evaluate_records(
        rows("validation", ("A", "A", "A", "A")),
        config=config,
        phase="validation",
        model_id="toy-model",
        dataset_name="toy",
    )
    calibration = build_threshold_calibration(
        validation, model_id="toy-model", dataset_name="toy"
    )
    assert all(
        int(row["budget_cap"]) == 2
        and int(row["validation_budget_cap"]) == 2
        for row in calibration["selections"]
    )

    # The held-out distribution disagrees with validation.  Without a hard
    # target cap these controls would consume all four records.
    test = evaluate_records(
        rows("test", ("A", "B", "B", "B")),
        config=config,
        phase="test",
        threshold_calibration=calibration,
        model_id="toy-model",
        dataset_name="toy",
    )
    assert all(
        int(row["budget_cap"]) == 2
        and float(row["actual_avg_samples"]) <= 2.0
        for row in test["dynamic_budget_matches"]
    )

def test_synthetic_aggregate_emits_all_canonical_methods():
    """Run the tiny no-GPU fixture and verify the public report contract."""

    report = run_smoke()
    fixed = {row["method"] for row in report["fixed_budget_results"]}
    dynamic = set(report["threshold_curves"])
    assert fixed == EXPECTED_FIXED_METHODS
    assert dynamic == EXPECTED_DYNAMIC_METHODS
    assert report["evaluation_namespace"] == "RelaCaTS"
    assert set(report["method_order"]) == fixed | dynamic


def test_optional_cisc_score_is_not_reduced_to_calibrated_proxy():
    """Exported CISC scores must survive grouping and affect its vote.

    Real v1 confidence artifacts normally lack this optional field and are
    therefore explicitly evaluated with the documented calibrated-confidence
    proxy.  When an artifact does provide an untrained CISC score, however, the
    evaluator must not discard it while slimming records into question groups.
    """

    records = []
    # Calibrated confidence favors B, while the independent CISC score favors
    # A.  The gold answer is A, making accidental proxy use observable.
    calibrated = (("B", 0.95, 0.01), ("B", 0.90, 0.02), ("A", 0.10, 0.99), ("A", 0.05, 0.98))
    for index, (answer, confidence, cisc_confidence) in enumerate(calibrated):
        records.append(
            {
                "sample_id": f"cisc-q-r{index}",
                "question_id": "cisc-q",
                "generation_index": index,
                "dataset_name": "synthetic",
                "correct_answer": "A",
                "extracted_answer": answer,
                "confidence": confidence,
                "cisc_confidence": cisc_confidence,
            }
        )

    report = evaluate_records(
        records,
        config=AggregateConfig(
            budgets=(4, 16),
            thresholds=(0.5,),
            curve_max_budget=4,
            budget_targets=(4,),
        ),
    )
    cisc = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "CISC" and row["budget"] == 4
    )
    relacats_sc = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "RelaCaTS-SC" and row["budget"] == 4
    )
    assert cisc["correct"] == 1
    assert relacats_sc["correct"] == 0


def test_esc_checks_non_overlapping_windows():
    """ESC must charge complete sequential windows, not a sliding window."""

    records = [
        {"extracted_answer": answer, "confidence": 0.5}
        for answer in ("A", "B", "B", "B")
    ]
    prediction, used, status = _dynamic_predict(
        "ESC", records, threshold=2, max_budget=4, min_valid=2
    )
    # The first (A,B) window is not unanimous; the second (B,B) window is,
    # so a sliding implementation would incorrectly stop after sample 3.
    assert prediction == "B"
    assert used == 4
    assert status == "early_stop"


def test_rasc_optional_reasoning_score_is_used():
    """A native RASC score must override the calibrated-confidence proxy."""

    records = []
    # Calibrated confidence favors B, while the reasoning/sufficiency score
    # favors A.  A capacity of two makes the selected answer observable.
    for index, (answer, confidence, rasc_score) in enumerate(
        (("B", 0.95, 0.90), ("A", 0.10, 0.95), ("A", 0.05, 0.96), ("B", 0.80, 0.10))
    ):
        records.append(
            {
                "sample_id": f"rasc-q-r{index}",
                "question_id": "rasc-q",
                "generation_index": index,
                "correct_answer": "A",
                "extracted_answer": answer,
                "confidence": confidence,
                "rasc_score": rasc_score,
            }
        )
    prediction, used, status = _dynamic_predict(
        "RASC", records, threshold=0.5, max_budget=4, min_valid=2, rasc_buffer_size=2
    )
    assert prediction == "A"
    assert used == 2
    assert status == "early_stop"


def test_self_certainty_rejects_truncated_top_k_vectors():
    """A top-k export must not be mistaken for a full-vocabulary score."""

    assert (
        _self_certainty_from_tokens(
            {
                "self_certainty_vocab_size": 4,
                "self_certainty_token_probabilities": [[0.25, 0.25]],
            }
        )
        is None
    )
