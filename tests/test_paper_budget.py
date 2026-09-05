"""CPU-only tests for the CaTS paper-style average-budget protocol."""

from __future__ import annotations

import pytest

from relacats_v2.common import atomic_write_jsonl
from relacats_v2.evaluation.aggregate import AggregateConfig, DYNAMIC_METHODS
from relacats_v2.evaluation.paper_budget import (
    SELECTION_RULE,
    _select_budget_only_matches,
    run_paper_aggregation,
)


def test_budget_only_selection_ignores_accuracy_and_stays_below_target() -> None:
    curves = {}
    for method in DYNAMIC_METHODS:
        curves[method] = [
            {
                "method": method,
                "parameter_type": (
                    "window_size" if method == "ESC" else "confidence_threshold"
                ),
                "window_size": 2 if method == "ESC" else None,
                "threshold": 2 if method == "ESC" else 0.10,
                "actual_avg_samples": 15.0,
                "accuracy": 0.99,
                "accuracy_percent": 99.0,
                "budget_cap": 32,
                "valid_samples": 30,
                "invalid_rate": 0.0,
            },
            {
                "method": method,
                "parameter_type": (
                    "window_size" if method == "ESC" else "confidence_threshold"
                ),
                "window_size": 3 if method == "ESC" else None,
                "threshold": 3 if method == "ESC" else 0.20,
                "actual_avg_samples": 15.9,
                "accuracy": 0.01,
                "accuracy_percent": 1.0,
                "budget_cap": 32,
                "valid_samples": 31,
                "invalid_rate": 0.0,
            },
            {
                "method": method,
                "parameter_type": (
                    "window_size" if method == "ESC" else "confidence_threshold"
                ),
                "window_size": 4 if method == "ESC" else None,
                "threshold": 4 if method == "ESC" else 0.30,
                "actual_avg_samples": 16.1,
                "accuracy": 1.0,
                "accuracy_percent": 100.0,
                "budget_cap": 32,
                "valid_samples": 32,
                "invalid_rate": 0.0,
            },
        ]

    matches = _select_budget_only_matches(curves, (16,))
    assert len(matches) == len(DYNAMIC_METHODS)
    assert all(row["actual_avg_samples"] == 15.9 for row in matches)
    assert all(row["accuracy_percent"] == 1.0 for row in matches)
    assert all(row["selection_uses_accuracy"] is False for row in matches)
    assert all(row["selection_rule"] == SELECTION_RULE for row in matches)
    assert all(row["budget_compliant"] is True for row in matches)


def _records(split: str) -> list[dict[str, object]]:
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


def _config() -> AggregateConfig:
    return AggregateConfig(
        budgets=(4, 16),
        thresholds=(0.0, 0.5, 0.8, 1.0),
        curve_max_budget=4,
        budget_targets=(3,),
        esc_window_sizes=(2, 3, 4),
        rasc_buffer_size=2,
    )


def test_paper_aggregation_is_reportable_and_uses_full_pool_curves(tmp_path) -> None:
    input_path = tmp_path / "test.jsonl"
    atomic_write_jsonl(input_path, _records("test"))
    report = run_paper_aggregation(
        [input_path],
        tmp_path / "results",
        config=_config(),
        model_id="toy-model",
        dataset_name="toy",
    )

    assert report["evaluation_phase"] == "paper"
    assert report["reportable"] is True
    assert report["paper_budget_protocol"]["selection_uses_accuracy"] is False
    assert report["paper_budget_protocol"]["target_is_average_budget"] is True
    assert report["paper_budget_protocol"]["per_question_cap"] == 4
    assert all(row["selection_split"] == "test" for row in report["dynamic_budget_matches"])
    assert all(row["actual_avg_samples"] <= 3 for row in report["dynamic_budget_matches"])
    assert (tmp_path / "results" / "evaluation.json").is_file()
    markdown = (tmp_path / "results" / "evaluation.md").read_text(encoding="utf-8")
    assert "budget-only" in markdown
    assert "accuracy is not used for selection" in markdown


def test_paper_aggregation_rejects_non_test_split(tmp_path) -> None:
    input_path = tmp_path / "validation.jsonl"
    atomic_write_jsonl(input_path, _records("validation"))
    with pytest.raises(ValueError, match="split='test'"):
        run_paper_aggregation(
            [input_path],
            tmp_path / "results",
            config=_config(),
            model_id="toy-model",
            dataset_name="toy",
        )
