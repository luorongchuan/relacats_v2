"""Pure-CPU smoke test for every RelaCaTS-v2 evaluation aggregator.

The synthetic run intentionally exercises the complete Table-2 method set.
The three confidence-aware methods proposed by this project use the
``RelaCaTS-*`` display names; the remaining names are baseline methods from
the original paper.  Keeping these assertions here prevents a future
evaluator change from silently dropping a baseline or writing the old
``CaTS-*`` labels into result rows.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Any, Sequence

from relacats_v2.common import atomic_write_jsonl
from relacats_v2.evaluation.aggregate import AggregateConfig, run_aggregation


# Keep this list explicit rather than deriving it from the implementation's
# constants.  The purpose of the smoke test is to catch an accidentally
# omitted method (or a regression to the old CaTS display names).
EXPECTED_FIXED_METHODS = {
    "SC",
    "CISC",
    "Self-Certainty",
    "Best-of-N",
    "RelaCaTS-SC",
}
EXPECTED_DYNAMIC_METHODS = {
    "RelaCaTS-ES",
    "ASC",
    "RelaCaTS-ASC",
    "ESC",
    "RASC",
}
LEGACY_RELACATS_NAMES = {"CaTS-SC", "CaTS-ES", "CaTS-ASC"}


def _record(
    question: int,
    index: int,
    gold: str,
    answer: str | None,
    confidence: float | None,
    yes_found: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "relacats_v2_confidence_record",
        "sample_id": f"q{question}-r{index}",
        "question_id": f"q{question}",
        "generation_index": index,
        "dataset_name": "synthetic",
        "correct_answer": gold,
        "extracted_answer": answer,
        "confidence": confidence,
        "yes_token_found_top20": yes_found,
    }


def synthetic_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (answer, confidence) in enumerate(
        (("A", 0.90), ("A", 0.80), ("B", 0.95), ("A", 0.70))
    ):
        records.append(_record(1, index, "A", answer, confidence))
    for index, (answer, confidence) in enumerate(
        (("A", 0.20), ("B", 0.75), ("B", 0.85), ("B", None))
    ):
        records.append(
            _record(2, index, "B", answer, confidence, yes_found=confidence is not None)
        )
    # This question deliberately has four malformed answers.  It must remain in
    # the denominator, as must the fourth entirely missing question requested
    # below.
    for index in range(4):
        records.append(_record(3, index, "C", None, 0.99))
    return records


def run_smoke(output_dir: str | Path | None = None) -> dict[str, Any]:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="relacats-eval-smoke-")
        root = Path(temporary.name)
    else:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    input_path = root / "synthetic_confidence.jsonl"
    result_dir = root / "results"
    atomic_write_jsonl(input_path, synthetic_records())
    config = AggregateConfig(
        budgets=(2, 4, 16),
        thresholds=(0.0, 0.5, 0.8, 1.0),
        curve_max_budget=4,
        budget_targets=(2,),
    )
    report = run_aggregation(
        [input_path],
        result_dir,
        config=config,
        expected_question_count=4,
    )
    diagnostics = report["diagnostics"]
    assert diagnostics["questions_total_denominator"] == 4
    assert diagnostics["questions_observed"] == 3
    assert diagnostics["questions_missing_entirely"] == 1
    assert diagnostics["questions_without_valid_answer"] == 1
    assert diagnostics["invalid_extracted_answers"] == 4
    assert diagnostics["missing_or_nonfinite_confidence"] == 1
    assert set(report["threshold_curves"]) == EXPECTED_DYNAMIC_METHODS
    fixed_methods = {row["method"] for row in report["fixed_budget_results"]}
    assert fixed_methods == EXPECTED_FIXED_METHODS

    # Every method must preserve the strict denominator, including the
    # entirely missing synthetic question.  This catches implementations that
    # accidentally filter malformed/missing rows for only one new method.
    for row in report["fixed_budget_results"]:
        assert row["questions_total"] == 4
        assert row["method"] not in LEGACY_RELACATS_NAMES
        assert math.isfinite(float(row["accuracy_percent"]))
    for method, rows in report["threshold_curves"].items():
        assert rows, method
        assert method not in LEGACY_RELACATS_NAMES
        for row in rows:
            assert row["questions_total"] == 4
            assert math.isfinite(float(row["accuracy_percent"]))

    # The persisted machine-readable and Markdown reports should expose the
    # same canonical names as the in-memory report.  Original-CaTS protocol
    # prose may still mention CaTS, so only inspect method-labelled rows.
    all_method_names = {
        row["method"]
        for row in report["fixed_budget_results"]
    } | {
        row["method"]
        for rows in report["threshold_curves"].values()
        for row in rows
    } | {row["method"] for row in report["dynamic_budget_matches"]}
    assert {
        row["method"] for row in report["dynamic_budget_matches"]
    } == EXPECTED_DYNAMIC_METHODS
    for row in report["dynamic_budget_matches"]:
        assert row["questions_total"] == 4
        assert math.isfinite(float(row["accuracy_percent"]))
    assert not (all_method_names & LEGACY_RELACATS_NAMES)
    assert EXPECTED_FIXED_METHODS | EXPECTED_DYNAMIC_METHODS <= all_method_names
    markdown = Path(report["output_files"]["markdown"]).read_text(encoding="utf-8")
    for name in ("RelaCaTS-SC", "RelaCaTS-ES", "RelaCaTS-ASC"):
        assert name in markdown
    csv_text = Path(report["output_files"]["csv"]).read_text(encoding="utf-8")
    for name in EXPECTED_FIXED_METHODS | EXPECTED_DYNAMIC_METHODS:
        assert name in csv_text
    sc4 = next(
        row
        for row in report["fixed_budget_results"]
        if row["method"] == "SC" and row["budget"] == 4
    )
    assert sc4["questions_total"] == 4
    assert sc4["correct"] == 2
    assert sc4["accuracy"] == 0.5
    for path in report["output_files"].values():
        assert Path(path).is_file()
    print(f"Synthetic evaluation smoke passed: {result_dir}")
    # Keep a user-requested output directory; automatically clean an implicit
    # temporary directory only after all assertions and writes have succeeded.
    if temporary is not None:
        temporary.cleanup()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    run_smoke(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
