"""CaTS paper-style budget-only aggregation on held-out test response pools.

This module deliberately leaves the stricter validation -> test contract in
``evaluation.aggregate`` untouched.  It implements the protocol described for
CaTS Table 2: for each dataset and dynamic method, scan the control-parameter
curve on the test response pool and select the parameter whose *average sample
cost* is closest to, but does not exceed, the requested target budget.
Ground-truth labels and accuracy are never part of the parameter-selection key.

The full response pool is still capped by ``curve_max_budget`` (32 by default),
so individual questions may consume more than the target average budget.  This
is intentional: the target is an average sample budget, matching the paper
wording, rather than a per-question hard cap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from relacats_v2.evaluation.aggregate import (
    AggregateConfig,
    DYNAMIC_METHODS,
    _control_parameter,
    _discover_confidence_files,
    _iter_files,
    _manifest_expected_questions,
    _parse_floats,
    _parse_ints,
    evaluate_records,
    write_reports,
)
from relacats_v2.evaluation.method_names import canonicalize_report_methods


PAPER_PROTOCOL = "cats-paper-budget-only"
SELECTION_RULE = (
    "CaTS paper-style budget-only selection on the test response pool: choose "
    "the closest actual average at or below the target; accuracy is not used"
)


def _select_budget_only_matches(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    budget_targets: Sequence[int],
) -> list[dict[str, Any]]:
    """Select dynamic controls using sample cost only, never accuracy."""

    matches: list[dict[str, Any]] = []
    for target in budget_targets:
        for method in DYNAMIC_METHODS:
            rows = list(curves.get(method, ()))
            if not rows:
                raise ValueError(f"No threshold/window curve rows for {method}")
            feasible = [
                row
                for row in rows
                if float(row["actual_avg_samples"]) <= float(target) + 1e-12
            ]
            if not feasible:
                minimum = min(float(row["actual_avg_samples"]) for row in rows)
                raise ValueError(
                    f"No paper-protocol parameter for {method} satisfies average "
                    f"budget <= {target}; minimum observed average is {minimum:.6f}"
                )

            # Accuracy is intentionally absent from this key.  First maximize
            # average sample use while staying at/below target; then use the
            # numeric control parameter only as a deterministic tie breaker.
            selected = min(
                feasible,
                key=lambda row: (
                    float(target) - float(row["actual_avg_samples"]),
                    float(_control_parameter(row)),
                ),
            )
            matches.append(
                {
                    **dict(selected),
                    "row_type": "dynamic_budget_match",
                    "budget": int(target),
                    "budget_target": int(target),
                    "budget_gap": float(selected["actual_avg_samples"]) - float(target),
                    "selection_split": "test",
                    "selection_rule": SELECTION_RULE,
                    "selection_uses_accuracy": False,
                    "budget_compliant": (
                        float(selected["actual_avg_samples"]) <= float(target) + 1e-12
                    ),
                }
            )
    return matches


def _paper_markdown(report: Mapping[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    lines = [
        "# RelaCaTS-v2 CaTS-paper budget evaluation",
        "",
        "Protocol: **CaTS paper-style per-dataset budget-only matching**.",
        "The control parameter is selected using average sample cost only; test accuracy is not used for selection.",
        f"Target average budget(s): **{', '.join(map(str, report['config']['budget_targets']))}**.",
        f"Maximum available response pool per question: **{report['config']['curve_max_budget']}**.",
        "",
        "## Coverage",
        "",
        f"- Questions in denominator: {diagnostics.get('questions_total_denominator', 0)}",
        f"- Questions observed: {diagnostics.get('questions_observed', 0)}",
        f"- Unique samples: {diagnostics.get('unique_samples', 0)}",
        f"- Invalid extracted answers: {diagnostics.get('invalid_extracted_answers', 0)}",
        "",
        "## Fixed-budget results",
        "",
        "| Method | Budget | Accuracy | Actual avg | Invalid rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["fixed_budget_results"]:
        lines.append(
            f"| {row['method']} | {row['budget']} | {row['accuracy_percent']:.2f}% | "
            f"{row['actual_avg_samples']:.3f} | {100.0 * row['invalid_rate']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Dynamic methods at requested average budgets",
            "",
            "| Method | Target | Threshold/window | Accuracy | Actual avg | Gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["dynamic_budget_matches"]:
        parameter = (
            row.get("window_size")
            if row.get("parameter_type") == "window_size"
            else row.get("threshold")
        )
        lines.append(
            f"| {row['method']} | {row['budget_target']} | {float(parameter):.4f} | "
            f"{row['accuracy_percent']:.2f}% | {row['actual_avg_samples']:.3f} | "
            f"{row['budget_gap']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Selection rule: choose the parameter with average samples closest to but not above the target. Accuracy/gold labels are excluded from the selection key.",
            "",
        ]
    )
    return "\n".join(lines)


def run_paper_aggregation(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    config: AggregateConfig | None = None,
    expected_question_count: int | None = None,
    *,
    model_id: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Run the paper-style budget protocol on records explicitly marked test."""

    files = _discover_confidence_files(inputs)
    if expected_question_count is None:
        expected_question_count = _manifest_expected_questions(files)
    config = config or AggregateConfig()

    # ``analysis`` exposes the full 0..1 / ESC-window curves with the full
    # candidate-pool cap.  We then perform the paper's budget-only selection
    # ourselves instead of using analysis' diagnostic target-capped matches.
    report = evaluate_records(
        _iter_files(files),
        config=config,
        expected_question_count=expected_question_count,
        phase="analysis",
        model_id=model_id,
        dataset_name=dataset_name,
    )

    observed_splits = set(report["diagnostics"].get("input_splits", ()))
    if observed_splits != {"test"}:
        raise ValueError(
            "Paper budget-only evaluation requires records explicitly labelled "
            f"split='test'; observed splits={sorted(observed_splits)}"
        )
    observed_datasets = set(report["diagnostics"].get("input_datasets", ()))
    if observed_datasets != {dataset_name}:
        raise ValueError(
            f"Input dataset mismatch: records={sorted(observed_datasets)}, "
            f"requested={dataset_name!r}"
        )

    report["dynamic_budget_matches"] = _select_budget_only_matches(
        report["threshold_curves"], config.budget_targets
    )
    report["evaluation_phase"] = "paper"
    report["reportable"] = True
    report["protocol"] = (
        "CaTS paper-style per-dataset budget-only matching on the test response "
        "pool; parameter selection uses average sample cost only and never accuracy"
    )
    report["paper_budget_protocol"] = {
        "name": PAPER_PROTOCOL,
        "selection_pool_split": "test",
        "selection_metric": "actual_avg_samples",
        "selection_uses_accuracy": False,
        "target_policy": "closest actual average at or below target",
        "per_question_cap": int(config.curve_max_budget),
        "target_is_average_budget": True,
    }
    report["input_files"] = [str(path) for path in files]
    report = canonicalize_report_methods(report)

    output = Path(output_dir).expanduser().resolve()
    report["output_files"] = {
        "json": str(output / "evaluation.json"),
        "csv": str(output / "evaluation.csv"),
        "markdown": str(output / "evaluation.md"),
    }
    write_reports(report, output)
    (output / "evaluation.md").write_text(_paper_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CaTS paper-style Table-2 aggregation: select dynamic controls by "
            "test-pool sample cost only, targeting an average budget"
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Confidence artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--budgets", type=_parse_ints, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--thresholds",
        type=_parse_floats,
        default=tuple(index / 100 for index in range(101)),
    )
    parser.add_argument("--curve-max-budget", type=int, default=32)
    parser.add_argument("--budget-targets", type=_parse_ints, default=(16,))
    parser.add_argument("--dynamic-min-valid", type=int, default=2)
    parser.add_argument("--rasc-buffer-size", type=int, default=5)
    parser.add_argument("--esc-window-sizes", type=_parse_ints, default=())
    parser.add_argument("--cisc-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cisc-normalization",
        choices=("softmax", "linear", "none"),
        default="softmax",
    )
    parser.add_argument("--expected-questions", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AggregateConfig(
        budgets=tuple(args.budgets),
        thresholds=tuple(args.thresholds),
        curve_max_budget=args.curve_max_budget,
        budget_targets=tuple(args.budget_targets),
        dynamic_min_valid=args.dynamic_min_valid,
        rasc_buffer_size=args.rasc_buffer_size,
        esc_window_sizes=tuple(args.esc_window_sizes),
        cisc_temperature=args.cisc_temperature,
        cisc_normalization=args.cisc_normalization,
    )
    report = run_paper_aggregation(
        args.input,
        args.output_dir,
        config=config,
        expected_question_count=args.expected_questions,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(report["output_files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
