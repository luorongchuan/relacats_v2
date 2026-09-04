"""CPU-only v2 reaggregation of the three existing RelaCaTS response pools.

The copied artifacts contain only records labelled ``split=test``.  To avoid
choosing dynamic thresholds on the same questions used for reporting, this
audit creates a deterministic, question-disjoint calibration holdout and
held-out test partition.  The resulting report records that provenance and is
not presented as an official-dataset validation split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from relacats_v2.common import atomic_write_json, read_json, read_jsonl
from relacats_v2.evaluation.aggregate import (
    AggregateConfig,
    DYNAMIC_METHODS,
    FIXED_METHODS,
    _discover_confidence_files,
    build_threshold_calibration,
    evaluate_records,
    write_reports,
)
from relacats_v2.evaluation.method_names import (
    TABLE2_METHOD_ORDER,
    canonical_method_name,
)


DEFAULT_MODELS = (
    "qwen2_5_7b_instruct",
    "llama3_1_8b_instruct",
    "deepseek_r1_distill_qwen_1_5b",
)
DEFAULT_DATASETS = ("object_counting", "math_qa", "arc_challenge")


def _is_validation_question(
    question_id: str, dataset: str, *, seed: int, fraction: float
) -> bool:
    digest = hashlib.sha256(
        f"relacats-v2-holdout\0{seed}\0{dataset}\0{question_id}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return unit < fraction


def _partition_records(
    files: Sequence[Path],
    dataset: str,
    role: str,
    *,
    seed: int,
    fraction: float,
) -> Iterator[dict[str, Any]]:
    if role not in {"validation", "test"}:
        raise ValueError(role)
    for path in files:
        for raw in read_jsonl(path):
            question_id = str(raw.get("question_id", ""))
            if not question_id:
                raise ValueError(f"Missing question_id in {path}")
            selected = _is_validation_question(
                question_id, dataset, seed=seed, fraction=fraction
            )
            if selected != (role == "validation"):
                continue
            record = dict(raw)
            record["source_split"] = record.get("split")
            record["split"] = role
            yield record


def _selected_rows(report: Mapping[str, Any], budget: int = 16) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in report.get("fixed_budget_results", ()):
        if int(raw.get("budget", -1)) == budget:
            rows.append(dict(raw))
    for raw in report.get("dynamic_budget_matches", ()):
        if int(raw.get("budget_target", raw.get("budget", -1))) == budget:
            rows.append(dict(raw))
    by_method = {canonical_method_name(row["method"]): row for row in rows}
    missing = set(TABLE2_METHOD_ORDER) - set(by_method)
    if missing:
        raise ValueError(f"Missing budget-{budget} methods: {sorted(missing)}")
    return [by_method[name] for name in TABLE2_METHOD_ORDER]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _old_selected_rows(path: Path, budget: int = 16) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    report = read_json(path)
    selected: dict[str, dict[str, Any]] = {}
    for raw in report.get("fixed_budget_results", ()):
        if int(raw.get("budget", -1)) == budget:
            selected[canonical_method_name(raw["method"])] = dict(raw)
    for raw in report.get("dynamic_budget_matches", ()):
        if int(raw.get("budget_target", raw.get("budget", -1))) == budget:
            selected[canonical_method_name(raw["method"])] = dict(raw)
    return selected


def _metric(row: Mapping[str, Any], name: str) -> float | None:
    aliases = {
        "actual_avg_samples": ("actual_avg_samples", "avg_samples_used"),
        "invalid_rate": ("invalid_rate",),
        "accuracy": ("accuracy",),
    }
    for key in aliases[name]:
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _aggregate_model_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    states: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        key = (str(row["model_id"]), str(row["method"]))
        questions = int(row["questions_total"])
        state = states[key]
        state["questions"] += questions
        state["correct"] += int(row["correct"])
        state["generated_samples"] += int(row["generated_samples"])
        state["valid_samples"] += int(row["valid_samples"])
        state["invalid_samples"] += int(row["invalid_samples"])
    result: list[dict[str, Any]] = []
    for model in DEFAULT_MODELS:
        for method in TABLE2_METHOD_ORDER:
            state = states.get((model, method))
            if not state:
                continue
            questions = int(state["questions"])
            generated = int(state["generated_samples"])
            result.append(
                {
                    "model_id": model,
                    "method": method,
                    "questions_total": questions,
                    "accuracy": state["correct"] / questions,
                    "actual_avg_samples": generated / questions,
                    "generated_samples": generated,
                    "valid_samples": int(state["valid_samples"]),
                    "invalid_rate": (
                        state["invalid_samples"] / generated if generated else 0.0
                    ),
                }
            )
    return result


def reaggregate(args: argparse.Namespace) -> Path:
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite v2 aggregation output: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building.", dir=output_root.parent)
    )
    config = AggregateConfig(budget_targets=(args.target_budget,))
    summary_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    raw_model_stats: dict[str, dict[str, Any]] = {}
    try:
        for model_id in args.models:
            raw_input = raw_invalid = raw_questions = 0
            for dataset in args.datasets:
                print(f"[v2 CPU] {model_id}/{dataset}: validating input", flush=True)
                confidence_root = source_root / model_id / "confidence" / dataset
                files = _discover_confidence_files([confidence_root])
                dataset_root = staging / model_id / dataset
                # Reports are written into the staging tree and atomically
                # renamed at the end.  Keep the threshold path recorded in
                # those reports stable by pointing metadata at the eventual
                # final output location rather than the ephemeral staging
                # directory.
                reported_threshold_path = (
                    output_root / model_id / dataset / "dynamic_thresholds.json"
                )
                provenance = {
                    "source_artifacts": [str(path) for path in files],
                    "source_split": "test",
                    "partition_strategy": "sha256 question-id holdout",
                    "validation_fraction": args.validation_fraction,
                    "partition_seed": args.seed,
                    "warning": (
                        "Audit holdout derived from an existing test response pool; "
                        "not an official independent validation split."
                    ),
                }

                print(f"[v2 CPU] {model_id}/{dataset}: validation calibration", flush=True)
                validation = evaluate_records(
                    _partition_records(
                        files,
                        dataset,
                        "validation",
                        seed=args.seed,
                        fraction=args.validation_fraction,
                    ),
                    config=config,
                    phase="validation",
                    model_id=model_id,
                    dataset_name=dataset,
                )
                validation["partition_provenance"] = provenance
                threshold_doc = build_threshold_calibration(
                    validation, model_id=model_id, dataset_name=dataset
                )
                threshold_doc["partition_provenance"] = provenance
                threshold_path = dataset_root / "dynamic_thresholds.json"
                atomic_write_json(threshold_path, threshold_doc)
                validation["threshold_calibration_file"] = str(
                    reported_threshold_path
                )
                write_reports(validation, dataset_root / "validation")

                # The test stage deliberately reloads the immutable artifact
                # from disk; it never receives validation curves directly.
                persisted_thresholds = read_json(threshold_path)
                print(f"[v2 CPU] {model_id}/{dataset}: held-out test", flush=True)
                test_report = evaluate_records(
                    _partition_records(
                        files,
                        dataset,
                        "test",
                        seed=args.seed,
                        fraction=args.validation_fraction,
                    ),
                    config=config,
                    phase="test",
                    threshold_calibration=persisted_thresholds,
                    model_id=model_id,
                    dataset_name=dataset,
                )
                test_report["partition_provenance"] = provenance
                test_report["threshold_calibration_file"] = str(
                    reported_threshold_path
                )
                write_reports(test_report, dataset_root / "test")

                validation_diag = validation["diagnostics"]
                test_diag = test_report["diagnostics"]
                raw_input += int(validation_diag["unique_samples"]) + int(
                    test_diag["unique_samples"]
                )
                raw_invalid += int(validation_diag["invalid_extracted_answers"]) + int(
                    test_diag["invalid_extracted_answers"]
                )
                raw_questions += int(validation_diag["questions_observed"]) + int(
                    test_diag["questions_observed"]
                )

                new_rows = _selected_rows(test_report, args.target_budget)
                old_path = source_root / model_id / "results" / dataset / "evaluation.json"
                old_rows = _old_selected_rows(old_path, args.target_budget)
                for row in new_rows:
                    summary = {
                        "model_id": model_id,
                        "dataset_name": dataset,
                        "partition": "held_out_test",
                        **row,
                    }
                    summary_rows.append(summary)
                    method = str(row["method"])
                    old = old_rows.get(method)
                    old_accuracy = _metric(old, "accuracy") if old else None
                    old_avg = _metric(old, "actual_avg_samples") if old else None
                    comparisons.append(
                        {
                            "model_id": model_id,
                            "dataset_name": dataset,
                            "method": method,
                            "old_scope": "full legacy test aggregation",
                            "new_scope": "80% deterministic held-out audit test",
                            "directly_comparable": False,
                            "old_accuracy": old_accuracy,
                            "new_accuracy": float(row["accuracy"]),
                            "accuracy_delta": (
                                float(row["accuracy"]) - old_accuracy
                                if old_accuracy is not None
                                else None
                            ),
                            "old_actual_avg_samples": old_avg,
                            "new_actual_avg_samples": float(row["actual_avg_samples"]),
                            "avg_samples_delta": (
                                float(row["actual_avg_samples"]) - old_avg
                                if old_avg is not None
                                else None
                            ),
                        }
                    )

            raw_model_stats[model_id] = {
                "questions": raw_questions,
                "samples": raw_input,
                "invalid_samples": raw_invalid,
                "invalid_rate": raw_invalid / raw_input if raw_input else 0.0,
            }

        model_method_rows = _aggregate_model_rows(summary_rows)
        atomic_write_json(staging / "summary.json", summary_rows)
        _write_csv(staging / "summary.csv", summary_rows)
        atomic_write_json(staging / "model_method_summary.json", model_method_rows)
        _write_csv(staging / "model_method_summary.csv", model_method_rows)
        atomic_write_json(staging / "old_new_comparison.json", comparisons)
        _write_csv(staging / "old_new_comparison.csv", comparisons)
        manifest = {
            "schema_version": "relacats-v2.cpu-reaggregation.1",
            "complete": True,
            "models": list(args.models),
            "datasets": list(args.datasets),
            "target_budget": args.target_budget,
            "validation_fraction": args.validation_fraction,
            "partition_seed": args.seed,
            "validation_only_threshold_selection": True,
            "test_reads_persisted_thresholds": True,
            "official_validation_split": False,
            "source_root": str(source_root),
            "raw_model_stats": raw_model_stats,
            "summary_rows": len(summary_rows),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_root)
    except BaseException:
        print(f"Incomplete staging directory retained for diagnosis: {staging}")
        raise
    print(f"Complete v2 CPU reaggregation: {output_root}")
    return output_root


def build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", default=str(package_root / "outputs/eval_outputs")
    )
    parser.add_argument(
        "--output-root", default=str(package_root / "outputs/eval_outputs_v2")
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-budget", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if args.target_budget <= 0:
        raise ValueError("target_budget must be positive")
    reaggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
