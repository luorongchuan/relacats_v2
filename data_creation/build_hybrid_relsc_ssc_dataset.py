"""Build the selective hybrid confidence-target dataset for RelaCaTS v2.

Repository split:
- relacats_v1 generates raw model responses and per-response P(True).
- relacats_v2 computes pseudo-labels, selects training data, and trains models.

Target protocol:
- seven MCQ datasets: count-based RelSC over all valid canonicalized relation views;
- GSM8K/SVAMP: confidence-weighted SSC from the separate identity-only N=32 pool.

Targets are computed *before* training-set subsampling.  The trainer later performs
CaTS-style confidence-bin balancing and applies the 100k/1k sample budgets.
Gold answers are never used to construct pseudo-labels.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_json
from relacats_v2.core import compute_relssc, compute_ssc_target_context
from relacats_v2.data_creation.build_relssc_dataset import (
    _normalise_answer_type,
    discover_question_files,
    split_question_ids,
    validate_question_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MCQ_INPUT = REPO_ROOT / "relacats_v1/outputs/generated_data"
DEFAULT_NUMERIC_INPUT = REPO_ROOT / "relacats_v1/outputs/generated_data_identity_only"
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v2/outputs/hybrid_relsc_ssc_dataset"

TRAIN_MCQ_DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "winogrande",
)
TRAIN_NUMERIC_DATASETS = ("gsm8k", "svamp")
TRAIN_DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "gsm8k",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "svamp",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq-input-root", default=str(DEFAULT_MCQ_INPUT))
    parser.add_argument("--numeric-input-root", default=str(DEFAULT_NUMERIC_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--datasets", nargs="+", choices=TRAIN_DATASETS, default=list(TRAIN_DATASETS)
    )
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-nonstandard-budget", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _valid_samples(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("is_valid_answer") is True
        and sample.get("canonicalized_answer") is not None
    ]


def pure_relsc_scores(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Pure count-based RelSC after inverse canonicalization, valid samples only."""
    valid = _valid_samples(samples)
    if not valid:
        return {}
    counts = Counter(str(sample["canonicalized_answer"]) for sample in valid)
    denominator = float(sum(counts.values()))
    return {answer: count / denominator for answer, count in counts.items()}


def numeric_ssc_scores(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """CaTS SSC on an identity-only numeric pool using stored P(True) weights."""
    result = compute_relssc(samples, zero_weight_policy="skip")
    return dict(result.scores) if result.defined else {}


def hybrid_scores(
    samples: Sequence[Mapping[str, Any]], *, answer_type: str
) -> tuple[str, dict[str, float]]:
    normalized = _normalise_answer_type(answer_type)
    if normalized == "option":
        return "relsc", pure_relsc_scores(samples)
    if normalized == "number":
        return "ssc", numeric_ssc_scores(samples)
    raise AssertionError(normalized)


def flatten_question(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = payload["samples"]
    answer_type = _normalise_answer_type(payload.get("answer_type", "option"))
    method, scores = hybrid_scores(samples, answer_type=answer_type)
    valid = _valid_samples(samples)
    identity_context = compute_ssc_target_context(samples)

    if not scores:
        return [], {
            "question_id": payload["question_id"],
            "dataset_name": payload.get("dataset_name"),
            "answer_type": answer_type,
            "target_method": method,
            "defined": False,
            "reason": "no valid target support",
            "valid_response_count": len(valid),
            "invalid_response_count": len(samples) - len(valid),
            "gold_used_in_target": False,
        }

    rows: list[dict[str, Any]] = []
    for sample in valid:
        canonical = str(sample["canonicalized_answer"])
        if canonical not in scores:
            continue
        target = float(scores[canonical])
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(f"{payload['question_id']}: invalid target {target!r}")

        original_identity_sc = identity_context.score(canonical)
        row = dict(sample)
        prompt = sample.get(
            "transformed_prompt",
            sample.get("original_prompt", sample.get("transformed_question", "")),
        )
        row.update(
            {
                "input": f"{prompt}{sample.get('response', '')}",
                "answer": sample["canonicalized_answer"],
                "hybrid_consistency": target,
                "target_method": method,
                "original_identity_sc": original_identity_sc,
                "question_hybrid_scores": dict(scores),
                "question_valid_response_count": len(valid),
                "question_invalid_response_count": len(samples) - len(valid),
                "relation_valid_ratio": float(identity_context.relation_valid_ratio),
                "attempted_budget": int(payload["attempted_budget"]),
                "target_provenance": (
                    "relsc_valid_count_after_inverse_mapping_without_gold"
                    if method == "relsc"
                    else "ssc_identity_confidence_weighted_without_gold"
                ),
            }
        )
        if method == "relsc":
            row["relsc"] = target
        else:
            row["ssc_weighted"] = target
        rows.append(row)

    top_answer = max(scores, key=scores.get)
    return rows, {
        "question_id": payload["question_id"],
        "dataset_name": payload.get("dataset_name"),
        "answer_type": answer_type,
        "relation_mode": payload.get("relation_mode"),
        "target_method": method,
        "defined": True,
        "top_answer": top_answer,
        "scores": dict(scores),
        "valid_response_count": len(valid),
        "invalid_response_count": len(samples) - len(valid),
        "valid_identity_sample_count": identity_context.valid_identity_samples,
        "invalid_identity_sample_count": identity_context.invalid_identity_samples,
        "relation_valid_ratio": identity_context.relation_valid_ratio,
        "gold_used_in_target": False,
    }


def build_dataset(
    *,
    dataset_name: str,
    files: Sequence[Path],
    output_root: Path,
    test_ratio: float,
    seed: int,
    allow_nonstandard_budget: bool,
) -> dict[str, Any]:
    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    question_summaries: list[dict[str, Any]] = []
    raw_count = 0
    seen_ids: set[str] = set()
    observed_methods: set[str] = set()

    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}")
        if payload.get("dataset_name") != dataset_name:
            raise ValueError(f"Dataset directory/payload mismatch at {path}")
        question_id = str(payload["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id {question_id!r}")
        seen_ids.add(question_id)

        samples = validate_question_payload(
            payload, allow_nonstandard_budget=allow_nonstandard_budget
        )
        raw_count += len(samples)
        rows, summary = flatten_question(payload)
        question_summaries.append(summary)
        observed_methods.add(str(summary["target_method"]))
        if rows:
            rows_by_question[question_id] = rows

    expected_method = "ssc" if dataset_name in TRAIN_NUMERIC_DATASETS else "relsc"
    if observed_methods and observed_methods != {expected_method}:
        raise ValueError(
            f"{dataset_name}: expected {expected_method}, got {sorted(observed_methods)}"
        )

    train_ids, test_ids = split_question_ids(
        list(rows_by_question), test_ratio=test_ratio, seed=seed
    )
    dataset_dir = output_root / dataset_name
    train_rows = [
        row for question_id in sorted(train_ids) for row in rows_by_question[question_id]
    ]
    test_rows = [
        row for question_id in sorted(test_ids) for row in rows_by_question[question_id]
    ]
    atomic_write_jsonl(dataset_dir / "train.jsonl", train_rows)
    atomic_write_jsonl(dataset_dir / "test.jsonl", test_rows)
    atomic_write_json(
        dataset_dir / "question_summaries.json",
        sorted(question_summaries, key=lambda item: item["question_id"]),
    )

    targets = [float(row["hybrid_consistency"]) for row in train_rows + test_rows]
    stats = {
        "schema_version": "relacats-v2.hybrid-dataset-stats.1",
        "dataset_name": dataset_name,
        "target_method": expected_method,
        "source_question_files": len(files),
        "defined_questions": len(rows_by_question),
        "skipped_undefined_questions": len(files) - len(rows_by_question),
        "train_questions": len(train_ids),
        "test_questions": len(test_ids),
        "raw_response_records": raw_count,
        "valid_training_records": len(train_rows) + len(test_rows),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "mean_hybrid_target": sum(targets) / len(targets) if targets else None,
        "high_hybrid_records_gt_0_75": sum(value > 0.75 for value in targets),
        "group_split_no_question_leakage": not bool(train_ids & test_ids),
        "gold_used_in_target": False,
    }
    atomic_write_json(dataset_dir / "stats.json", stats)
    return stats


def main() -> None:
    args = parse_args()
    mcq_root = resolve_path(args.mcq_input_root)
    numeric_root = resolve_path(args.numeric_input_root)
    output_root = resolve_path(args.output_root)
    selected = tuple(args.datasets)

    mcq_selected = [name for name in selected if name in TRAIN_MCQ_DATASETS]
    numeric_selected = [name for name in selected if name in TRAIN_NUMERIC_DATASETS]

    files_by_dataset: dict[str, list[Path]] = {}
    if mcq_selected:
        files_by_dataset.update(discover_question_files(mcq_root, mcq_selected))
    if numeric_selected:
        files_by_dataset.update(discover_question_files(numeric_root, numeric_selected))

    summaries: list[dict[str, Any]] = []
    for dataset_name in selected:
        files = files_by_dataset.get(dataset_name)
        if not files:
            raise FileNotFoundError(f"No source files resolved for {dataset_name}")
        stats = build_dataset(
            dataset_name=dataset_name,
            files=files,
            output_root=output_root,
            test_ratio=args.test_ratio,
            seed=args.seed,
            allow_nonstandard_budget=args.allow_nonstandard_budget,
        )
        summaries.append(stats)
        print(
            f"{dataset_name}: method={stats['target_method']} "
            f"questions={stats['defined_questions']} "
            f"train/test rows={stats['train_records']}/{stats['test_records']}"
        )

    manifest = {
        "schema_version": "relacats-v2.hybrid-dataset-manifest.1",
        "mcq_input_root": str(mcq_root),
        "numeric_input_root": str(numeric_root),
        "output_root": str(output_root),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "formal_budget_enforced": not args.allow_nonstandard_budget,
        "invalid_policy": "valid canonical answers only in target denominators/training pool",
        "split_unit": "original_question_id",
        "target_field": "hybrid_consistency",
        "target_protocol": {
            "mcq": "RelSC valid canonical frequency after inverse option mapping",
            "numeric": "CaTS SSC confidence-weighted identity-only frequency",
        },
        "selection_order": "compute targets first; CaTS-style bin-balanced training subsampling second",
        "gold_used_in_target": False,
        "datasets": summaries,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(f"Wrote hybrid manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
