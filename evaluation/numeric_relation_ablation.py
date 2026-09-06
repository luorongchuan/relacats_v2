"""CPU-only diagnosis for numeric RelaCaTS relations on GSM8K/SVAMP.

This script consumes already-generated question JSONs.  It does not generate
new model responses and therefore does not require a GPU.

It is intended for the situation where matched 32I SSC baselines and the
numeric-metamorphic RelaCaTS pool already exist.  The script answers three
questions:

1. Which relation subtype helps or hurts calibration?
2. Does a relation preserve correct identity consensus and repair/break wrong
   identity consensus?
3. Is the lexical dependency correction aggressively collapsing samples?

The primary Table-1 comparison should still come from ``reproduce_table1``.
The subset rows here are diagnostics: they use fewer than 32 stored responses
when only a subset of relation views is selected, so they should not be cited
as equal-budget final results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from relacats_v2.core import (
    annotate_dependency_weights,
    compute_relssc,
    compute_relssc_full,
)
from relacats_v2.evaluation.reproduce_table1 import (
    MetricRow,
    Observation,
    _correct,
    _index_payloads,
    _load_payloads,
    _metric_row,
    _normalize_answer,
    _ssc_observation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELATIONAL_ROOT = (
    REPO_ROOT.parent
    / "relacats_v1/outputs/generated_data/llama3_1_8b_instruct"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT.parent
    / "relacats_v1/outputs/generated_data_identity_only/llama3_1_8b_instruct"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "relacats_v2/outputs/numeric_relation_ablation"

RELATION_SUBTYPES = (
    "identity",
    "layout_wrapper",
    "number_representation",
    "equivalent_quantity",
)

# Diagnostic profiles.  These are deliberately transparent rather than tuned.
PROFILE_SUBTYPES: dict[str, tuple[str, ...]] = {
    "Identity-only (stored g0)": ("identity",),
    "Layout-only": ("layout_wrapper",),
    "Number-representation-only": ("number_representation",),
    "Equivalent-quantity-only": ("equivalent_quantity",),
    "Identity+Layout": ("identity", "layout_wrapper"),
    "Identity+Number": ("identity", "number_representation"),
    "Identity+Equivalent": ("identity", "equivalent_quantity"),
    "Drop-Layout": ("identity", "number_representation", "equivalent_quantity"),
    "Drop-Number": ("identity", "layout_wrapper", "equivalent_quantity"),
    "Drop-Equivalent": ("identity", "layout_wrapper", "number_representation"),
    "All-Relations": RELATION_SUBTYPES,
}


@dataclass(frozen=True)
class RelationDiagnostic:
    dataset: str
    relation_subtype: str
    available_questions: int
    identity_correct_questions: int
    identity_wrong_questions: int
    relation_accuracy: float
    correct_preservation_rate: float
    harm_rate_given_identity_correct: float
    error_repair_rate: float
    wrong_consensus_break_rate: float
    wrong_same_answer_rate: float
    winner_agreement_rate: float
    attempted_samples: int
    valid_samples: int
    valid_response_rate: float


@dataclass(frozen=True)
class DependencyDiagnostic:
    dataset: str
    questions: int
    valid_samples: int
    clusters: int
    mean_cluster_size: float
    max_cluster_size: int
    multi_sample_cluster_fraction: float
    sample_fraction_in_multi_sample_clusters: float
    effective_cluster_mass_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_RELATIONAL_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--datasets", nargs="+", default=("gsm8k", "svamp"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--strategy-similarity-threshold", type=float, default=0.86)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _relation_subtype(sample: Mapping[str, Any]) -> str:
    subtype = str(sample.get("relation_subtype", "")).strip().lower()
    if subtype:
        return subtype
    relation_type = str(sample.get("relation_type", "")).strip().lower()
    relation_id = str(sample.get("relation_id", "g0")).strip().lower()
    if relation_type == "identity" or relation_id == "g0":
        return "identity"
    return relation_type or relation_id


def _samples_for_subtypes(
    payload: Mapping[str, Any], subtypes: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(subtypes)
    return [
        dict(sample)
        for sample in payload.get("samples", [])
        if isinstance(sample, Mapping) and _relation_subtype(sample) in allowed
    ]


def _observation_from_samples(
    dataset: str,
    payload: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    method: str,
) -> Observation | None:
    if not samples:
        return None
    result = compute_relssc(
        samples,
        zero_weight_policy="skip",
        enforce_v1_weights=True,
    )
    if not result.defined or result.top_answer is None:
        return None
    score = result.score(result.top_answer)
    if score is None:
        return None
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method=method,
        dataset=dataset,
        question_id=str(payload.get("question_id", "")),
        confidence=float(score),
        correct=_correct(result.top_answer, gold),
        predicted_answer=result.top_answer,
        gold_answer=gold,
    )


def _full_observation(
    dataset: str,
    payload: Mapping[str, Any],
    *,
    beta: float,
    threshold: float,
) -> Observation | None:
    samples = [
        dict(sample)
        for sample in payload.get("samples", [])
        if isinstance(sample, Mapping)
    ]
    weighted, _ = annotate_dependency_weights(
        samples,
        beta=beta,
        similarity_threshold=threshold,
        answer_sensitive=True,
    )
    result = compute_relssc_full(weighted, zero_weight_policy="skip")
    if not result.defined or result.top_answer is None:
        return None
    score = result.score(result.top_answer)
    if score is None:
        return None
    gold = _normalize_answer(payload.get("gold_original_answer"))
    return Observation(
        method="Full-RelSSC",
        dataset=dataset,
        question_id=str(payload.get("question_id", "")),
        confidence=float(score),
        correct=_correct(result.top_answer, gold),
        predicted_answer=result.top_answer,
        gold_answer=gold,
    )


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _relation_diagnostic(
    dataset: str,
    payloads: Sequence[Mapping[str, Any]],
    subtype: str,
) -> RelationDiagnostic:
    available = 0
    identity_correct = 0
    identity_wrong = 0
    relation_correct = 0
    preserved = 0
    harmed = 0
    repaired = 0
    wrong_broken = 0
    wrong_same = 0
    winner_agree = 0
    attempted = 0
    valid = 0

    for payload in payloads:
        identity_samples = _samples_for_subtypes(payload, ("identity",))
        relation_samples = _samples_for_subtypes(payload, (subtype,))
        if not identity_samples or not relation_samples:
            continue

        identity_obs = _observation_from_samples(
            dataset, payload, identity_samples, "identity"
        )
        relation_obs = _observation_from_samples(
            dataset, payload, relation_samples, subtype
        )
        if identity_obs is None or relation_obs is None:
            continue

        available += 1
        attempted += len(relation_samples)
        valid += sum(
            bool(sample.get("is_valid_answer", True))
            and sample.get("canonicalized_answer", sample.get("canonical_answer")) is not None
            for sample in relation_samples
        )
        relation_correct += relation_obs.correct
        winner_agree += int(
            relation_obs.predicted_answer == identity_obs.predicted_answer
        )

        if identity_obs.correct:
            identity_correct += 1
            if relation_obs.correct:
                preserved += 1
            else:
                harmed += 1
        else:
            identity_wrong += 1
            if relation_obs.correct:
                repaired += 1
            if relation_obs.predicted_answer != identity_obs.predicted_answer:
                wrong_broken += 1
            elif not relation_obs.correct:
                wrong_same += 1

    return RelationDiagnostic(
        dataset=dataset,
        relation_subtype=subtype,
        available_questions=available,
        identity_correct_questions=identity_correct,
        identity_wrong_questions=identity_wrong,
        relation_accuracy=_safe_rate(relation_correct, available),
        correct_preservation_rate=_safe_rate(preserved, identity_correct),
        harm_rate_given_identity_correct=_safe_rate(harmed, identity_correct),
        error_repair_rate=_safe_rate(repaired, identity_wrong),
        wrong_consensus_break_rate=_safe_rate(wrong_broken, identity_wrong),
        wrong_same_answer_rate=_safe_rate(wrong_same, identity_wrong),
        winner_agreement_rate=_safe_rate(winner_agree, available),
        attempted_samples=attempted,
        valid_samples=valid,
        valid_response_rate=_safe_rate(valid, attempted),
    )


def _dependency_diagnostic(
    dataset: str,
    payloads: Sequence[Mapping[str, Any]],
    *,
    beta: float,
    threshold: float,
) -> DependencyDiagnostic:
    question_count = 0
    valid_samples = 0
    cluster_count = 0
    cluster_size_sum = 0
    max_cluster_size = 0
    multi_clusters = 0
    samples_in_multi = 0
    effective_mass = 0.0

    for payload in payloads:
        samples = [
            dict(sample)
            for sample in payload.get("samples", [])
            if isinstance(sample, Mapping)
        ]
        _, summary = annotate_dependency_weights(
            samples,
            beta=beta,
            similarity_threshold=threshold,
            answer_sensitive=True,
        )
        question_count += 1
        valid_samples += summary.valid_sample_count
        sizes = list(summary.cluster_sizes.values())
        cluster_count += len(sizes)
        cluster_size_sum += sum(sizes)
        if sizes:
            max_cluster_size = max(max_cluster_size, max(sizes))
        multi_clusters += sum(size > 1 for size in sizes)
        samples_in_multi += sum(size for size in sizes if size > 1)
        effective_mass += summary.effective_cluster_mass

    return DependencyDiagnostic(
        dataset=dataset,
        questions=question_count,
        valid_samples=valid_samples,
        clusters=cluster_count,
        mean_cluster_size=_safe_rate(cluster_size_sum, cluster_count),
        max_cluster_size=max_cluster_size,
        multi_sample_cluster_fraction=_safe_rate(multi_clusters, cluster_count),
        sample_fraction_in_multi_sample_clusters=_safe_rate(
            samples_in_multi, valid_samples
        ),
        effective_cluster_mass_ratio=_safe_rate(effective_mass, valid_samples),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{100.0 * value:.2f}"


def main() -> None:
    args = parse_args()
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0,1]")
    if not 0.0 <= args.strategy_similarity_threshold <= 1.0:
        raise ValueError("--strategy-similarity-threshold must be in [0,1]")

    input_root = resolve_path(args.input_root)
    baseline_root = resolve_path(args.baseline_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[MetricRow] = []
    relation_rows: list[RelationDiagnostic] = []
    dependency_rows: list[DependencyDiagnostic] = []
    question_details: list[dict[str, Any]] = []

    print("Numeric RelaCaTS relation diagnosis")
    print("==================================")
    print("Subset rows are diagnostics, not equal-budget final comparisons.\n")

    for dataset in args.datasets:
        baseline_index = _index_payloads(_load_payloads(baseline_root, dataset))
        relational_index = _index_payloads(_load_payloads(input_root, dataset))
        shared_ids = sorted(set(baseline_index) & set(relational_index))
        if not shared_ids:
            raise ValueError(f"{dataset}: no shared question IDs")

        baseline_observations_paper: list[Observation] = []
        baseline_observations_valid: list[Observation] = []
        profile_observations: dict[str, list[Observation]] = {
            name: [] for name in PROFILE_SUBTYPES
        }
        full_observations: list[Observation] = []
        relational_payloads: list[Mapping[str, Any]] = []

        for question_id in shared_ids:
            baseline_payload = baseline_index[question_id]
            relational_payload = relational_index[question_id]
            relational_payloads.append(relational_payload)

            paper_obs = _ssc_observation(
                dataset,
                baseline_payload,
                scope="all",
                invalid_policy="paper",
            )
            if paper_obs is not None:
                baseline_observations_paper.append(paper_obs)

            valid_obs = _ssc_observation(
                dataset,
                baseline_payload,
                scope="all",
                invalid_policy="valid-only",
            )
            if valid_obs is not None:
                baseline_observations_valid.append(valid_obs)

            detail: dict[str, Any] = {
                "dataset": dataset,
                "question_id": question_id,
                "gold_answer": relational_payload.get("gold_original_answer"),
            }
            for profile_name, subtypes in PROFILE_SUBTYPES.items():
                selected = _samples_for_subtypes(relational_payload, subtypes)
                observation = _observation_from_samples(
                    dataset,
                    relational_payload,
                    selected,
                    profile_name,
                )
                if observation is not None:
                    profile_observations[profile_name].append(observation)
                    detail[profile_name] = {
                        "n_samples": len(selected),
                        "prediction": observation.predicted_answer,
                        "confidence": observation.confidence,
                        "correct": observation.correct,
                    }

            full_obs = _full_observation(
                dataset,
                relational_payload,
                beta=args.beta,
                threshold=args.strategy_similarity_threshold,
            )
            if full_obs is not None:
                full_observations.append(full_obs)
                detail["Full-RelSSC"] = {
                    "prediction": full_obs.predicted_answer,
                    "confidence": full_obs.confidence,
                    "correct": full_obs.correct,
                }
            question_details.append(detail)

        metric_rows.append(
            _metric_row(dataset, "SSC-32I-paper", baseline_observations_paper, args.n_bins)
        )
        metric_rows.append(
            _metric_row(dataset, "SSC-32I-valid-only", baseline_observations_valid, args.n_bins)
        )
        for profile_name in PROFILE_SUBTYPES:
            metric_rows.append(
                _metric_row(
                    dataset,
                    profile_name,
                    profile_observations[profile_name],
                    args.n_bins,
                )
            )
        metric_rows.append(
            _metric_row(dataset, "Full-RelSSC", full_observations, args.n_bins)
        )

        for subtype in RELATION_SUBTYPES[1:]:
            relation_rows.append(
                _relation_diagnostic(dataset, relational_payloads, subtype)
            )
        dependency_rows.append(
            _dependency_diagnostic(
                dataset,
                relational_payloads,
                beta=args.beta,
                threshold=args.strategy_similarity_threshold,
            )
        )

        print(f"[{dataset}] shared questions: {len(shared_ids)}")
        for row in metric_rows:
            if row.dataset != dataset:
                continue
            print(
                f"  {row.method:<28s} "
                f"n={row.n:<4d} acc={100*row.accuracy:6.2f} "
                f"ECE={100*row.ece_paper:6.2f} Brier={row.brier:.4f}"
            )
        print()

    _write_csv(
        output_dir / "numeric_relation_ablation_metrics.csv",
        [asdict(row) for row in metric_rows],
    )
    _write_csv(
        output_dir / "numeric_relation_diagnostics.csv",
        [asdict(row) for row in relation_rows],
    )
    _write_csv(
        output_dir / "numeric_dependency_diagnostics.csv",
        [asdict(row) for row in dependency_rows],
    )
    with (output_dir / "numeric_relation_question_details.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in question_details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Relation preservation / repair diagnostics")
    print("------------------------------------------")
    for row in relation_rows:
        print(
            f"{row.dataset:<7s} {row.relation_subtype:<23s} "
            f"preserve={_fmt(row.correct_preservation_rate)}% "
            f"harm={_fmt(row.harm_rate_given_identity_correct)}% "
            f"repair={_fmt(row.error_repair_rate)}% "
            f"break-wrong={_fmt(row.wrong_consensus_break_rate)}% "
            f"valid={_fmt(row.valid_response_rate)}%"
        )

    print("\nDependency-clustering diagnostics")
    print("---------------------------------")
    for row in dependency_rows:
        print(
            f"{row.dataset:<7s} clusters={row.clusters} "
            f"mean_size={row.mean_cluster_size:.3f} max_size={row.max_cluster_size} "
            f"samples_in_multi={100*row.sample_fraction_in_multi_sample_clusters:.2f}% "
            f"effective_mass_ratio={row.effective_cluster_mass_ratio:.3f}"
        )

    print("\nOutputs:")
    for name in (
        "numeric_relation_ablation_metrics.csv",
        "numeric_relation_diagnostics.csv",
        "numeric_dependency_diagnostics.csv",
        "numeric_relation_question_details.jsonl",
    ):
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
