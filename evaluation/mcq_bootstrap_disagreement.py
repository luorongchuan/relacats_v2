"""Paired bootstrap and RelSC/RelSSC disagreement analysis for MCQ RelaCaTS.

This module is CPU-only and consumes already-generated question JSON files.
It does not regenerate responses and does not modify training data.

The analysis has two goals:

1. Paired bootstrap confidence intervals over questions for the matched-budget
   comparisons used in the MCQ study.
2. Per-question diagnosis of when count-based relational self-consistency
   (RelSC) and confidence-weighted relational self-consistency (RelSSC)
   disagree.

Terminology
-----------
``32I-SC`` and ``32I-SSC`` are computed from the separate identity-only
32-response pool. ``Mapped-SC-paper`` is the count-based SC computed over the
relation-view pool after canonicalization, using the paper-compatible invalid
answer policy. ``RelSC-valid`` is the same count-based relational aggregation
but drops invalid/canonicalization-failed samples so that its support domain
matches RelSSC. ``RelSSC`` uses the existing RelaCaTS confidence-weighted
aggregation over valid canonicalized answers.

Gold labels are used here only for post-hoc evaluation/diagnosis.  They must
not be used to define test-time adaptive weights or tune a final method on the
reported test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from relacats_v2.core import compute_relssc
from relacats_v2.evaluation.reproduce_table1 import (
    Observation,
    _finite_probability,
    _filtered_samples,
    _index_payloads,
    _load_payloads,
    _relssc_observation,
    _sample_answer,
    _sc_observation,
    _ssc_observation,
    calculate_ece_paper,
    calculate_ece_strict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "relacats_v1/outputs/generated_data/qwen2_5_7b_instruct"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT
    / "relacats_v1/outputs/generated_data_identity_only/qwen2_5_7b_instruct"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "relacats_v2/outputs/mcq_bootstrap_disagreement"


@dataclass(frozen=True)
class BootstrapRow:
    dataset: str
    comparison: str
    metric: str
    n: int
    reference_value: float
    target_value: float
    point_delta_target_minus_reference: float
    ci_low_95: float
    ci_high_95: float
    probability_improvement: float
    ci_excludes_zero: bool
    bootstrap_replicates: int
    seed: int


@dataclass(frozen=True)
class DisagreementRow:
    dataset: str
    question_id: str
    relsc_prediction: str | None
    relssc_prediction: str | None
    relsc_confidence: float
    relssc_confidence: float
    confidence_shift_relssc_minus_relsc: float
    relsc_correct: int
    relssc_correct: int
    winner_changed: bool
    outcome: str
    tv_distance: float
    brier_relsc: float
    brier_relssc: float
    brier_delta_relssc_minus_relsc: float
    valid_sample_count: int
    confidence_weight_sample_count: int


@dataclass(frozen=True)
class DisagreementSummary:
    dataset: str
    n: int
    winner_changed_rate: float
    relsc_accuracy: float
    relssc_accuracy: float
    accuracy_delta_relssc_minus_relsc: float
    relsc_only_correct: int
    relssc_only_correct: int
    both_correct: int
    both_wrong: int
    mean_tv_distance: float
    median_tv_distance: float
    mean_abs_confidence_shift: float
    mean_brier_relsc: float
    mean_brier_relssc: float
    mean_brier_delta_relssc_minus_relsc: float


@dataclass(frozen=True)
class TVBinSummary:
    dataset: str
    tv_bin: str
    n: int
    mean_tv_distance: float
    winner_changed_rate: float
    relsc_accuracy: float
    relssc_accuracy: float
    accuracy_delta_relssc_minus_relsc: float
    relsc_only_correct_rate: float
    relssc_only_correct_rate: float
    mean_brier_delta_relssc_minus_relsc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=(
            "arc_easy",
            "commonsense_qa",
            "logiqa",
            "openbookqa",
            "reclor",
            "sciq",
            "winogrande",
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _metric(observations: Sequence[Observation], metric: str, n_bins: int) -> float:
    if not observations:
        return float("nan")
    labels = np.asarray([obs.correct for obs in observations], dtype=float)
    scores = np.asarray([obs.confidence for obs in observations], dtype=float)
    if metric == "accuracy":
        return 100.0 * float(np.mean(labels))
    if metric == "brier":
        return 100.0 * float(np.mean((scores - labels) ** 2))
    if metric == "ece_paper":
        return 100.0 * calculate_ece_paper(labels, scores, n_bins=n_bins)
    if metric == "ece_strict":
        return 100.0 * calculate_ece_strict(labels, scores, n_bins=n_bins)
    raise ValueError(f"unknown metric: {metric}")


def _bootstrap_pair(
    *,
    dataset: str,
    comparison: str,
    reference: Sequence[Observation],
    target: Sequence[Observation],
    metric: str,
    n_bins: int,
    replicates: int,
    seed: int,
) -> BootstrapRow:
    reference_by_id = {obs.question_id: obs for obs in reference}
    target_by_id = {obs.question_id: obs for obs in target}
    shared_ids = sorted(set(reference_by_id) & set(target_by_id))
    if not shared_ids:
        raise ValueError(f"{dataset}/{comparison}: no paired observations")

    ref = [reference_by_id[qid] for qid in shared_ids]
    tgt = [target_by_id[qid] for qid in shared_ids]
    n = len(shared_ids)

    ref_labels = np.asarray([obs.correct for obs in ref], dtype=float)
    ref_scores = np.asarray([obs.confidence for obs in ref], dtype=float)
    tgt_labels = np.asarray([obs.correct for obs in tgt], dtype=float)
    tgt_scores = np.asarray([obs.confidence for obs in tgt], dtype=float)

    def metric_arrays(labels: np.ndarray, scores: np.ndarray) -> float:
        if metric == "accuracy":
            return 100.0 * float(np.mean(labels))
        if metric == "brier":
            return 100.0 * float(np.mean((scores - labels) ** 2))
        if metric == "ece_paper":
            return 100.0 * calculate_ece_paper(labels, scores, n_bins=n_bins)
        if metric == "ece_strict":
            return 100.0 * calculate_ece_strict(labels, scores, n_bins=n_bins)
        raise ValueError(metric)

    reference_value = metric_arrays(ref_labels, ref_scores)
    target_value = metric_arrays(tgt_labels, tgt_scores)
    point_delta = target_value - reference_value

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, n, size=n)
        ref_value = metric_arrays(ref_labels[chosen], ref_scores[chosen])
        tgt_value = metric_arrays(tgt_labels[chosen], tgt_scores[chosen])
        deltas[index] = tgt_value - ref_value

    low, high = np.percentile(deltas, [2.5, 97.5])
    if metric == "accuracy":
        probability_improvement = float(
            np.mean(deltas > 0) + 0.5 * np.mean(deltas == 0)
        )
    else:
        probability_improvement = float(
            np.mean(deltas < 0) + 0.5 * np.mean(deltas == 0)
        )

    return BootstrapRow(
        dataset=dataset,
        comparison=comparison,
        metric=metric,
        n=n,
        reference_value=reference_value,
        target_value=target_value,
        point_delta_target_minus_reference=point_delta,
        ci_low_95=float(low),
        ci_high_95=float(high),
        probability_improvement=probability_improvement,
        ci_excludes_zero=bool(high < 0 or low > 0),
        bootstrap_replicates=replicates,
        seed=seed,
    )


def _relsc_distribution(payload: Mapping[str, Any]) -> tuple[dict[str, float], int]:
    support: dict[str, float] = {}
    denominator = 0
    for sample in _filtered_samples(payload, scope="all"):
        answer = _sample_answer(sample, keep_invalid=False)
        if answer is None:
            continue
        support[answer] = support.get(answer, 0.0) + 1.0
        denominator += 1
    if denominator <= 0:
        return {}, 0
    return {answer: value / denominator for answer, value in support.items()}, denominator


def _relssc_distribution(payload: Mapping[str, Any]) -> tuple[dict[str, float], int]:
    samples = [
        dict(sample)
        for sample in payload.get("samples", [])
        if isinstance(sample, Mapping)
    ]
    result = compute_relssc(
        samples,
        zero_weight_policy="skip",
        enforce_v1_weights=True,
    )
    if not result.defined:
        return {}, 0
    return dict(result.scores), result.valid_sample_count


def _tv_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in keys)


def _outcome(relsc_correct: int, relssc_correct: int) -> str:
    if relsc_correct and relssc_correct:
        return "both_correct"
    if relsc_correct and not relssc_correct:
        return "relsc_only_correct"
    if relssc_correct and not relsc_correct:
        return "relssc_only_correct"
    return "both_wrong"


def _disagreement_row(
    dataset: str,
    payload: Mapping[str, Any],
) -> DisagreementRow | None:
    relsc = _sc_observation(
        dataset,
        payload,
        scope="all",
        invalid_policy="valid-only",
    )
    relssc = _relssc_observation(dataset, payload)
    if relsc is None or relssc is None:
        return None

    relsc_dist, relsc_n = _relsc_distribution(payload)
    relssc_dist, relssc_n = _relssc_distribution(payload)
    if not relsc_dist or not relssc_dist:
        return None

    tv = _tv_distance(relsc_dist, relssc_dist)
    brier_relsc = (float(relsc.confidence) - float(relsc.correct)) ** 2
    brier_relssc = (float(relssc.confidence) - float(relssc.correct)) ** 2

    return DisagreementRow(
        dataset=dataset,
        question_id=relsc.question_id,
        relsc_prediction=relsc.predicted_answer,
        relssc_prediction=relssc.predicted_answer,
        relsc_confidence=float(relsc.confidence),
        relssc_confidence=float(relssc.confidence),
        confidence_shift_relssc_minus_relsc=float(relssc.confidence - relsc.confidence),
        relsc_correct=int(relsc.correct),
        relssc_correct=int(relssc.correct),
        winner_changed=bool(relsc.predicted_answer != relssc.predicted_answer),
        outcome=_outcome(int(relsc.correct), int(relssc.correct)),
        tv_distance=float(tv),
        brier_relsc=float(brier_relsc),
        brier_relssc=float(brier_relssc),
        brier_delta_relssc_minus_relsc=float(brier_relssc - brier_relsc),
        valid_sample_count=relsc_n,
        confidence_weight_sample_count=relssc_n,
    )


def _summary(dataset: str, rows: Sequence[DisagreementRow]) -> DisagreementSummary:
    n = len(rows)
    if n <= 0:
        raise ValueError(f"{dataset}: empty disagreement rows")
    relsc_acc = 100.0 * np.mean([row.relsc_correct for row in rows])
    relssc_acc = 100.0 * np.mean([row.relssc_correct for row in rows])
    brier_relsc = 100.0 * np.mean([row.brier_relsc for row in rows])
    brier_relssc = 100.0 * np.mean([row.brier_relssc for row in rows])
    return DisagreementSummary(
        dataset=dataset,
        n=n,
        winner_changed_rate=float(np.mean([row.winner_changed for row in rows])),
        relsc_accuracy=float(relsc_acc),
        relssc_accuracy=float(relssc_acc),
        accuracy_delta_relssc_minus_relsc=float(relssc_acc - relsc_acc),
        relsc_only_correct=sum(row.outcome == "relsc_only_correct" for row in rows),
        relssc_only_correct=sum(row.outcome == "relssc_only_correct" for row in rows),
        both_correct=sum(row.outcome == "both_correct" for row in rows),
        both_wrong=sum(row.outcome == "both_wrong" for row in rows),
        mean_tv_distance=float(np.mean([row.tv_distance for row in rows])),
        median_tv_distance=float(np.median([row.tv_distance for row in rows])),
        mean_abs_confidence_shift=float(
            np.mean([abs(row.confidence_shift_relssc_minus_relsc) for row in rows])
        ),
        mean_brier_relsc=float(brier_relsc),
        mean_brier_relssc=float(brier_relssc),
        mean_brier_delta_relssc_minus_relsc=float(brier_relssc - brier_relsc),
    )


def _tv_bin(value: float) -> str:
    if value <= 1e-12:
        return "0"
    if value <= 0.02:
        return "(0,0.02]"
    if value <= 0.05:
        return "(0.02,0.05]"
    if value <= 0.10:
        return "(0.05,0.10]"
    if value <= 0.20:
        return "(0.10,0.20]"
    return ">0.20"


def _tv_bin_summaries(dataset: str, rows: Sequence[DisagreementRow]) -> list[TVBinSummary]:
    order = ["0", "(0,0.02]", "(0.02,0.05]", "(0.05,0.10]", "(0.10,0.20]", ">0.20"]
    grouped: dict[str, list[DisagreementRow]] = {label: [] for label in order}
    for row in rows:
        grouped[_tv_bin(row.tv_distance)].append(row)

    result: list[TVBinSummary] = []
    for label in order:
        members = grouped[label]
        if not members:
            continue
        n = len(members)
        relsc_acc = float(np.mean([row.relsc_correct for row in members]))
        relssc_acc = float(np.mean([row.relssc_correct for row in members]))
        result.append(
            TVBinSummary(
                dataset=dataset,
                tv_bin=label,
                n=n,
                mean_tv_distance=float(np.mean([row.tv_distance for row in members])),
                winner_changed_rate=float(np.mean([row.winner_changed for row in members])),
                relsc_accuracy=100.0 * relsc_acc,
                relssc_accuracy=100.0 * relssc_acc,
                accuracy_delta_relssc_minus_relsc=100.0 * (relssc_acc - relsc_acc),
                relsc_only_correct_rate=float(
                    np.mean([row.outcome == "relsc_only_correct" for row in members])
                ),
                relssc_only_correct_rate=float(
                    np.mean([row.outcome == "relssc_only_correct" for row in members])
                ),
                mean_brier_delta_relssc_minus_relsc=100.0
                * float(np.mean([row.brier_delta_relssc_minus_relsc for row in members])),
            )
        )
    return result


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


def main() -> None:
    args = parse_args()
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    input_root = resolve_path(args.input_root)
    baseline_root = resolve_path(args.baseline_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_rows: list[BootstrapRow] = []
    disagreement_rows: list[DisagreementRow] = []
    summary_rows: list[DisagreementSummary] = []
    tv_rows: list[TVBinSummary] = []

    print("MCQ RelaCaTS bootstrap + RelSC/RelSSC disagreement analysis")
    print("=============================================================")
    print(f"Bootstrap replicates: {args.bootstrap_replicates}, seed={args.seed}\n")

    for dataset_index, dataset in enumerate(args.datasets):
        baseline_index = _index_payloads(_load_payloads(baseline_root, dataset))
        relation_index = _index_payloads(_load_payloads(input_root, dataset))
        shared_ids = sorted(set(baseline_index) & set(relation_index))
        if not shared_ids:
            raise ValueError(f"{dataset}: no shared question IDs")

        base_sc: list[Observation] = []
        base_ssc: list[Observation] = []
        mapped_sc_paper: list[Observation] = []
        relsc_valid: list[Observation] = []
        relssc: list[Observation] = []
        dataset_disagreement: list[DisagreementRow] = []

        for question_id in shared_ids:
            baseline_payload = baseline_index[question_id]
            relation_payload = relation_index[question_id]

            observations = (
                (
                    base_sc,
                    _sc_observation(
                        dataset,
                        baseline_payload,
                        scope="all",
                        invalid_policy="paper",
                    ),
                ),
                (
                    base_ssc,
                    _ssc_observation(
                        dataset,
                        baseline_payload,
                        scope="all",
                        invalid_policy="paper",
                    ),
                ),
                (
                    mapped_sc_paper,
                    _sc_observation(
                        dataset,
                        relation_payload,
                        scope="all",
                        invalid_policy="paper",
                    ),
                ),
                (
                    relsc_valid,
                    _sc_observation(
                        dataset,
                        relation_payload,
                        scope="all",
                        invalid_policy="valid-only",
                    ),
                ),
                (relssc, _relssc_observation(dataset, relation_payload)),
            )
            for target_list, observation in observations:
                if observation is not None:
                    target_list.append(observation)

            disagreement = _disagreement_row(dataset, relation_payload)
            if disagreement is not None:
                dataset_disagreement.append(disagreement)
                disagreement_rows.append(disagreement)

        comparisons = (
            ("32I-SC -> Mapped-SC-paper", base_sc, mapped_sc_paper),
            ("32I-SSC -> RelSSC", base_ssc, relssc),
            ("RelSC-valid -> RelSSC", relsc_valid, relssc),
        )
        metrics = ("accuracy", "ece_paper", "ece_strict", "brier")
        for comparison_index, (name, reference, target) in enumerate(comparisons):
            for metric_index, metric in enumerate(metrics):
                bootstrap_rows.append(
                    _bootstrap_pair(
                        dataset=dataset,
                        comparison=name,
                        reference=reference,
                        target=target,
                        metric=metric,
                        n_bins=args.n_bins,
                        replicates=args.bootstrap_replicates,
                        seed=(
                            args.seed
                            + 10000 * dataset_index
                            + 100 * comparison_index
                            + metric_index
                        ),
                    )
                )

        summary = _summary(dataset, dataset_disagreement)
        summary_rows.append(summary)
        tv_rows.extend(_tv_bin_summaries(dataset, dataset_disagreement))

        print(f"[{dataset}] shared={len(shared_ids)}")
        print(
            "  RelSC-valid vs RelSSC: "
            f"winner_changed={100.0*summary.winner_changed_rate:.2f}%  "
            f"RelSC-only-correct={summary.relsc_only_correct}  "
            f"RelSSC-only-correct={summary.relssc_only_correct}  "
            f"mean-TV={summary.mean_tv_distance:.4f}  "
            f"dBrier={summary.mean_brier_delta_relssc_minus_relsc:+.4f}"
        )

        key_rows = [
            row
            for row in bootstrap_rows
            if row.dataset == dataset
            and row.comparison == "32I-SC -> Mapped-SC-paper"
            and row.metric in {"accuracy", "ece_paper", "brier"}
        ]
        for row in key_rows:
            print(
                f"  bootstrap {row.metric:<9s}: "
                f"delta={row.point_delta_target_minus_reference:+.3f} "
                f"95%CI=[{row.ci_low_95:+.3f},{row.ci_high_95:+.3f}] "
                f"P(improve)={row.probability_improvement:.3f}"
            )
        print()

    _write_csv(
        output_dir / "bootstrap_paired_deltas.csv",
        [asdict(row) for row in bootstrap_rows],
    )
    _write_csv(
        output_dir / "relsc_relssc_disagreement_questions.csv",
        [asdict(row) for row in disagreement_rows],
    )
    _write_csv(
        output_dir / "relsc_relssc_disagreement_summary.csv",
        [asdict(row) for row in summary_rows],
    )
    _write_csv(
        output_dir / "relsc_relssc_tv_bins.csv",
        [asdict(row) for row in tv_rows],
    )

    report = {
        "input_root": str(input_root),
        "baseline_root": str(baseline_root),
        "datasets": list(args.datasets),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "notes": [
            "Bootstrap resampling is paired by question ID.",
            "ECE/Brier deltas < 0 are improvements; accuracy deltas > 0 are improvements.",
            "Mapped-SC-paper reproduces the paper-compatible invalid-answer policy used in the reported MCQ table.",
            "RelSC-valid drops invalid/canonicalization-failed answers to match the RelSSC support domain.",
            "Gold labels are diagnostic only and must not define a final test-time adaptive rule.",
        ],
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Outputs:")
    for filename in (
        "bootstrap_paired_deltas.csv",
        "relsc_relssc_disagreement_questions.csv",
        "relsc_relssc_disagreement_summary.csv",
        "relsc_relssc_tv_bins.csv",
        "analysis_report.json",
    ):
        print(f"  {output_dir / filename}")


if __name__ == "__main__":
    main()
