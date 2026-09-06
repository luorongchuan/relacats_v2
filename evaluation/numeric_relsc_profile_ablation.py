"""CPU-only pure-RelSC profile ablation for GSM8K/SVAMP.

This script reuses already-generated JSONs.  It diagnoses which numeric
relation subtypes help or hurt count-based RelSC.

Two baselines are reported for every subset profile:

1. SSC-32I: the full 32-response identity-only CaTS baseline.
2. SSC-matched: an identity-only SSC baseline using the same number of stored
   responses as that profile uses on each question.

The matched-budget comparison is the important diagnostic for relation quality:
it separates "this relation is harmful" from "this subset merely used fewer
samples than the 32I baseline".

Subset profiles still reuse the already-generated relation pool, so they are
not final equal-generation-budget results.  Once a profile is selected, it
should be regenerated with the full N=32 budget redistributed over the chosen
views before a final paper comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from relacats_v2.evaluation.reproduce_table1 import (
    Observation,
    _filtered_samples,
    _index_payloads,
    _load_payloads,
    _sample_answer,
    _sc_observation,
    _ssc_observation,
    calculate_ece_paper,
    calculate_ece_strict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

RELATION_SUBTYPES = (
    "identity",
    "layout_wrapper",
    "number_representation",
    "equivalent_quantity",
)

PROFILE_SUBTYPES: dict[str, tuple[str, ...]] = {
    "Identity-only (stored g0)": ("identity",),
    "Layout-only": ("layout_wrapper",),
    "Number-only": ("number_representation",),
    "Equivalent-only": ("equivalent_quantity",),
    "Identity+Layout": ("identity", "layout_wrapper"),
    "Identity+Number": ("identity", "number_representation"),
    "Identity+Equivalent": ("identity", "equivalent_quantity"),
    "Drop-Layout": ("identity", "number_representation", "equivalent_quantity"),
    "Drop-Number": ("identity", "layout_wrapper", "equivalent_quantity"),
    "Drop-Equivalent": ("identity", "layout_wrapper", "number_representation"),
    "All-Relations": RELATION_SUBTYPES,
}


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    method: str
    n: int
    mean_selected_samples: float
    mean_valid_samples: float
    accuracy: float
    ece_paper: float
    ece_strict: float
    brier: float
    ssc32_accuracy: float
    ssc32_ece_paper: float
    ssc32_brier: float
    delta_accuracy_vs_ssc32i: float
    delta_ece_vs_ssc32i: float
    delta_brier_vs_ssc32i: float
    ssc_matched_accuracy: float
    ssc_matched_ece_paper: float
    ssc_matched_brier: float
    delta_accuracy_vs_ssc_matched: float
    delta_ece_vs_ssc_matched: float
    delta_brier_vs_ssc_matched: float


@dataclass(frozen=True)
class BootstrapRow:
    dataset: str
    profile: str
    comparison: str
    metric: str
    n: int
    baseline_value: float
    profile_value: float
    delta_profile_minus_baseline: float
    ci_low_95: float
    ci_high_95: float
    probability_improvement: float
    bootstrap_replicates: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--datasets", nargs="+", default=("gsm8k", "svamp"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def relation_subtype(sample: Mapping[str, Any]) -> str:
    subtype = str(sample.get("relation_subtype", "")).strip().lower()
    if subtype:
        return subtype
    relation_id = str(sample.get("relation_id", "g0")).strip().lower()
    relation_type = str(sample.get("relation_type", "")).strip().lower()
    if relation_id == "g0" or relation_type == "identity":
        return "identity"
    return relation_type or relation_id


def select_samples(
    payload: Mapping[str, Any], subtypes: Sequence[str]
) -> list[dict[str, Any]]:
    allowed = set(subtypes)
    return [
        dict(sample)
        for sample in payload.get("samples", [])
        if isinstance(sample, Mapping) and relation_subtype(sample) in allowed
    ]


def profile_observation(
    dataset: str,
    payload: Mapping[str, Any],
    profile: str,
    subtypes: Sequence[str],
) -> tuple[Observation | None, int, int]:
    selected = select_samples(payload, subtypes)
    valid = sum(
        _sample_answer(sample, keep_invalid=False) is not None for sample in selected
    )
    if not selected:
        return None, 0, 0
    sliced = dict(payload)
    sliced["samples"] = selected
    obs = _sc_observation(dataset, sliced, scope="all", invalid_policy="valid-only")
    if obs is not None:
        obs = Observation(
            method=profile,
            dataset=obs.dataset,
            question_id=obs.question_id,
            confidence=obs.confidence,
            correct=obs.correct,
            predicted_answer=obs.predicted_answer,
            gold_answer=obs.gold_answer,
        )
    return obs, len(selected), valid


def matched_ssc_observation(
    dataset: str,
    baseline_payload: Mapping[str, Any],
    sample_budget: int,
    method: str,
) -> Observation | None:
    """SSC on the first deterministic k identity-only samples for this question."""
    if sample_budget <= 0:
        return None
    samples = _filtered_samples(baseline_payload, scope="all")
    if not samples:
        return None
    k = min(sample_budget, len(samples))
    sliced = dict(baseline_payload)
    sliced["samples"] = samples[:k]
    obs = _ssc_observation(
        dataset,
        sliced,
        scope="all",
        invalid_policy="paper",
    )
    if obs is None:
        return None
    return Observation(
        method=method,
        dataset=obs.dataset,
        question_id=obs.question_id,
        confidence=obs.confidence,
        correct=obs.correct,
        predicted_answer=obs.predicted_answer,
        gold_answer=obs.gold_answer,
    )


def metric_value(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    n_bins: int,
) -> float:
    if metric == "accuracy":
        return 100.0 * float(np.mean(labels))
    if metric == "brier":
        return 100.0 * float(np.mean((scores - labels) ** 2))
    label_list = labels.tolist()
    score_list = scores.tolist()
    if metric == "ece_paper":
        return 100.0 * calculate_ece_paper(label_list, score_list, n_bins=n_bins)
    if metric == "ece_strict":
        return 100.0 * calculate_ece_strict(label_list, score_list, n_bins=n_bins)
    raise ValueError(metric)


def observation_metrics(
    observations: Sequence[Observation],
    shared_ids: Sequence[str],
    n_bins: int,
) -> tuple[float, float, float, float]:
    by_id = {obs.question_id: obs for obs in observations}
    labels = np.asarray([by_id[q].correct for q in shared_ids], dtype=float)
    scores = np.asarray([by_id[q].confidence for q in shared_ids], dtype=float)
    return (
        metric_value(labels, scores, "accuracy", n_bins),
        metric_value(labels, scores, "ece_paper", n_bins),
        metric_value(labels, scores, "ece_strict", n_bins),
        metric_value(labels, scores, "brier", n_bins),
    )


def metric_row(
    dataset: str,
    method: str,
    observations: Sequence[Observation],
    selected_counts: Sequence[int],
    valid_counts: Sequence[int],
    baseline32: Sequence[Observation],
    baseline_matched: Sequence[Observation],
    n_bins: int,
) -> MetricRow:
    target_by_id = {obs.question_id: obs for obs in observations}
    base32_by_id = {obs.question_id: obs for obs in baseline32}
    matched_by_id = {obs.question_id: obs for obs in baseline_matched}
    shared = sorted(set(target_by_id) & set(base32_by_id) & set(matched_by_id))
    if not shared:
        raise ValueError(f"{dataset}/{method}: no paired observations")

    acc, ece, ece_strict, brier = observation_metrics(observations, shared, n_bins)
    b32_acc, b32_ece, _, b32_brier = observation_metrics(
        baseline32, shared, n_bins
    )
    bm_acc, bm_ece, _, bm_brier = observation_metrics(
        baseline_matched, shared, n_bins
    )

    return MetricRow(
        dataset=dataset,
        method=method,
        n=len(shared),
        mean_selected_samples=(
            float(np.mean(selected_counts)) if selected_counts else float("nan")
        ),
        mean_valid_samples=(
            float(np.mean(valid_counts)) if valid_counts else float("nan")
        ),
        accuracy=acc,
        ece_paper=ece,
        ece_strict=ece_strict,
        brier=brier,
        ssc32_accuracy=b32_acc,
        ssc32_ece_paper=b32_ece,
        ssc32_brier=b32_brier,
        delta_accuracy_vs_ssc32i=acc - b32_acc,
        delta_ece_vs_ssc32i=ece - b32_ece,
        delta_brier_vs_ssc32i=brier - b32_brier,
        ssc_matched_accuracy=bm_acc,
        ssc_matched_ece_paper=bm_ece,
        ssc_matched_brier=bm_brier,
        delta_accuracy_vs_ssc_matched=acc - bm_acc,
        delta_ece_vs_ssc_matched=ece - bm_ece,
        delta_brier_vs_ssc_matched=brier - bm_brier,
    )


def bootstrap_rows(
    dataset: str,
    profile: str,
    comparison: str,
    baseline: Sequence[Observation],
    target: Sequence[Observation],
    n_bins: int,
    replicates: int,
    seed: int,
) -> list[BootstrapRow]:
    base = {obs.question_id: obs for obs in baseline}
    tgt = {obs.question_id: obs for obs in target}
    shared = sorted(set(base) & set(tgt))
    if not shared:
        return []

    bl = np.asarray([base[q].correct for q in shared], dtype=float)
    bs = np.asarray([base[q].confidence for q in shared], dtype=float)
    tl = np.asarray([tgt[q].correct for q in shared], dtype=float)
    ts = np.asarray([tgt[q].confidence for q in shared], dtype=float)
    n = len(shared)
    out: list[BootstrapRow] = []

    for metric_index, metric in enumerate(
        ("accuracy", "ece_paper", "ece_strict", "brier")
    ):
        baseline_value = metric_value(bl, bs, metric, n_bins)
        profile_value = metric_value(tl, ts, metric, n_bins)
        rng = np.random.default_rng(seed + metric_index)
        deltas = np.empty(replicates, dtype=float)
        for i in range(replicates):
            idx = rng.integers(0, n, size=n)
            deltas[i] = (
                metric_value(tl[idx], ts[idx], metric, n_bins)
                - metric_value(bl[idx], bs[idx], metric, n_bins)
            )
        low, high = np.percentile(deltas, [2.5, 97.5])
        if metric == "accuracy":
            p_improve = float(
                np.mean(deltas > 0) + 0.5 * np.mean(deltas == 0)
            )
        else:
            p_improve = float(
                np.mean(deltas < 0) + 0.5 * np.mean(deltas == 0)
            )
        out.append(
            BootstrapRow(
                dataset=dataset,
                profile=profile,
                comparison=comparison,
                metric=metric,
                n=n,
                baseline_value=baseline_value,
                profile_value=profile_value,
                delta_profile_minus_baseline=profile_value - baseline_value,
                ci_low_95=float(low),
                ci_high_95=float(high),
                probability_improvement=p_improve,
                bootstrap_replicates=replicates,
            )
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input_root)
    baseline_root = resolve(args.baseline_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[MetricRow] = []
    boot_rows: list[BootstrapRow] = []
    detail_rows: list[dict[str, Any]] = []

    print("Pure RelSC numeric relation-profile ablation")
    print("==========================================")
    print(
        "Each subset is compared both with SSC-32I and with an identity-only "
        "SSC baseline using the same per-question sample budget.\n"
    )

    for dataset in args.datasets:
        baseline_index = _index_payloads(_load_payloads(baseline_root, dataset))
        relation_index = _index_payloads(_load_payloads(input_root, dataset))
        shared = sorted(set(baseline_index) & set(relation_index))
        if not shared:
            raise ValueError(f"{dataset}: no shared questions")

        baseline32_obs: list[Observation] = []
        for qid in shared:
            obs = _ssc_observation(
                dataset,
                baseline_index[qid],
                scope="all",
                invalid_policy="paper",
            )
            if obs is not None:
                baseline32_obs.append(obs)

        print(f"[{dataset}] questions={len(shared)}")
        dataset_metrics: list[MetricRow] = []

        for profile, subtypes in PROFILE_SUBTYPES.items():
            observations: list[Observation] = []
            matched_baseline_obs: list[Observation] = []
            selected_counts: list[int] = []
            valid_counts: list[int] = []

            for qid in shared:
                obs, selected_n, valid_n = profile_observation(
                    dataset,
                    relation_index[qid],
                    profile,
                    subtypes,
                )
                if obs is None:
                    continue
                matched = matched_ssc_observation(
                    dataset,
                    baseline_index[qid],
                    selected_n,
                    f"SSC-matched::{profile}",
                )
                if matched is None:
                    continue

                observations.append(obs)
                matched_baseline_obs.append(matched)
                selected_counts.append(selected_n)
                valid_counts.append(valid_n)
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "question_id": qid,
                        "profile": profile,
                        "selected_samples": selected_n,
                        "valid_samples": valid_n,
                        "profile_prediction": obs.predicted_answer,
                        "profile_confidence": obs.confidence,
                        "profile_correct": obs.correct,
                        "matched_ssc_prediction": matched.predicted_answer,
                        "matched_ssc_confidence": matched.confidence,
                        "matched_ssc_correct": matched.correct,
                    }
                )

            row = metric_row(
                dataset,
                profile,
                observations,
                selected_counts,
                valid_counts,
                baseline32_obs,
                matched_baseline_obs,
                args.n_bins,
            )
            metric_rows.append(row)
            dataset_metrics.append(row)

            boot_rows.extend(
                bootstrap_rows(
                    dataset,
                    profile,
                    "SSC-32I -> profile",
                    baseline32_obs,
                    observations,
                    args.n_bins,
                    args.bootstrap_replicates,
                    args.seed,
                )
            )
            boot_rows.extend(
                bootstrap_rows(
                    dataset,
                    profile,
                    "SSC-matched -> profile",
                    matched_baseline_obs,
                    observations,
                    args.n_bins,
                    args.bootstrap_replicates,
                    args.seed + 100,
                )
            )

        ranked = sorted(
            dataset_metrics,
            key=lambda r: (
                r.delta_ece_vs_ssc_matched,
                r.delta_brier_vs_ssc_matched,
                -r.delta_accuracy_vs_ssc_matched,
            ),
        )
        print("  Best profiles vs same-budget identity SSC:")
        for row in ranked[:8]:
            print(
                f"    {row.method:<24s} n~{row.mean_selected_samples:5.1f} "
                f"RelSC acc={row.accuracy:6.2f} ECE={row.ece_paper:6.2f} "
                f"Brier={row.brier:6.3f} | "
                f"SSC-k acc={row.ssc_matched_accuracy:6.2f} "
                f"ECE={row.ssc_matched_ece_paper:6.2f} "
                f"Brier={row.ssc_matched_brier:6.3f} | "
                f"dAcc={row.delta_accuracy_vs_ssc_matched:+6.2f} "
                f"dECE={row.delta_ece_vs_ssc_matched:+6.2f} "
                f"dBrier={row.delta_brier_vs_ssc_matched:+6.3f}"
            )
        print()

    write_csv(
        output_dir / "numeric_relsc_profile_metrics.csv",
        [asdict(r) for r in metric_rows],
    )
    write_csv(
        output_dir / "numeric_relsc_profile_bootstrap.csv",
        [asdict(r) for r in boot_rows],
    )
    write_csv(
        output_dir / "numeric_relsc_profile_questions.csv",
        detail_rows,
    )

    print("Outputs:")
    print(f"  {output_dir / 'numeric_relsc_profile_metrics.csv'}")
    print(f"  {output_dir / 'numeric_relsc_profile_bootstrap.csv'}")
    print(f"  {output_dir / 'numeric_relsc_profile_questions.csv'}")


if __name__ == "__main__":
    main()
