"""CPU-only decomposition of numeric RelSC gains on GSM8K/SVAMP.

This diagnostic separates three effects that are confounded when a count-based
RelSC profile is compared directly with CaTS SSC:

1. sampling/relation effect: RelSC(profile) vs same-budget identity SC-valid;
2. confidence-weight effect: identity SC-valid vs identity SSC-valid;
3. invalid-policy effect: identity SSC-valid vs identity SSC-paper.

All baselines use the same per-question number of responses as the relation
profile.  Existing generated JSONs are reused; no model generation or GPU is
required.  Subset profiles remain diagnostic because they reuse stored views
rather than redistributing a fresh N=32 generation budget.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from relacats_v2.evaluation.numeric_relsc_profile_ablation import (
    PROFILE_SUBTYPES,
    profile_observation,
)
from relacats_v2.evaluation.reproduce_table1 import (
    Observation,
    _filtered_samples,
    _index_payloads,
    _load_payloads,
    _sc_observation,
    _ssc_observation,
    calculate_ece_paper,
    calculate_ece_strict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DecompositionRow:
    dataset: str
    profile: str
    method: str
    n: int
    mean_budget: float
    accuracy: float
    ece_paper: float
    ece_strict: float
    brier: float


@dataclass(frozen=True)
class DeltaRow:
    dataset: str
    profile: str
    comparison: str
    metric: str
    n: int
    reference_value: float
    target_value: float
    delta_target_minus_reference: float
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


def metric_value(labels: np.ndarray, scores: np.ndarray, metric: str, n_bins: int) -> float:
    if metric == "accuracy":
        return 100.0 * float(np.mean(labels))
    if metric == "brier":
        return 100.0 * float(np.mean((scores - labels) ** 2))
    labels_l = labels.tolist()
    scores_l = scores.tolist()
    if metric == "ece_paper":
        return 100.0 * calculate_ece_paper(labels_l, scores_l, n_bins=n_bins)
    if metric == "ece_strict":
        return 100.0 * calculate_ece_strict(labels_l, scores_l, n_bins=n_bins)
    raise ValueError(metric)


def metric_tuple(observations: Sequence[Observation], ids: Sequence[str], n_bins: int) -> tuple[float, float, float, float]:
    by_id = {obs.question_id: obs for obs in observations}
    labels = np.asarray([by_id[q].correct for q in ids], dtype=float)
    scores = np.asarray([by_id[q].confidence for q in ids], dtype=float)
    return (
        metric_value(labels, scores, "accuracy", n_bins),
        metric_value(labels, scores, "ece_paper", n_bins),
        metric_value(labels, scores, "ece_strict", n_bins),
        metric_value(labels, scores, "brier", n_bins),
    )


def matched_identity_observation(
    dataset: str,
    payload: Mapping[str, object],
    budget: int,
    *,
    method: str,
    weighted: bool,
    invalid_policy: str,
) -> Observation | None:
    if budget <= 0:
        return None
    samples = _filtered_samples(payload, scope="all")
    if not samples:
        return None
    sliced = dict(payload)
    sliced["samples"] = samples[: min(budget, len(samples))]
    if weighted:
        obs = _ssc_observation(dataset, sliced, scope="all", invalid_policy=invalid_policy)
    else:
        obs = _sc_observation(dataset, sliced, scope="all", invalid_policy=invalid_policy)
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


def bootstrap_pair(
    dataset: str,
    profile: str,
    comparison: str,
    reference: Sequence[Observation],
    target: Sequence[Observation],
    n_bins: int,
    replicates: int,
    seed: int,
) -> list[DeltaRow]:
    ref = {obs.question_id: obs for obs in reference}
    tgt = {obs.question_id: obs for obs in target}
    ids = sorted(set(ref) & set(tgt))
    if not ids:
        return []
    rl = np.asarray([ref[q].correct for q in ids], dtype=float)
    rs = np.asarray([ref[q].confidence for q in ids], dtype=float)
    tl = np.asarray([tgt[q].correct for q in ids], dtype=float)
    ts = np.asarray([tgt[q].confidence for q in ids], dtype=float)
    n = len(ids)
    rows: list[DeltaRow] = []
    for m_idx, metric in enumerate(("accuracy", "ece_paper", "ece_strict", "brier")):
        rv = metric_value(rl, rs, metric, n_bins)
        tv = metric_value(tl, ts, metric, n_bins)
        rng = np.random.default_rng(seed + m_idx)
        deltas = np.empty(replicates, dtype=float)
        for i in range(replicates):
            idx = rng.integers(0, n, size=n)
            deltas[i] = metric_value(tl[idx], ts[idx], metric, n_bins) - metric_value(rl[idx], rs[idx], metric, n_bins)
        low, high = np.percentile(deltas, [2.5, 97.5])
        if metric == "accuracy":
            p_improve = float(np.mean(deltas > 0) + 0.5 * np.mean(deltas == 0))
        else:
            p_improve = float(np.mean(deltas < 0) + 0.5 * np.mean(deltas == 0))
        rows.append(
            DeltaRow(
                dataset=dataset,
                profile=profile,
                comparison=comparison,
                metric=metric,
                n=n,
                reference_value=rv,
                target_value=tv,
                delta_target_minus_reference=tv - rv,
                ci_low_95=float(low),
                ci_high_95=float(high),
                probability_improvement=p_improve,
                bootstrap_replicates=replicates,
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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

    metric_rows: list[DecompositionRow] = []
    delta_rows: list[DeltaRow] = []

    print("Numeric RelSC decomposition: relation vs weighting vs invalid policy")
    print("=================================================================")
    print("Primary causal diagnostic: RelSC(profile) vs same-budget identity SC-valid.\n")

    for dataset in args.datasets:
        baseline = _index_payloads(_load_payloads(baseline_root, dataset))
        relation = _index_payloads(_load_payloads(input_root, dataset))
        shared = sorted(set(baseline) & set(relation))
        if not shared:
            raise ValueError(f"{dataset}: no shared questions")
        print(f"[{dataset}] questions={len(shared)}")

        profile_summaries: list[tuple[str, float, float, float, float]] = []
        for profile, subtypes in PROFILE_SUBTYPES.items():
            target: list[Observation] = []
            sc_valid: list[Observation] = []
            ssc_valid: list[Observation] = []
            ssc_paper: list[Observation] = []
            budgets: list[int] = []

            for qid in shared:
                obs, budget, _ = profile_observation(dataset, relation[qid], profile, subtypes)
                if obs is None or budget <= 0:
                    continue
                scv = matched_identity_observation(dataset, baseline[qid], budget, method=f"SC-k-valid::{profile}", weighted=False, invalid_policy="valid-only")
                ssv = matched_identity_observation(dataset, baseline[qid], budget, method=f"SSC-k-valid::{profile}", weighted=True, invalid_policy="valid-only")
                ssp = matched_identity_observation(dataset, baseline[qid], budget, method=f"SSC-k-paper::{profile}", weighted=True, invalid_policy="paper")
                if scv is None or ssv is None or ssp is None:
                    continue
                target.append(obs)
                sc_valid.append(scv)
                ssc_valid.append(ssv)
                ssc_paper.append(ssp)
                budgets.append(budget)

            maps = [
                {obs.question_id for obs in target},
                {obs.question_id for obs in sc_valid},
                {obs.question_id for obs in ssc_valid},
                {obs.question_id for obs in ssc_paper},
            ]
            ids = sorted(set.intersection(*maps)) if maps else []
            if not ids:
                continue
            mean_budget = float(np.mean(budgets))
            methods = [
                ("RelSC-profile", target),
                ("SC-k-valid", sc_valid),
                ("SSC-k-valid", ssc_valid),
                ("SSC-k-paper", ssc_paper),
            ]
            values: dict[str, tuple[float, float, float, float]] = {}
            for method, observations in methods:
                vals = metric_tuple(observations, ids, args.n_bins)
                values[method] = vals
                metric_rows.append(
                    DecompositionRow(
                        dataset=dataset,
                        profile=profile,
                        method=method,
                        n=len(ids),
                        mean_budget=mean_budget,
                        accuracy=vals[0],
                        ece_paper=vals[1],
                        ece_strict=vals[2],
                        brier=vals[3],
                    )
                )

            delta_rows.extend(bootstrap_pair(dataset, profile, "SC-k-valid -> RelSC-profile [relation effect]", sc_valid, target, args.n_bins, args.bootstrap_replicates, args.seed))
            delta_rows.extend(bootstrap_pair(dataset, profile, "SSC-k-valid -> SC-k-valid [remove confidence weighting]", ssc_valid, sc_valid, args.n_bins, args.bootstrap_replicates, args.seed))
            delta_rows.extend(bootstrap_pair(dataset, profile, "SSC-k-paper -> SSC-k-valid [invalid-policy effect]", ssc_paper, ssc_valid, args.n_bins, args.bootstrap_replicates, args.seed))

            rel = values["RelSC-profile"]
            sc = values["SC-k-valid"]
            profile_summaries.append((profile, mean_budget, rel[0] - sc[0], rel[1] - sc[1], rel[3] - sc[3]))

        profile_summaries.sort(key=lambda x: (x[3], x[4], -x[2]))
        print("  Relation effect only: RelSC(profile) - identity SC-k-valid")
        for profile, budget, dacc, dece, dbrier in profile_summaries[:8]:
            print(f"    {profile:<24s} n~{budget:5.1f} dAcc={dacc:+6.2f} dECE={dece:+6.2f} dBrier={dbrier:+7.3f}")
        print()

    write_csv(output_dir / "numeric_relsc_decomposition_metrics.csv", [asdict(r) for r in metric_rows])
    write_csv(output_dir / "numeric_relsc_decomposition_bootstrap.csv", [asdict(r) for r in delta_rows])
    print("Outputs:")
    print(f"  {output_dir / 'numeric_relsc_decomposition_metrics.csv'}")
    print(f"  {output_dir / 'numeric_relsc_decomposition_bootstrap.csv'}")


if __name__ == "__main__":
    main()
