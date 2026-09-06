"""CPU-only reliability-weight sweep for already-generated numeric RelaCaTS data.

This script does NOT generate responses. It reuses the existing 32-response
numeric-metamorphic pools and evaluates the relation-weighted form

    sum_{g,i} r_g c_{g,i} 1[a_{g,i}=a] / sum_{g,i} r_g c_{g,i}.

It is a diagnostic / development utility. Do not select weights on the same
questions that are reported as final test results. A clean protocol is to pick
weights on a development model/split, freeze them, and evaluate held-out
models/splits.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from relacats_v2.core import compute_relssc
from relacats_v2.evaluation.reproduce_table1 import (
    Observation,
    _correct,
    _index_payloads,
    _load_payloads,
    _metric_row,
    _normalize_answer,
    _ssc_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--datasets", nargs="+", default=("gsm8k", "svamp"))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--identity-weights", nargs="+", type=float, default=(1.0, 1.5, 2.0, 3.0))
    p.add_argument("--layout-weights", nargs="+", type=float, default=(1.0,))
    p.add_argument("--number-weights", nargs="+", type=float, default=(1.0, 0.75, 0.5))
    p.add_argument("--equivalent-weights", nargs="+", type=float, default=(1.0, 0.75, 0.5, 0.25, 0.0))
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


def weighted_observation(
    dataset: str,
    payload: Mapping[str, Any],
    weights: Mapping[str, float],
    method: str,
) -> Observation | None:
    records: list[dict[str, Any]] = []
    for raw in payload.get("samples", []):
        if not isinstance(raw, Mapping):
            continue
        record = dict(raw)
        record["relation_weight"] = float(weights.get(relation_subtype(record), 1.0))
        record["dependency_weight"] = 1.0
        records.append(record)
    result = compute_relssc(
        records,
        zero_weight_policy="skip",
        enforce_v1_weights=False,
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


def main() -> None:
    args = parse_args()
    for values in (
        args.identity_weights,
        args.layout_weights,
        args.number_weights,
        args.equivalent_weights,
    ):
        if any(value < 0 for value in values):
            raise ValueError("relation weights must be non-negative")

    input_root = resolve(args.input_root)
    baseline_root = resolve(args.baseline_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(
        args.identity_weights,
        args.layout_weights,
        args.number_weights,
        args.equivalent_weights,
    ))

    rows: list[dict[str, Any]] = []
    print("Numeric RelaCaTS relation-weight sweep")
    print("======================================")
    print(f"Configurations: {len(configs)}")
    print("Diagnostic only: freeze weights before final held-out evaluation.\n")

    for dataset in args.datasets:
        baseline = _index_payloads(_load_payloads(baseline_root, dataset))
        relational = _index_payloads(_load_payloads(input_root, dataset))
        shared = sorted(set(baseline) & set(relational))
        if not shared:
            raise ValueError(f"{dataset}: no shared questions")

        baseline_obs: list[Observation] = []
        for qid in shared:
            obs = _ssc_observation(
                dataset,
                baseline[qid],
                scope="all",
                invalid_policy="paper",
            )
            if obs is not None:
                baseline_obs.append(obs)
        baseline_metric = _metric_row(dataset, "SSC-32I", baseline_obs, args.n_bins)
        print(
            f"[{dataset}] SSC-32I: acc={baseline_metric.accuracy:.2f} "
            f"ECE={baseline_metric.ece_paper:.2f} Brier={baseline_metric.brier:.4f}"
        )

        for identity_w, layout_w, number_w, equivalent_w in configs:
            weights = {
                "identity": identity_w,
                "layout_wrapper": layout_w,
                "number_representation": number_w,
                "equivalent_quantity": equivalent_w,
            }
            method = (
                f"rI={identity_w:g},rL={layout_w:g},"
                f"rN={number_w:g},rE={equivalent_w:g}"
            )
            observations: list[Observation] = []
            for qid in shared:
                obs = weighted_observation(
                    dataset, relational[qid], weights, method
                )
                if obs is not None:
                    observations.append(obs)
            metric = _metric_row(dataset, method, observations, args.n_bins)
            row = asdict(metric)
            row.update({
                "r_identity": identity_w,
                "r_layout": layout_w,
                "r_number": number_w,
                "r_equivalent": equivalent_w,
                "delta_ece_vs_ssc32i": metric.ece_paper - baseline_metric.ece_paper,
                "delta_accuracy_vs_ssc32i": metric.accuracy - baseline_metric.accuracy,
                "delta_brier_vs_ssc32i": metric.brier - baseline_metric.brier,
            })
            rows.append(row)

        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        by_ece = sorted(dataset_rows, key=lambda row: (row["ece_paper"], -row["accuracy"]))[:8]
        print("  Lowest-ECE diagnostic configurations:")
        for row in by_ece:
            print(
                "   "
                f"I={row['r_identity']:g} L={row['r_layout']:g} "
                f"N={row['r_number']:g} E={row['r_equivalent']:g}  "
                f"acc={row['accuracy']:.2f} ECE={row['ece_paper']:.2f} "
                f"Brier={row['brier']:.4f} dECE={row['delta_ece_vs_ssc32i']:+.2f}"
            )
        print()

    output_path = output_dir / "numeric_relation_weight_sweep.csv"
    if rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
