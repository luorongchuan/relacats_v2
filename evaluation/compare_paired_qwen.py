"""Compare paired Qwen SSC-vs-RelSC training/evaluation runs.

The two runs share training rows and sampling protocol; only the calibration
pseudo-label differs.  This report combines:

1. CaTS/Table-2 style accuracy at a requested average budget (default 16), and
2. per-response confidence quality (accuracy, ECE, Brier, AUROC)

for the SSC-trained baseline and RelSC-trained model.

CISC, Self-Certainty and RASC may be proxy implementations when their native
auxiliary scores are absent from the confidence artifacts; the underlying
paper-budget report records that status.  This script never upgrades a proxy
into an exact baseline claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from relacats_v2.common import atomic_write_json, read_json, read_jsonl

DEFAULT_DATASETS = ("object_counting", "math_qa", "arc_challenge")
CALIBRATED_METHODS = {
    "RelaCaTS-SC": ("CaTS-SC", "RelaCaTS-SC"),
    "RelaCaTS-ES": ("CaTS-ES", "RelaCaTS-ES"),
    "RelaCaTS-ASC": ("CaTS-ASC", "RelaCaTS-ASC"),
}
TABLE_ORDER = (
    "SC",
    "CISC",
    "Self-Certainty",
    "RelaCaTS-SC",
    "Best-of-N",
    "RelaCaTS-ES",
    "ASC",
    "RelaCaTS-ASC",
    "ESC",
    "RASC",
)


def _finite_probability(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """AUROC via average ranks, including exact tie handling."""
    if len(labels) != len(scores) or not labels:
        return None
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end

    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


def _ece(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> float:
    if not labels:
        return float("nan")
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        lo = bin_index / bins
        hi = (bin_index + 1) / bins
        members = [
            index
            for index, score in enumerate(scores)
            if (lo <= score < hi) or (bin_index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = sum(scores[index] for index in members) / len(members)
        accuracy = sum(labels[index] for index in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def confidence_metrics(paths: Iterable[Path]) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    for path in paths:
        for record in read_jsonl(path):
            score = _finite_probability(record.get("confidence"))
            if score is None or "is_correct" not in record:
                continue
            labels.append(1 if bool(record["is_correct"]) else 0)
            scores.append(score)
    if not labels:
        raise ValueError("No usable confidence records found")
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)
    auc = _auc_binary(labels, scores)
    return {
        "n": len(labels),
        "response_accuracy_percent": 100.0 * sum(labels) / len(labels),
        "ece_percent": 100.0 * _ece(labels, scores),
        "brier_percent": 100.0 * brier,
        "auroc": auc,
    }


def _confidence_files(eval_root: Path, tag: str, dataset: str) -> list[Path]:
    root = eval_root / tag / "confidence" / "test" / dataset
    files = sorted(root.glob("shard-*-of-*/confidence.jsonl"))
    if not files:
        raise FileNotFoundError(f"No confidence artifacts under {root}")
    return files


def _report_path(eval_root: Path, tag: str, dataset: str) -> Path:
    return eval_root / tag / "results" / "test" / dataset / "evaluation.json"


def _table_rows(report: Mapping[str, Any], budget: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.get("fixed_budget_results", []):
        if int(row.get("budget", -1)) == budget:
            rows[str(row["method"])] = dict(row)
    for row in report.get("dynamic_budget_matches", []):
        target = row.get("budget_target", row.get("budget", -1))
        if int(target) == budget:
            rows[str(row["method"])] = dict(row)
    return rows


def _accuracy(row: Mapping[str, Any]) -> float:
    if "accuracy_percent" in row:
        return float(row["accuracy_percent"])
    return 100.0 * float(row["accuracy"])


def _avg_samples(row: Mapping[str, Any]) -> float:
    return float(row.get("actual_avg_samples", row.get("budget", float("nan"))))


def _method_status(report: Mapping[str, Any], method: str) -> str:
    metadata = report.get("method_metadata", {})
    entry = metadata.get(method, {}) if isinstance(metadata, Mapping) else {}
    if isinstance(entry, Mapping):
        return str(entry.get("implementation_status", "unspecified"))
    return "unspecified"


def build_comparison(
    eval_root: Path,
    *,
    ssc_tag: str,
    relsc_tag: str,
    datasets: Sequence[str],
    budget: int,
) -> dict[str, Any]:
    table_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        ssc_report = read_json(_report_path(eval_root, ssc_tag, dataset))
        rel_report = read_json(_report_path(eval_root, relsc_tag, dataset))
        ssc_methods = _table_rows(ssc_report, budget)
        rel_methods = _table_rows(rel_report, budget)

        for method in TABLE_ORDER:
            if method not in ssc_methods or method not in rel_methods:
                continue
            ssc_row = ssc_methods[method]
            rel_row = rel_methods[method]
            baseline_name, relsc_name = CALIBRATED_METHODS.get(method, (method, method))
            ssc_acc = _accuracy(ssc_row)
            rel_acc = _accuracy(rel_row)
            table_rows.append(
                {
                    "dataset": dataset,
                    "comparison_method": method,
                    "ssc_display_name": baseline_name,
                    "relsc_display_name": relsc_name,
                    "ssc_accuracy_percent": ssc_acc,
                    "relsc_accuracy_percent": rel_acc,
                    "delta_accuracy_pp": rel_acc - ssc_acc,
                    "ssc_actual_avg_samples": _avg_samples(ssc_row),
                    "relsc_actual_avg_samples": _avg_samples(rel_row),
                    "ssc_implementation_status": _method_status(ssc_report, method),
                    "relsc_implementation_status": _method_status(rel_report, method),
                }
            )

        ssc_metrics = confidence_metrics(_confidence_files(eval_root, ssc_tag, dataset))
        rel_metrics = confidence_metrics(_confidence_files(eval_root, relsc_tag, dataset))
        calibration_rows.append(
            {
                "dataset": dataset,
                "ssc": ssc_metrics,
                "relsc": rel_metrics,
                "delta_response_accuracy_pp": (
                    rel_metrics["response_accuracy_percent"]
                    - ssc_metrics["response_accuracy_percent"]
                ),
                "delta_ece_pp": rel_metrics["ece_percent"] - ssc_metrics["ece_percent"],
                "delta_brier_pp": (
                    rel_metrics["brier_percent"] - ssc_metrics["brier_percent"]
                ),
                "delta_auroc": (
                    None
                    if ssc_metrics["auroc"] is None or rel_metrics["auroc"] is None
                    else rel_metrics["auroc"] - ssc_metrics["auroc"]
                ),
            }
        )

    return {
        "schema_version": "relacats-v2.paired-qwen-comparison.1",
        "eval_root": str(eval_root),
        "ssc_tag": ssc_tag,
        "relsc_tag": relsc_tag,
        "budget": budget,
        "datasets": list(datasets),
        "table2_rows": table_rows,
        "confidence_quality": calibration_rows,
        "note": (
            "Negative delta_ece_pp/delta_brier_pp means RelSC is better. "
            "Positive delta_accuracy_pp/delta_auroc means RelSC is better. "
            "CISC/Self-Certainty/RASC may be proxy rows when native scores are absent."
        ),
    }


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "paired_comparison.json", report)

    with (output_dir / "table2_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = list(report["table2_rows"])
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "confidence_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "ssc_n",
            "relsc_n",
            "ssc_response_accuracy_percent",
            "relsc_response_accuracy_percent",
            "delta_response_accuracy_pp",
            "ssc_ece_percent",
            "relsc_ece_percent",
            "delta_ece_pp",
            "ssc_brier_percent",
            "relsc_brier_percent",
            "delta_brier_pp",
            "ssc_auroc",
            "relsc_auroc",
            "delta_auroc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in report["confidence_quality"]:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "ssc_n": item["ssc"]["n"],
                    "relsc_n": item["relsc"]["n"],
                    "ssc_response_accuracy_percent": item["ssc"]["response_accuracy_percent"],
                    "relsc_response_accuracy_percent": item["relsc"]["response_accuracy_percent"],
                    "delta_response_accuracy_pp": item["delta_response_accuracy_pp"],
                    "ssc_ece_percent": item["ssc"]["ece_percent"],
                    "relsc_ece_percent": item["relsc"]["ece_percent"],
                    "delta_ece_pp": item["delta_ece_pp"],
                    "ssc_brier_percent": item["ssc"]["brier_percent"],
                    "relsc_brier_percent": item["relsc"]["brier_percent"],
                    "delta_brier_pp": item["delta_brier_pp"],
                    "ssc_auroc": item["ssc"]["auroc"],
                    "relsc_auroc": item["relsc"]["auroc"],
                    "delta_auroc": item["delta_auroc"],
                }
            )

    lines = [
        "# Paired Qwen SSC vs RelSC",
        "",
        f"Average sample budget: **{report['budget']}**",
        "",
        "## Table-2 style accuracy",
        "",
        "| Dataset | SSC-side method | RelSC-side method | SSC Acc. | RelSC Acc. | Δ pp | SSC avg N | RelSC avg N |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["table2_rows"]:
        lines.append(
            f"| {row['dataset']} | {row['ssc_display_name']} | {row['relsc_display_name']} | "
            f"{row['ssc_accuracy_percent']:.2f} | {row['relsc_accuracy_percent']:.2f} | "
            f"{row['delta_accuracy_pp']:+.2f} | {row['ssc_actual_avg_samples']:.3f} | "
            f"{row['relsc_actual_avg_samples']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-response confidence quality",
            "",
            "Lower ECE/Brier is better; higher AUROC is better.",
            "",
            "| Dataset | SSC ECE | RelSC ECE | Δ ECE | SSC Brier | RelSC Brier | Δ Brier | SSC AUROC | RelSC AUROC | Δ AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["confidence_quality"]:
        ssc_auc = item["ssc"]["auroc"]
        rel_auc = item["relsc"]["auroc"]
        delta_auc = item["delta_auroc"]
        lines.append(
            f"| {item['dataset']} | {item['ssc']['ece_percent']:.2f} | {item['relsc']['ece_percent']:.2f} | "
            f"{item['delta_ece_pp']:+.2f} | {item['ssc']['brier_percent']:.2f} | "
            f"{item['relsc']['brier_percent']:.2f} | {item['delta_brier_pp']:+.2f} | "
            f"{ssc_auc:.4f} | {rel_auc:.4f} | {delta_auc:+.4f} |"
            if ssc_auc is not None and rel_auc is not None and delta_auc is not None
            else f"| {item['dataset']} | {item['ssc']['ece_percent']:.2f} | {item['relsc']['ece_percent']:.2f} | "
            f"{item['delta_ece_pp']:+.2f} | {item['ssc']['brier_percent']:.2f} | "
            f"{item['relsc']['brier_percent']:.2f} | {item['delta_brier_pp']:+.2f} | n/a | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "> CISC, Self-Certainty and RASC are reportable as exact only when their native auxiliary scores are present. Otherwise the evaluator marks them as proxies.",
            "",
        ]
    )
    (output_dir / "paired_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--ssc-tag", default="qwen_paired_ssc")
    parser.add_argument("--relsc-tag", default="qwen_paired_relsc")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_comparison(
        Path(args.eval_root).expanduser().resolve(),
        ssc_tag=args.ssc_tag,
        relsc_tag=args.relsc_tag,
        datasets=tuple(args.datasets),
        budget=args.budget,
    )
    output = Path(args.output_dir).expanduser().resolve()
    write_outputs(report, output)
    print(f"Wrote paired comparison to {output}")
    print(output / "paired_comparison.md")


if __name__ == "__main__":
    main()
