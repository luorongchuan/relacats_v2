"""Build a strict paired target-ablation dataset for SSC versus RelSC.

Purpose
-------
This dataset is not the final budget-matched RelaCaTS training protocol.  It is
an isolation experiment for the question:

    with the same training rows, same sampling, same causal-LM examples,
    same optimizer settings, and same budgets, does replacing the calibration
    target SSC by RelSC improve the trained confidence model?

Construction
------------
* Training rows are always the valid identity-only N=32 response records.
* ``ssc_consistency`` is CaTS confidence-weighted SSC computed from those same
  identity-only responses.
* For the seven MCQ datasets, ``relsc_consistency`` is count-based RelSC
  computed from the separate fixed-budget relational pool and mapped back to
  canonical option space.
* For GSM8K/SVAMP, ``relsc_consistency == ssc_consistency`` because the final
  method deliberately falls back to SSC on numeric tasks.

Thus the two training runs can use exactly the same JSONL rows.  The relation
pool is used only to produce the alternative RelSC pseudo-label for MCQ rows.
Gold answers are never used in either target.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_json
from relacats_v2.data_creation.build_relssc_dataset import split_question_ids

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    parser.add_argument("--relation-root", required=True)
    parser.add_argument("--identity-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=TRAIN_DATASETS, default=list(TRAIN_DATASETS))
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _discover(root: Path, dataset: str) -> dict[str, Path]:
    dataset_dir = root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    result: dict[str, Path] = {}
    for path in sorted(dataset_dir.rglob("*.json")):
        if path.name.endswith(("metadata.json", "manifest.json", "stats.json")):
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or "question_id" not in payload or "samples" not in payload:
            continue
        qid = str(payload["question_id"])
        if qid in result:
            raise ValueError(f"{dataset}: duplicate question_id {qid!r}")
        result[qid] = path
    if not result:
        raise FileNotFoundError(f"No question JSON files found under {dataset_dir}")
    return result


def _valid(sample: Mapping[str, Any]) -> bool:
    return sample.get("is_valid_answer") is True and sample.get("canonicalized_answer") is not None


def weighted_ssc_scores(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Original CaTS SSC: P(True)-weighted answer mass over valid responses."""
    mass: dict[str, float] = defaultdict(float)
    total = 0.0
    for sample in samples:
        if not _valid(sample):
            continue
        try:
            confidence = float(sample["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < 0:
            continue
        answer = str(sample["canonicalized_answer"])
        mass[answer] += confidence
        total += confidence
    if total <= 0:
        return {}
    return {answer: value / total for answer, value in mass.items()}


def pure_relsc_scores(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Count-based RelSC over valid canonicalized relation-view responses."""
    counts = Counter(
        str(sample["canonicalized_answer"])
        for sample in samples
        if _valid(sample)
    )
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {answer: count / total for answer, count in counts.items()}


def _payload(path: Path, dataset: str, qid: str) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    if str(payload.get("question_id")) != qid:
        raise ValueError(f"question_id mismatch: {path}")
    if payload.get("dataset_name") not in {None, dataset}:
        raise ValueError(f"dataset mismatch: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples: {path}")
    return payload


def build_question_rows(
    identity_payload: dict[str, Any],
    relation_payload: dict[str, Any] | None,
    *,
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity_samples = identity_payload["samples"]
    ssc = weighted_ssc_scores(identity_samples)
    if not ssc:
        return [], {"defined": False, "reason": "undefined SSC"}

    if dataset in TRAIN_MCQ_DATASETS:
        if relation_payload is None:
            return [], {"defined": False, "reason": "missing relation payload"}
        relsc = pure_relsc_scores(relation_payload["samples"])
        if not relsc:
            return [], {"defined": False, "reason": "undefined RelSC"}
    else:
        relsc = dict(ssc)

    rows: list[dict[str, Any]] = []
    missing_relsc_answers = 0
    for sample in identity_samples:
        if not _valid(sample):
            continue
        answer = str(sample["canonicalized_answer"])
        if answer not in ssc:
            continue
        rel_target = float(relsc.get(answer, 0.0))
        if answer not in relsc:
            missing_relsc_answers += 1
        prompt = sample.get(
            "transformed_prompt",
            sample.get("original_prompt", sample.get("transformed_question", "")),
        )
        response = sample.get("response", "")
        if not prompt or not response:
            continue
        row = dict(sample)
        row.update(
            {
                "transformed_prompt": prompt,
                "response": response,
                "input": f"{prompt}{response}",
                "answer": sample["canonicalized_answer"],
                "ssc_consistency": float(ssc[answer]),
                "relsc_consistency": rel_target,
                "paired_selection_consistency": float(ssc[answer]),
                "target_method_ssc": "ssc_32i_ptrue_weighted",
                "target_method_relsc": (
                    "relsc_fixed_budget_relation_pool"
                    if dataset in TRAIN_MCQ_DATASETS
                    else "ssc_fallback_numeric"
                ),
                "gold_used_in_target": False,
            }
        )
        rows.append(row)

    return rows, {
        "defined": bool(rows),
        "ssc_scores": ssc,
        "relsc_scores": relsc,
        "identity_valid_rows": len(rows),
        "identity_answers_missing_from_relsc": missing_relsc_answers,
        "gold_used_in_target": False,
    }


def main() -> None:
    args = parse_args()
    relation_root = resolve_path(args.relation_root)
    identity_root = resolve_path(args.identity_root)
    output_root = resolve_path(args.output_root)

    manifest_datasets: list[dict[str, Any]] = []

    for dataset in args.datasets:
        identity_files = _discover(identity_root, dataset)
        relation_files = _discover(relation_root, dataset) if dataset in TRAIN_MCQ_DATASETS else {}

        qids = sorted(identity_files)
        if dataset in TRAIN_MCQ_DATASETS:
            missing = sorted(set(qids) - set(relation_files))
            if missing:
                raise ValueError(
                    f"{dataset}: {len(missing)} identity questions have no relation counterpart; "
                    f"first={missing[0]!r}"
                )

        rows_by_qid: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for qid in qids:
            identity_payload = _payload(identity_files[qid], dataset, qid)
            relation_payload = (
                _payload(relation_files[qid], dataset, qid)
                if dataset in TRAIN_MCQ_DATASETS
                else None
            )
            rows, summary = build_question_rows(
                identity_payload, relation_payload, dataset=dataset
            )
            summary = dict(summary)
            summary.update({"question_id": qid, "dataset_name": dataset})
            summaries[qid] = summary
            if rows:
                rows_by_qid[qid] = rows

        train_ids, test_ids = split_question_ids(
            list(rows_by_qid), test_ratio=args.test_ratio, seed=args.seed
        )
        train_rows = [row for qid in sorted(train_ids) for row in rows_by_qid[qid]]
        test_rows = [row for qid in sorted(test_ids) for row in rows_by_qid[qid]]

        dataset_dir = output_root / dataset
        atomic_write_jsonl(dataset_dir / "train.jsonl", train_rows)
        atomic_write_jsonl(dataset_dir / "test.jsonl", test_rows)
        atomic_write_json(
            dataset_dir / "question_summaries.json",
            [summaries[qid] for qid in sorted(summaries)],
        )

        differing = sum(
            abs(float(row["ssc_consistency"]) - float(row["relsc_consistency"])) > 1e-12
            for row in train_rows + test_rows
        )
        stats = {
            "dataset_name": dataset,
            "source_questions": len(qids),
            "defined_questions": len(rows_by_qid),
            "train_records": len(train_rows),
            "test_records": len(test_rows),
            "records_with_different_targets": differing,
            "target_fields": ["ssc_consistency", "relsc_consistency"],
            "selection_field": "paired_selection_consistency",
            "training_rows_source": "identity_only_n32",
            "gold_used_in_target": False,
        }
        atomic_write_json(dataset_dir / "stats.json", stats)
        manifest_datasets.append(stats)
        print(
            f"{dataset}: questions={stats['defined_questions']} "
            f"train/test={stats['train_records']}/{stats['test_records']} "
            f"different-target-rows={differing}"
        )

    manifest = {
        "schema_version": "relacats-v2.paired-ssc-relsc.1",
        "identity_root": str(identity_root),
        "relation_root": str(relation_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "training_rows_source": "identity_only_n32",
        "ssc_target": "P(True)-weighted SSC from identity-only N=32 pool",
        "relsc_target": "count-based RelSC from fixed-budget relation pool for MCQ; SSC fallback for numeric",
        "paired_selection_field": "paired_selection_consistency",
        "pairing_claim": "same rows and selection; only calibration target differs",
        "gold_used_in_target": False,
        "datasets": manifest_datasets,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(f"Wrote paired manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
