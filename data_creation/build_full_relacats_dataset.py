"""Build the full RelaCaTS offline pseudo-label dataset.

This is the theory-faithful companion to ``build_relssc_dataset.py``.  It keeps
all of that builder's strict payload validation and split logic, while adding
Eq. (19) dependency correction, Eq. (20) weighted RelSSC, and Eq. (24)
consensus-fragility labels.  Gold answers remain diagnostics only.

The output format is deliberately compatible with the existing trainer fields
(``ssc``, ``relssc``, ``relation_valid_ratio``), and additionally contains:

* ``dependency_cluster_id`` / ``dependency_cluster_size``;
* ``dependency_weight``;
* ``fragility_target`` / ``consensus_fragility``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_json
from relacats_v2.core import attach_full_targets, attach_v2_target_inputs
from relacats_v2.data_creation import build_relssc_dataset as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "relacats_v2/outputs/generated_data"
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v2/outputs/full_relacats_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--strategy-similarity-threshold", type=float, default=0.86)
    parser.add_argument("--lambda-v", type=float, default=0.5)
    parser.add_argument("--allow-nonstandard-budget", action="store_true")
    return parser.parse_args()


def _flatten_full(
    payload: dict[str, Any],
    *,
    beta: float,
    similarity_threshold: float,
    lambda_v: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = payload["samples"]
    attached, relssc, fragility, dependency = attach_full_targets(
        samples,
        beta=beta,
        similarity_threshold=similarity_threshold,
        lambda_v=lambda_v,
    )
    attached, ssc_context = attach_v2_target_inputs(attached)

    if not relssc.defined or not ssc_context.defined:
        reason = relssc.reason if not relssc.defined else (
            "no valid identity-view answer remains; skip this question"
        )
        return [], {
            "question_id": payload["question_id"],
            "dataset_name": payload.get("dataset_name"),
            "defined": False,
            "reason": reason,
            "dependency_cluster_count": dependency.cluster_count,
            "dependency_effective_cluster_mass": dependency.effective_cluster_mass,
            "fragility_defined": fragility.defined,
            "gold_used_in_target": False,
        }

    rows: list[dict[str, Any]] = []
    for sample in attached:
        if sample.get("is_valid_answer") is not True:
            continue
        rel_target = sample.get("relssc")
        frag_target = sample.get("fragility_target")
        if rel_target is None or frag_target is None:
            continue
        prompt = sample.get(
            "transformed_prompt",
            sample.get("original_prompt", sample.get("transformed_question", "")),
        )
        row = dict(sample)
        row.update(
            {
                "input": f"{prompt}{sample.get('response', '')}",
                "answer": sample.get("canonicalized_answer"),
                "relssc": float(rel_target),
                "relational_consistency": float(rel_target),
                "fragility_target": float(frag_target),
                "consensus_fragility": float(frag_target),
                "question_relssc_scores": dict(relssc.scores),
                "question_relssc_total_weight": relssc.total_weight,
                "question_fragility_scores": dict(fragility.scores),
                "question_identity_support": dict(fragility.identity_support),
                "question_relational_support": dict(fragility.relational_support),
                "question_dependency_cluster_count": dependency.cluster_count,
                "question_dependency_effective_cluster_mass": (
                    dependency.effective_cluster_mass
                ),
                "attempted_budget": int(payload["attempted_budget"]),
                "target_provenance": "full_relacats_without_gold",
                "gold_used_in_target": False,
            }
        )
        rows.append(row)

    return rows, {
        "question_id": payload["question_id"],
        "dataset_name": payload.get("dataset_name"),
        "defined": True,
        "top_answer": relssc.top_answer,
        "relssc_scores": dict(relssc.scores),
        "fragility_scores": dict(fragility.scores),
        "identity_support": dict(fragility.identity_support),
        "relational_support": dict(fragility.relational_support),
        "dependency_cluster_count": dependency.cluster_count,
        "dependency_cluster_sizes": dict(dependency.cluster_sizes),
        "dependency_effective_cluster_mass": dependency.effective_cluster_mass,
        "valid_response_count": relssc.valid_sample_count,
        "invalid_response_count": relssc.invalid_sample_count,
        "ssc_scores": dict(ssc_context.scores),
        "valid_identity_sample_count": ssc_context.valid_identity_samples,
        "invalid_identity_sample_count": ssc_context.invalid_identity_samples,
        "valid_relation_sample_count": ssc_context.valid_relation_samples,
        "total_relation_sample_count": ssc_context.total_relation_samples,
        "relation_valid_ratio": ssc_context.relation_valid_ratio,
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
    beta: float,
    similarity_threshold: float,
    lambda_v: float,
) -> dict[str, Any]:
    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    raw_count = 0
    seen_ids: set[str] = set()

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
        samples = base.validate_question_payload(
            payload, allow_nonstandard_budget=allow_nonstandard_budget
        )
        raw_count += len(samples)
        rows, summary = _flatten_full(
            payload,
            beta=beta,
            similarity_threshold=similarity_threshold,
            lambda_v=lambda_v,
        )
        summaries.append(summary)
        if rows:
            rows_by_question[question_id] = rows

    train_ids, test_ids = base.split_question_ids(
        list(rows_by_question), test_ratio=test_ratio, seed=seed
    )
    dataset_dir = output_root / dataset_name
    train_rows = [
        row for qid in sorted(train_ids) for row in rows_by_question[qid]
    ]
    test_rows = [
        row for qid in sorted(test_ids) for row in rows_by_question[qid]
    ]
    atomic_write_jsonl(dataset_dir / "train.jsonl", train_rows)
    atomic_write_jsonl(dataset_dir / "test.jsonl", test_rows)
    atomic_write_json(
        dataset_dir / "question_summaries.json",
        sorted(summaries, key=lambda item: item["question_id"]),
    )

    return {
        "dataset_name": dataset_name,
        "raw_response_records": raw_count,
        "questions": len(summaries),
        "defined_questions": len(rows_by_question),
        "train_questions": len(train_ids),
        "test_questions": len(test_ids),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "beta": beta,
        "strategy_similarity_threshold": similarity_threshold,
        "lambda_v": lambda_v,
    }


def main() -> None:
    args = parse_args()
    input_root = base.resolve_path(args.input_root)
    output_root = base.resolve_path(args.output_root)
    files_by_dataset = base.discover_question_files(input_root, args.datasets)

    dataset_summaries: list[dict[str, Any]] = []
    for dataset_name, paths in files_by_dataset.items():
        stats = build_dataset(
            dataset_name=dataset_name,
            files=paths,
            output_root=output_root,
            test_ratio=args.test_ratio,
            seed=args.seed,
            allow_nonstandard_budget=args.allow_nonstandard_budget,
            beta=args.beta,
            similarity_threshold=args.strategy_similarity_threshold,
            lambda_v=args.lambda_v,
        )
        dataset_summaries.append(stats)
        print(
            f"{dataset_name}: questions={stats['defined_questions']} "
            f"train/test rows={stats['train_records']}/{stats['test_records']}"
        )

    manifest = {
        "schema_version": "relacats-v2.full-dataset-manifest.1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "beta": args.beta,
        "strategy_similarity_threshold": args.strategy_similarity_threshold,
        "lambda_v": args.lambda_v,
        "dependency_rule": "explicit cluster id else deterministic lexical-Jaccard fallback",
        "relssc_weight": "relation_weight * dependency_weight * confidence",
        "fragility_equation": 24,
        "target_modes": ["ssc", "relssc_replace", "residual"],
        "default_target_mode": "residual",
        "default_lambda_rel": 0.5,
        "generation_filter_modes": ["ssc", "relssc_fragility"],
        "gold_used_in_target": False,
        "datasets": dataset_summaries,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(f"Wrote full RelaCaTS dataset manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
