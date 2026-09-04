"""Build Full RelaCaTS labels from existing v1 raw completions, strictly.

This is the recommended migration path for the already-generated 32-response
teacher pools.  It never edits the v1 JSON files.  For every stored response it
re-runs the RelaCaTS-v2 explicit-final-answer parser, re-canonicalizes the
answer through the saved relation mapping, drops invalid/ambiguous responses
from the training pool, and only then computes dependency correction, RelSSC,
and (when relation witnesses exist) fragility.

Identity-only numeric datasets such as GSM8K/SVAMP have no cross-relation
witness.  Their q/RelSSC labels remain usable, but fragility is marked
unavailable rather than incorrectly supervised as f*=0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_json
from relacats_v2.core import (
    OptionPermutation,
    attach_full_targets,
    attach_v2_target_inputs,
    canonicalize_answer,
)
from relacats_v2.data_creation import build_relssc_dataset as base
from relacats_v2.evaluation.answer_parsing import (
    STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
    extract_explicit_answer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "relacats_v1/outputs/generated_data/qwen2_5_7b_instruct"
DEFAULT_OUTPUT = (
    REPO_ROOT / "relacats_v2/outputs/full_relacats_dataset/qwen2_5_7b_instruct"
)


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


def _answer_type(payload: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    value = sample.get("answer_type", payload.get("answer_type", "option letter"))
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "number" if token in {"number", "numeric", "scalar"} else "option letter"


def _strict_reparse_sample(
    payload: Mapping[str, Any], sample: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copied sample whose answer fields come only from the v2 parser."""

    copied = dict(sample)
    answer_type = _answer_type(payload, sample)
    dataset_name = str(sample.get("dataset_name", payload.get("dataset_name", "")))
    extracted = extract_explicit_answer(
        dataset_name,
        str(sample.get("response", "")),
        answer_type=answer_type,
    )
    if answer_type == "number":
        result = canonicalize_answer(
            extracted,
            {"relation_type": "identity"},
            answer_type="number",
        )
    else:
        permutation = OptionPermutation.from_metadata(sample)
        result = canonicalize_answer(
            extracted,
            permutation,
            answer_type="option",
            labels=permutation.labels,
        )
    copied.update(result.to_record_fields())
    copied["answer_parser_version"] = STRICT_EXPLICIT_ANSWER_PARSER_VERSION
    copied["answer_reparsed_from_response"] = True
    return copied


def _strict_reparse_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError(f"{payload.get('question_id')}: samples must be a list")
    counters = {
        "old_valid": 0,
        "strict_valid": 0,
        "lost_by_strict": 0,
        "recovered_by_strict": 0,
        "canonical_changed": 0,
    }
    reparsed: list[dict[str, Any]] = []
    for sample in samples:
        old_valid = isinstance(sample, dict) and sample.get("is_valid_answer") is True
        if old_valid:
            counters["old_valid"] += 1
        copied = _strict_reparse_sample(payload, sample)
        strict_valid = copied.get("is_valid_answer") is True
        if strict_valid:
            counters["strict_valid"] += 1
        if old_valid and not strict_valid:
            counters["lost_by_strict"] += 1
        elif not old_valid and strict_valid:
            counters["recovered_by_strict"] += 1
        if old_valid and strict_valid:
            old_answer = sample.get("canonicalized_answer")
            new_answer = copied.get("canonicalized_answer")
            if str(old_answer).strip() != str(new_answer).strip():
                counters["canonical_changed"] += 1
        reparsed.append(copied)
    copied_payload = dict(payload)
    copied_payload["samples"] = reparsed
    copied_payload["strict_parser_version"] = STRICT_EXPLICIT_ANSWER_PARSER_VERSION
    copied_payload["strict_valid_response_count"] = counters["strict_valid"]
    copied_payload["strict_invalid_response_count"] = len(reparsed) - counters["strict_valid"]
    return copied_payload, counters


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
            "fragility_available": False,
            "gold_used_in_target": False,
        }

    # Eq. (24) is a cross-relation diagnostic.  With only one positive-mass
    # view, s_R == s_0 and the variance term is zero by construction; that is
    # "unobserved fragility", not evidence that fragility truly equals zero.
    fragility_available = bool(
        fragility.defined and len(fragility.view_supports) >= 2
    )

    rows: list[dict[str, Any]] = []
    for sample in attached:
        if sample.get("is_valid_answer") is not True:
            continue
        rel_target = sample.get("relssc")
        if rel_target is None:
            continue
        frag_target = sample.get("fragility_target") if fragility_available else None
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
                "fragility_available": frag_target is not None,
                "fragility_target": float(frag_target) if frag_target is not None else None,
                "consensus_fragility": float(frag_target) if frag_target is not None else None,
                "question_relssc_scores": dict(relssc.scores),
                "question_relssc_total_weight": relssc.total_weight,
                "question_fragility_scores": dict(fragility.scores) if fragility_available else {},
                "question_identity_support": dict(fragility.identity_support),
                "question_relational_support": (
                    dict(fragility.relational_support) if fragility_available else {}
                ),
                "question_dependency_cluster_count": dependency.cluster_count,
                "question_dependency_effective_cluster_mass": dependency.effective_cluster_mass,
                "attempted_budget": int(payload["attempted_budget"]),
                "target_provenance": "strict_full_relacats_without_gold",
                "gold_used_in_target": False,
                "answer_parser_version": STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
            }
        )
        rows.append(row)

    return rows, {
        "question_id": payload["question_id"],
        "dataset_name": payload.get("dataset_name"),
        "defined": True,
        "top_answer": relssc.top_answer,
        "relssc_scores": dict(relssc.scores),
        "fragility_available": fragility_available,
        "fragility_scores": dict(fragility.scores) if fragility_available else {},
        "identity_support": dict(fragility.identity_support),
        "relational_support": dict(fragility.relational_support) if fragility_available else {},
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
    seen_ids: set[str] = set()
    raw_count = 0
    audit_totals = {
        "old_valid": 0,
        "strict_valid": 0,
        "lost_by_strict": 0,
        "recovered_by_strict": 0,
        "canonical_changed": 0,
    }

    for path in files:
        original = read_json(path)
        if not isinstance(original, dict):
            raise ValueError(f"Expected JSON object at {path}")
        if original.get("dataset_name") != dataset_name:
            raise ValueError(f"Dataset directory/payload mismatch at {path}")
        question_id = str(original["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id {question_id!r}")
        seen_ids.add(question_id)

        # Validate the immutable v1 artifact structure first; then replace only
        # answer-derived fields in memory using the strict v2 protocol.
        validated = base.validate_question_payload(
            original, allow_nonstandard_budget=allow_nonstandard_budget
        )
        raw_count += len(validated)
        payload, audit = _strict_reparse_payload(original)
        for key in audit_totals:
            audit_totals[key] += audit[key]

        rows, summary = _flatten_full(
            payload,
            beta=beta,
            similarity_threshold=similarity_threshold,
            lambda_v=lambda_v,
        )
        summary.update({f"strict_{key}": value for key, value in audit.items()})
        summaries.append(summary)
        if rows:
            rows_by_question[question_id] = rows

    train_ids, test_ids = base.split_question_ids(
        list(rows_by_question), test_ratio=test_ratio, seed=seed
    )
    dataset_dir = output_root / dataset_name
    train_rows = [row for qid in sorted(train_ids) for row in rows_by_question[qid]]
    test_rows = [row for qid in sorted(test_ids) for row in rows_by_question[qid]]
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
        "fragility_available_questions": sum(
            bool(item.get("fragility_available")) for item in summaries
        ),
        "beta": beta,
        "strategy_similarity_threshold": similarity_threshold,
        "lambda_v": lambda_v,
        **audit_totals,
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
            f"train/test={stats['train_records']}/{stats['test_records']} "
            f"strict_valid={stats['strict_valid']}/{stats['raw_response_records']} "
            f"frag_questions={stats['fragility_available_questions']}"
        )

    manifest = {
        "schema_version": "relacats-v2.strict-full-dataset-manifest.1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "strict_parser_version": STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "beta": args.beta,
        "strategy_similarity_threshold": args.strategy_similarity_threshold,
        "lambda_v": args.lambda_v,
        "dependency_rule": "explicit cluster id else deterministic lexical-Jaccard fallback",
        "relssc_weight": "relation_weight * dependency_weight * confidence",
        "identity_only_fragility_policy": "masked/unavailable",
        "invalid_policy": "raw retained; strict-invalid excluded from labels/training",
        "gold_used_in_target": False,
        "datasets": dataset_summaries,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(f"Wrote strict Full RelaCaTS manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
