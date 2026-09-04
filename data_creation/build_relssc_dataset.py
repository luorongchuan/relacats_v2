"""Build local, question-grouped RelaCaTS v2 training JSONL files.

Raw invalid generations are retained in ``generated_data`` for auditability,
but—as in the original CaTS dataset builder—only valid extracted answers enter
the training JSONL and the RelSSC denominator.  Gold answers are copied solely
for diagnostics and are never read by :func:`compute_relssc`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from relacats_v2.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    stable_id,
)
from relacats_v2.core import (
    OptionPermutation,
    attach_relssc_targets,
    attach_v2_target_inputs,
    canonicalize_answer,
    compute_relssc,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "relacats_v2/outputs/generated_data"
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v2/outputs/relssc_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-nonstandard-budget",
        action="store_true",
        help=(
            "Permit smoke budgets other than the formal profiles "
            "(ordinary MCQ 4x8, WinoGrande 2x16, numeric 1x32)."
        ),
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def discover_question_files(input_root: Path, datasets: Sequence[str] | None) -> dict[str, list[Path]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Generated-data root does not exist: {input_root}")
    selected = set(datasets or [])
    result: dict[str, list[Path]] = {}
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if selected and dataset_dir.name not in selected:
            continue
        paths = sorted((dataset_dir / "questions").glob("*.json"))
        if paths:
            result[dataset_dir.name] = paths
    missing = selected - set(result)
    if missing:
        raise FileNotFoundError(
            f"No generated question files for dataset(s): {sorted(missing)}"
        )
    if not result:
        raise FileNotFoundError(f"No */questions/*.json files under {input_root}")
    return result


def _normalise_answer_type(value: Any) -> str:
    """Normalize serialized answer-type spellings to ``option``/``number``."""

    token = str(value or "option").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"number", "numeric", "scalar"}:
        return "number"
    if token in {"option", "option_letter", "multiple_choice", "letter"}:
        return "option"
    raise ValueError(f"unsupported answer_type {value!r}")


def _payload_is_numeric(payload: dict[str, Any], samples: Sequence[dict[str, Any]]) -> bool:
    """Determine whether a raw payload uses the numeric identity fallback."""

    if "answer_type" in payload:
        return _normalise_answer_type(payload["answer_type"]) == "number"
    for sample in samples:
        if isinstance(sample, dict) and "answer_type" in sample:
            return _normalise_answer_type(sample["answer_type"]) == "number"
    return False


def _validate_v1_weights(payload_id: str, sample: dict[str, Any]) -> None:
    """Reject accidental non-v1 relation/dependency weights."""

    try:
        relation_weight = float(sample.get("relation_weight", 1.0))
        dependency_weight = float(sample.get("dependency_weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{payload_id}: relation/dependency weights must be numeric") from exc
    if relation_weight != 1.0:
        raise ValueError(f"{payload_id}: RelaCaTS v2 data contract requires r_g=1")
    if dependency_weight != 1.0:
        raise ValueError(f"{payload_id}: RelaCaTS v2 data contract requires d_gi=1")


def _validate_numeric_payload(
    payload: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    allow_nonstandard_budget: bool,
) -> list[dict[str, Any]]:
    """Validate a GSM8K/SVAMP 1-view × 32-response raw payload.

    Numeric answers have no finite option-label space.  The only valid v1
    relation is identity, so permutation metadata must be absent/empty and
    canonicalization is strict scalar-number normalization.  ``gold`` is not
    consulted here; it remains an audit field only.
    """

    question_id = str(payload.get("question_id", "<unknown>"))
    dataset_name = str(payload.get("dataset_name", ""))
    relation_mode = str(
        payload.get("relation_mode", "identity_only")
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if relation_mode not in {"identity", "identity_only"}:
        raise ValueError(
            f"{question_id}: numeric payload requires identity_only relation_mode, "
            f"got {payload.get('relation_mode')!r}"
        )
    if dataset_name not in {"gsm8k", "svamp"}:
        raise ValueError(
            f"{question_id}: numeric identity fallback is only supported for "
            f"gsm8k/svamp, got {dataset_name!r}"
        )

    num_views = payload["num_views"]
    per_view = payload["samples_per_view"]
    budget = payload["attempted_budget"]
    for name, value in (
        ("num_views", num_views),
        ("samples_per_view", per_view),
        ("attempted_budget", budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{question_id}: {name} must be a positive integer")
    if num_views * per_view != budget or len(samples) != budget:
        raise ValueError(
            f"{question_id}: budget metadata/data mismatch "
            f"({num_views}x{per_view}, attempted={budget}, records={len(samples)})"
        )
    if not allow_nonstandard_budget and (num_views, per_view, budget) != (1, 32, 32):
        raise ValueError(
            f"{question_id}: formal numeric budget must be 1x32=32, got "
            f"{num_views}x{per_view}={budget}"
        )

    relation_counts = Counter(str(record.get("relation_id")) for record in samples)
    if set(relation_counts) != {"g0"} or relation_counts.get("g0") != per_view:
        raise ValueError(
            f"{question_id}: numeric identity payload must contain {per_view} g0 "
            f"samples, got {dict(relation_counts)}"
        )

    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{question_id}: sample {index} must be an object")
        if sample.get("question_id") != payload["question_id"]:
            raise ValueError(f"{question_id}: sample {index} question_id mismatch")
        if sample.get("dataset_name") != dataset_name:
            raise ValueError(f"{question_id}: sample {index} dataset mismatch")
        if _normalise_answer_type(sample.get("answer_type", "number")) != "number":
            raise ValueError(f"{question_id}: sample {index} must have answer_type=number")
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{question_id}: sample {index} has no sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"{question_id}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if str(sample.get("relation_id", "")) != "g0":
            raise ValueError(f"{question_id}: numeric sample {index} must use relation_id=g0")
        relation_type = str(sample.get("relation_type", "identity")).strip().lower()
        if relation_type != "identity":
            raise ValueError(f"{question_id}: numeric sample {index} must use identity relation")
        sample_mode = str(
            sample.get("relation_mode", "identity_only")
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if sample_mode not in {"identity", "identity_only"}:
            raise ValueError(f"{question_id}: numeric sample {index} has non-identity mode")
        view_index = sample.get("view_index")
        if isinstance(view_index, bool) or not isinstance(view_index, int) or view_index != 0:
            raise ValueError(f"{question_id}: numeric sample {index} must have view_index=0")

        # Numeric identity records should not smuggle option permutations into
        # the fallback.  Accept both ``None`` (generator output) and ``{}``
        # (older serializers), but reject any non-empty mapping/list.
        for field in ("option_labels", "original_options", "transformed_options"):
            value = sample.get(field)
            if value not in (None, [], ()):
                raise ValueError(f"{question_id}: numeric sample {index} has {field}")
        for field in ("permutation", "inverse_permutation"):
            value = sample.get(field)
            if value not in (None, {}):
                raise ValueError(f"{question_id}: numeric sample {index} has {field}")

        recomputed = canonicalize_answer(
            sample.get("extracted_answer"),
            {"relation_type": "identity"},
            answer_type="number",
        )
        stored_valid = sample.get("is_valid_answer")
        stored_canonical = sample.get("canonicalized_answer")
        if not isinstance(stored_valid, bool):
            raise ValueError(f"{question_id}: sample {index} lacks boolean is_valid_answer")
        if stored_valid:
            if not recomputed.valid or stored_canonical != recomputed.canonicalized_answer:
                raise ValueError(
                    f"{question_id}: numeric sample {index} canonicalized answer "
                    "does not match extracted scalar answer"
                )
        elif recomputed.valid or stored_canonical is not None:
            raise ValueError(
                f"{question_id}: numeric sample {index} invalid-answer fields are inconsistent"
            )
        if "confidence" not in sample:
            raise ValueError(f"{question_id}: sample {index} lacks confidence")
        _validate_v1_weights(question_id, sample)
    return samples


def validate_question_payload(
    payload: dict[str, Any], *, allow_nonstandard_budget: bool
) -> list[dict[str, Any]]:
    required = {
        "question_id",
        "dataset_name",
        "num_views",
        "samples_per_view",
        "attempted_budget",
        "samples",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Question payload missing fields: {sorted(missing)}")
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise ValueError(f"{payload['question_id']}: samples must be a list")
    if _payload_is_numeric(payload, samples):
        return _validate_numeric_payload(
            payload,
            samples,
            allow_nonstandard_budget=allow_nonstandard_budget,
        )
    if str(payload.get("dataset_name", "")).strip().lower() in {"gsm8k", "svamp"}:
        raise ValueError(
            f"{payload['question_id']}: {payload['dataset_name']} requires "
            "answer_type=number and identity_only relation metadata"
        )
    if "answer_type" in payload and _normalise_answer_type(payload["answer_type"]) != "option":
        raise ValueError(
            f"{payload['question_id']}: option-permutation payload must use "
            "answer_type=option letter"
        )
    dataset_name = str(payload.get("dataset_name", "")).strip().lower()
    is_winogrande = dataset_name == "winogrande"
    def strict_positive_int(name: str) -> int:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{payload.get('question_id', '<unknown>')}: {name} must be a positive integer"
            )
        return value

    num_views = strict_positive_int("num_views")
    per_view = strict_positive_int("samples_per_view")
    budget = strict_positive_int("attempted_budget")
    if num_views * per_view != budget or len(samples) != budget:
        raise ValueError(
            f"{payload['question_id']}: budget metadata/data mismatch "
            f"({num_views}x{per_view}, attempted={budget}, records={len(samples)})"
        )
    if is_winogrande and num_views != 2:
        raise ValueError(
            f"{payload['question_id']}: WinoGrande requires exactly two distinct "
            f"views (identity and swap), got {num_views}"
        )
    formal_profile = (2, 16, 32) if is_winogrande else (4, 8, 32)
    if not allow_nonstandard_budget and (num_views, per_view, budget) != formal_profile:
        profile_name = "WinoGrande 2x16=32" if is_winogrande else "4x8=32"
        raise ValueError(
            f"{payload['question_id']}: formal budget must be {profile_name}, got "
            f"{num_views}x{per_view}={budget}"
        )
    relation_counts = Counter(str(record.get("relation_id")) for record in samples)
    expected_ids = {"g0", "g1"} if is_winogrande else {
        f"g{index}" for index in range(num_views)
    }
    if set(relation_counts) != expected_ids or any(
        count != per_view for count in relation_counts.values()
    ):
        raise ValueError(
            f"{payload['question_id']}: expected {per_view} samples for each of "
            f"{sorted(expected_ids)}, got {dict(relation_counts)}"
        )
    if is_winogrande and bool(payload.get("allow_repeated_views", False)):
        raise ValueError(
            f"{payload['question_id']}: WinoGrande must not allow repeated views"
        )
    sample_ids: set[str] = set()
    wino_orders: set[tuple[int, ...]] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{payload['question_id']}: sample {index} must be an object")
        if sample.get("question_id") != payload["question_id"]:
            raise ValueError(f"{payload['question_id']}: sample {index} question_id mismatch")
        if sample.get("dataset_name") != payload["dataset_name"]:
            raise ValueError(f"{payload['question_id']}: sample {index} dataset mismatch")
        if "answer_type" in sample and _normalise_answer_type(sample["answer_type"]) != "option":
            raise ValueError(
                f"{payload['question_id']}: option sample {index} must have "
                "answer_type=option letter"
            )
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{payload['question_id']}: sample {index} has no sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"{payload['question_id']}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        relation_id = str(sample.get("relation_id", ""))
        if not relation_id.startswith("g") or not relation_id[1:].isdigit():
            raise ValueError(f"{payload['question_id']}: invalid relation_id {relation_id!r}")
        if "view_index" not in sample or isinstance(sample["view_index"], bool) or not isinstance(sample["view_index"], int):
            raise ValueError(f"{payload['question_id']}: sample {index} lacks integer view_index")
        if sample["view_index"] != int(relation_id[1:]):
            raise ValueError(f"{payload['question_id']}: view_index/relation_id mismatch")
        # Reconstructing metadata here catches accidental forward/inverse
        # direction swaps before they contaminate every pseudo-label.
        try:
            permutation = OptionPermutation.from_metadata(sample)
        except Exception as exc:
            raise ValueError(
                f"{payload['question_id']}: invalid permutation in sample {index}: {exc}"
            ) from exc
        if is_winogrande:
            if len(permutation.labels) != 2 or permutation.labels != ("A", "B"):
                raise ValueError(
                    f"{payload['question_id']}: WinoGrande samples must use exactly "
                    "A/B option labels"
                )
            if bool(sample.get("is_duplicate_view", False)):
                raise ValueError(
                    f"{payload['question_id']}: WinoGrande sample {index} is marked "
                    "as a duplicate view"
                )
            wino_orders.add(tuple(permutation.forward_indices))
        # Recompute canonicalization from the extracted transformed-space
        # answer.  A stale/hand-edited canonicalized_answer (or a swapped
        # forward/inverse map) must never silently alter pseudo-labels.
        recomputed = canonicalize_answer(
            sample.get("extracted_answer"),
            permutation,
            answer_type="option",
            labels=permutation.labels,
        )
        stored_valid = sample.get("is_valid_answer")
        stored_canonical = sample.get("canonicalized_answer")
        if stored_valid is True:
            if not recomputed.valid or stored_canonical != recomputed.canonicalized_answer:
                raise ValueError(
                    f"{payload['question_id']}: sample {index} canonicalized answer "
                    "does not match extracted answer and inverse permutation"
                )
        elif stored_valid is False:
            if recomputed.valid or stored_canonical is not None:
                raise ValueError(
                    f"{payload['question_id']}: sample {index} invalid-answer fields "
                    "are inconsistent with canonicalization"
                )
        original_options = sample.get("original_options", payload.get("original_options"))
        transformed_options = sample.get("transformed_options")
        if isinstance(original_options, list) and isinstance(transformed_options, list):
            expected = list(permutation.permute_options(original_options))
            if expected != [str(value) for value in transformed_options]:
                raise ValueError(
                    f"{payload['question_id']}: transformed_options mismatch in sample {index}"
                )
        if "is_valid_answer" not in sample or not isinstance(sample["is_valid_answer"], bool):
            raise ValueError(
                f"{payload['question_id']}: sample {index} lacks boolean is_valid_answer"
            )
        if "confidence" not in sample:
            raise ValueError(f"{payload['question_id']}: sample {index} lacks confidence")
        # Explicitly reject accidental future v2 weights in a v1 builder.
        if float(sample.get("relation_weight", 1.0)) != 1.0:
            raise ValueError(
                f"{payload['question_id']}: RelaCaTS v2 data contract requires r_g=1"
            )
        if float(sample.get("dependency_weight", 1.0)) != 1.0:
            raise ValueError(
                f"{payload['question_id']}: RelaCaTS v2 data contract requires d_gi=1"
            )
    if is_winogrande and wino_orders != {(0, 1), (1, 0)}:
        raise ValueError(
            f"{payload['question_id']}: WinoGrande must contain exactly identity and "
            f"swap permutations, got {sorted(wino_orders)}"
        )
    return samples


def split_question_ids(
    question_ids: Sequence[str], test_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    """Deterministic group split; one original question never leaks across splits."""

    if not 0 <= test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1)")
    ordered = sorted(question_ids, key=lambda qid: stable_id(seed, qid, length=32))
    if len(ordered) <= 1 or test_ratio == 0:
        return set(ordered), set()
    test_count = max(1, int(round(len(ordered) * test_ratio)))
    test_count = min(test_count, len(ordered) - 1)
    test_ids = set(ordered[:test_count])
    return set(ordered[test_count:]), test_ids


def flatten_question(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = payload["samples"]
    relssc_result = compute_relssc(samples, zero_weight_policy="skip")
    if not relssc_result.defined:
        return [], {
            "question_id": payload["question_id"],
            "dataset_name": payload.get("dataset_name"),
            "answer_type": _normalise_answer_type(payload.get("answer_type", "option")),
            "relation_mode": payload.get("relation_mode", "option_permutation"),
            "defined": False,
            "reason": relssc_result.reason,
            "valid_response_count": relssc_result.valid_sample_count,
            "invalid_response_count": relssc_result.invalid_sample_count,
            "gold_used_in_target": False,
        }
    attached = attach_relssc_targets(samples)
    attached, ssc_context = attach_v2_target_inputs(attached)
    if not ssc_context.defined:
        return [], {
            "question_id": payload["question_id"],
            "dataset_name": payload.get("dataset_name"),
            "answer_type": _normalise_answer_type(payload.get("answer_type", "option")),
            "relation_mode": payload.get("relation_mode", "option_permutation"),
            "defined": False,
            "reason": "no valid identity-view answer remains; skip this question",
            "valid_response_count": relssc_result.valid_sample_count,
            "invalid_response_count": relssc_result.invalid_sample_count,
            "valid_identity_sample_count": ssc_context.valid_identity_samples,
            "invalid_identity_sample_count": ssc_context.invalid_identity_samples,
            "relation_valid_ratio": ssc_context.relation_valid_ratio,
            "gold_used_in_target": False,
        }
    rows: list[dict[str, Any]] = []
    for sample in attached:
        target = sample["relational_consistency"]
        if target is None or sample["is_valid_answer"] is not True:
            continue
        row = dict(sample)
        # ``transformed_prompt`` is emitted by the GPU generator.  Keeping a
        # deterministic fallback makes hand-inspected/minimal numeric payloads
        # (which have no option view) buildable without weakening validation of
        # the answer/RelSSC fields.
        prompt = sample.get(
            "transformed_prompt",
            sample.get("original_prompt", sample.get("transformed_question", "")),
        )
        row.update(
            {
                "input": f"{prompt}{sample.get('response', '')}",
                "answer": sample["canonicalized_answer"],
                "relssc": float(target),
                "relational_consistency": float(target),
                # ``ssc`` and ``relation_valid_ratio`` were attached above.
                # They are the independent v2 inputs used to resolve the
                # configured calibration target at training time.
                "question_relssc_scores": dict(relssc_result.scores),
                "question_relssc_total_weight": relssc_result.total_weight,
                "question_valid_response_count": relssc_result.valid_sample_count,
                "question_invalid_response_count": relssc_result.invalid_sample_count,
                "attempted_budget": int(payload["attempted_budget"]),
                "target_provenance": "relssc_without_gold",
                "v2_target_inputs_provenance": "ssc_and_relssc_without_gold",
            }
        )
        rows.append(row)
    return rows, {
        "question_id": payload["question_id"],
        "dataset_name": payload.get("dataset_name"),
        "answer_type": _normalise_answer_type(payload.get("answer_type", "option")),
        "relation_mode": payload.get("relation_mode", "option_permutation"),
        "defined": True,
        "top_answer": relssc_result.top_answer,
        "scores": dict(relssc_result.scores),
        "total_weight": relssc_result.total_weight,
        "valid_response_count": relssc_result.valid_sample_count,
        "invalid_response_count": relssc_result.invalid_sample_count,
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
) -> dict[str, Any]:
    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    question_summaries: list[dict[str, Any]] = []
    raw_count = 0
    seen_ids: set[str] = set()
    dataset_answer_types: set[str] = set()
    dataset_relation_modes: set[str] = set()
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}")
        if payload.get("dataset_name") != dataset_name:
            raise ValueError(f"Dataset directory/payload mismatch at {path}")
        question_id = str(payload["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id {question_id!r}")
        seen_ids.add(question_id)
        samples = validate_question_payload(
            payload, allow_nonstandard_budget=allow_nonstandard_budget
        )
        dataset_answer_types.add(
            _normalise_answer_type(payload.get("answer_type", "option"))
        )
        dataset_relation_modes.add(
            str(payload.get("relation_mode", "option_permutation"))
        )
        raw_count += len(samples)
        rows, summary = flatten_question(payload)
        question_summaries.append(summary)
        if rows:
            rows_by_question[question_id] = rows

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
    targets = [row["relational_consistency"] for row in train_rows + test_rows]
    ssc_targets = [row["ssc"] for row in train_rows + test_rows]
    relation_valid_ratios = [
        summary["relation_valid_ratio"]
        for summary in question_summaries
        if summary.get("defined") and summary.get("relation_valid_ratio") is not None
    ]
    stats = {
        "schema_version": "relacats-v2.dataset-stats.1",
        "dataset_name": dataset_name,
        "answer_types": sorted(dataset_answer_types),
        "relation_modes": sorted(dataset_relation_modes),
        "source_question_files": len(files),
        "defined_questions": len(rows_by_question),
        "skipped_zero_weight_questions": len(files) - len(rows_by_question),
        "train_questions": len(train_ids),
        "test_questions": len(test_ids),
        "raw_response_records": raw_count,
        "valid_training_records": len(train_rows) + len(test_rows),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "mean_relssc_target": sum(targets) / len(targets) if targets else None,
        "mean_ssc_target": sum(ssc_targets) / len(ssc_targets) if ssc_targets else None,
        "mean_relation_valid_ratio": (
            sum(relation_valid_ratios) / len(relation_valid_ratios)
            if relation_valid_ratios
            else None
        ),
        "high_relssc_records_gt_0_75": sum(value > 0.75 for value in targets),
        "group_split_no_question_leakage": not bool(train_ids & test_ids),
        "gold_used_in_target": False,
    }
    atomic_write_json(dataset_dir / "stats.json", stats)
    return stats


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    files_by_dataset = discover_question_files(input_root, args.datasets)
    summaries: list[dict[str, Any]] = []
    for dataset_name, files in files_by_dataset.items():
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
            f"{dataset_name}: questions={stats['defined_questions']} "
            f"train/test rows={stats['train_records']}/{stats['test_records']}"
        )
    manifest = {
        "schema_version": "relacats-v2.dataset-manifest.1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "formal_budget_enforced": not args.allow_nonstandard_budget,
        "invalid_policy": "retained in raw; excluded from RelSSC denominator/training",
        "split_unit": "original_question_id",
        "target_inputs": {
            "ssc": "count-based consistency on valid identity-view responses",
            "relssc": "confidence-weighted consistency over all relation views",
            "relation_valid_ratio": (
                "valid / attempted non-identity, non-duplicate relation responses; "
                "zero for identity-only datasets"
            ),
        },
        "target_modes": ["ssc", "relssc_replace", "residual"],
        "default_target_mode": "residual",
        "default_lambda_rel": 0.5,
        "generation_selection_target": "ssc",
        "gold_used_in_target": False,
        "datasets": summaries,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(f"Wrote RelaCaTS v2 dataset manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
