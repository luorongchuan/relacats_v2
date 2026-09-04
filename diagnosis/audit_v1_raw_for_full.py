"""Audit RelaCaTS-v1 raw teacher data before Full-RelaCaTS reuse.

This command is deliberately read-only.  It compares the answer fields stored
by the v1 generator with the strict explicit-final-answer parser used by v2,
recomputes canonicalization from the relation metadata, checks confidence
values, and summarizes the result per model/dataset.

Typical use::

    python -m relacats_v2.diagnosis.audit_v1_raw_for_full \
      --input-root relacats_v1/outputs/generated_data \
      --output relacats_v2/outputs/audits/v1_raw_for_full.json

No raw JSON file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from relacats_v2.common import atomic_write_json
from relacats_v2.core import OptionPermutation, canonicalize_answer
from relacats_v2.evaluation.answer_parsing import (
    STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
    extract_explicit_answer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "relacats_v1/outputs/generated_data"
DEFAULT_OUTPUT = REPO_ROOT / "relacats_v2/outputs/audits/v1_raw_for_full.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--max-questions-per-dataset",
        type=int,
        help="Optional quick-audit cap; omit for the full raw-data audit.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _answer_type(payload: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    value = sample.get("answer_type", payload.get("answer_type", "option letter"))
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "number" if token in {"number", "numeric", "scalar"} else "option letter"


def _normal_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _strict_canonical(
    payload: Mapping[str, Any], sample: Mapping[str, Any]
) -> tuple[Any, Any]:
    answer_type = _answer_type(payload, sample)
    dataset_name = str(sample.get("dataset_name", payload.get("dataset_name", "")))
    response = str(sample.get("response", ""))
    extracted = extract_explicit_answer(
        dataset_name,
        response,
        answer_type=answer_type,
    )
    if answer_type == "number":
        canonical = canonicalize_answer(
            extracted,
            {"relation_type": "identity"},
            answer_type="number",
        )
    else:
        permutation = OptionPermutation.from_metadata(sample)
        canonical = canonicalize_answer(
            extracted,
            permutation,
            answer_type="option",
            labels=permutation.labels,
        )
    return extracted, canonical


def _profile(payload: Mapping[str, Any]) -> str:
    samples = payload.get("samples", [])
    return (
        f"{payload.get('num_views')}x{payload.get('samples_per_view')}="
        f"{payload.get('attempted_budget')}|records={len(samples) if isinstance(samples, list) else 'NA'}"
    )


def audit_dataset(paths: list[Path]) -> dict[str, Any]:
    counters = Counter()
    profiles = Counter()
    relation_modes = Counter()
    strict_answer_examples: list[dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            counters["invalid_payload"] += 1
            continue
        samples = payload.get("samples")
        if not isinstance(samples, list):
            counters["invalid_payload"] += 1
            continue

        counters["questions"] += 1
        profiles[_profile(payload)] += 1
        relation_modes[str(payload.get("relation_mode"))] += 1

        for sample_index, sample in enumerate(samples):
            counters["samples"] += 1
            if not isinstance(sample, dict):
                counters["malformed_sample"] += 1
                continue
            if not str(sample.get("response", "")):
                counters["missing_response"] += 1

            old_valid = sample.get("is_valid_answer") is True
            if old_valid:
                counters["old_valid"] += 1
            else:
                counters["old_invalid"] += 1

            confidence = sample.get("confidence")
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                counters["bad_confidence"] += 1
            else:
                if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                    counters["bad_confidence"] += 1
                else:
                    counters["good_confidence"] += 1

            try:
                strict_extracted, strict_result = _strict_canonical(payload, sample)
            except Exception as exc:  # audit should continue and report structural failures
                counters["strict_parse_exception"] += 1
                if len(strict_answer_examples) < 20:
                    strict_answer_examples.append(
                        {
                            "file": str(path),
                            "sample_index": sample_index,
                            "kind": "exception",
                            "error": repr(exc),
                        }
                    )
                continue

            strict_valid = bool(strict_result.valid)
            if strict_valid:
                counters["strict_valid"] += 1
            else:
                counters["strict_invalid"] += 1

            if old_valid and strict_valid:
                counters["both_valid"] += 1
            elif old_valid and not strict_valid:
                counters["lost_by_strict"] += 1
            elif not old_valid and strict_valid:
                counters["recovered_by_strict"] += 1
            else:
                counters["both_invalid"] += 1

            old_canonical = _normal_text(sample.get("canonicalized_answer"))
            strict_canonical = _normal_text(strict_result.canonicalized_answer)
            if old_valid and strict_valid:
                if old_canonical == strict_canonical:
                    counters["canonical_agree"] += 1
                else:
                    counters["canonical_changed"] += 1
                    if len(strict_answer_examples) < 20:
                        strict_answer_examples.append(
                            {
                                "file": str(path),
                                "sample_index": sample_index,
                                "kind": "canonical_changed",
                                "old_extracted": sample.get("extracted_answer"),
                                "strict_extracted": strict_extracted,
                                "old_canonical": old_canonical,
                                "strict_canonical": strict_canonical,
                                "response_tail": str(sample.get("response", ""))[-500:],
                            }
                        )

    samples = counters["samples"]
    old_valid = counters["old_valid"]
    strict_valid = counters["strict_valid"]
    return {
        "questions": counters["questions"],
        "samples": samples,
        "profiles": dict(profiles),
        "relation_modes": dict(relation_modes),
        "old_valid": old_valid,
        "strict_valid": strict_valid,
        "old_valid_rate": old_valid / samples if samples else None,
        "strict_valid_rate": strict_valid / samples if samples else None,
        "lost_by_strict": counters["lost_by_strict"],
        "recovered_by_strict": counters["recovered_by_strict"],
        "canonical_changed": counters["canonical_changed"],
        "canonical_agree": counters["canonical_agree"],
        "missing_response": counters["missing_response"],
        "bad_confidence": counters["bad_confidence"],
        "strict_parse_exception": counters["strict_parse_exception"],
        "examples": strict_answer_examples,
    }


def main() -> None:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output = resolve_path(args.output)
    if not input_root.exists():
        raise FileNotFoundError(f"Raw-data root does not exist: {input_root}")
    if args.max_questions_per_dataset is not None and args.max_questions_per_dataset <= 0:
        raise ValueError("--max-questions-per-dataset must be positive")

    selected_models = set(args.models or [])
    selected_datasets = set(args.datasets or [])
    report: dict[str, Any] = {
        "schema_version": "relacats-v2.v1-raw-audit.1",
        "input_root": str(input_root),
        "strict_parser_version": STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
        "read_only": True,
        "models": {},
    }

    for model_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if selected_models and model_dir.name not in selected_models:
            continue
        model_report: dict[str, Any] = {}
        for dataset_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if selected_datasets and dataset_dir.name not in selected_datasets:
                continue
            paths = sorted((dataset_dir / "questions").glob("*.json"))
            if args.max_questions_per_dataset is not None:
                paths = paths[: args.max_questions_per_dataset]
            if not paths:
                continue
            stats = audit_dataset(paths)
            model_report[dataset_dir.name] = stats
            print(
                f"{model_dir.name}/{dataset_dir.name}: "
                f"questions={stats['questions']} samples={stats['samples']} "
                f"old_valid={stats['old_valid_rate']:.4f} "
                f"strict_valid={stats['strict_valid_rate']:.4f} "
                f"lost={stats['lost_by_strict']} recovered={stats['recovered_by_strict']} "
                f"changed={stats['canonical_changed']} bad_conf={stats['bad_confidence']}"
            )
        if model_report:
            report["models"][model_dir.name] = model_report

    if not report["models"]:
        raise FileNotFoundError(f"No model/dataset question JSON files under {input_root}")
    atomic_write_json(output, report)
    print(f"Wrote read-only audit report: {output}")


if __name__ == "__main__":
    main()
