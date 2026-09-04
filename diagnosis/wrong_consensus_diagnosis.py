"""Diagnose CaTS high-SSC wrong consensus with RelaCaTS-v1 witnesses.

This module intentionally separates the two uses of the gold answer:

* ``identify`` uses gold only to select *wrong* original-CaTS cases;
* ``compare`` recomputes RelSSC solely from canonicalized answers and model
  confidence.  Gold is consulted afterwards only for diagnostic accuracy.

The split also makes the expensive generation phase resumable.  The generated
``candidates.jsonl`` is accepted by
``relacats_v2.data_creation.generate_relational_data --candidate-file``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence

from relacats_v2.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    read_jsonl,
    stable_id,
)
from relacats_v2.core import canonicalize_answer, compute_relssc


SCHEMA_VERSION = 1
SUPPORTED_OPTION_DATASETS = {
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "openbookqa",
    "logiqa",
    "reclor",
    "math_qa",
}

_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is\s*)?[:=]?\s*"
    r"\(?\s*([A-E])\s*\)?",
    flags=re.IGNORECASE,
)
_OPTION_LINE_RE = re.compile(
    r"(?m)^[ \t]*([A-Ea-e])[.)][ \t]+(.+?)[ \t]*$"
)
_CONTROL_TOKEN_RE = re.compile(
    r"(?:<\|[^>\n]+\|>|<｜[^>\n]+｜>|</?s>|\[/?INST\])"
)
_USER_MARKERS = (
    "<|im_start|>user\n",
    "<|start_header_id|>user<|end_header_id|>",
    "<｜User｜>",
    "[INST]",
)
_END_MARKERS = (
    "<|im_end|>",
    "<|eot_id|>",
    "<｜Assistant｜>",
    "<|start_header_id|>assistant<|end_header_id|>",
    "[/INST]",
)


class DiagnosisInputError(ValueError):
    """Raised when an input artifact is ambiguous or malformed."""


@dataclass(frozen=True)
class WeightedConsensus:
    answer: str | None
    score: float | None
    scores: Mapping[str, float]
    total_confidence: float
    valid_count: int
    invalid_count: int


def _normalise_dataset_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "arceasy": "arc_easy",
        "arcchallenge": "arc_challenge",
        "commonsenseqa": "commonsense_qa",
        "open_book_qa": "openbookqa",
        "mathqa": "math_qa",
    }
    return aliases.get(name, name)


def _option_label(value: Any, number_of_options: int | None = None) -> str | None:
    """Normalize a stored/extracted multiple-choice answer to A--E."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        index = value - 1
        if 0 <= index < (number_of_options or 5):
            return chr(65 + index)
        return None
    token = str(value).strip()
    matches = list(_ANSWER_RE.finditer(token))
    if matches:
        label = matches[-1].group(1).upper()
    elif re.fullmatch(r"\(?\s*[A-Ea-e]\s*\)?[.:;]?", token):
        label = re.search(r"[A-Ea-e]", token).group(0).upper()  # type: ignore[union-attr]
    elif re.fullmatch(r"[1-5]", token):
        label = chr(64 + int(token))
    else:
        return None
    if number_of_options is not None and ord(label) - 64 > number_of_options:
        return None
    return label


def extract_response_answer(response: Any) -> str | None:
    """Extract the last explicit option answer from a CaTS response."""

    if not isinstance(response, str):
        return _option_label(response)
    matches = list(_ANSWER_RE.finditer(response))
    return matches[-1].group(1).upper() if matches else None


def weighted_consensus(
    answers: Sequence[Any], confidences: Sequence[Any]
) -> WeightedConsensus:
    """Recompute original SSC while explicitly excluding invalid answers."""

    if len(answers) != len(confidences):
        raise DiagnosisInputError(
            f"answers/confidences length mismatch: {len(answers)} != {len(confidences)}"
        )
    support: dict[str, list[float]] = defaultdict(list)
    invalid_count = 0
    for index, (answer_value, confidence_value) in enumerate(zip(answers, confidences)):
        answer = _option_label(answer_value)
        if answer is None:
            invalid_count += 1
            continue
        if isinstance(confidence_value, bool) or not isinstance(
            confidence_value, (int, float)
        ):
            raise DiagnosisInputError(
                f"confidence {index} must be numeric; got {confidence_value!r}"
            )
        confidence = float(confidence_value)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise DiagnosisInputError(
                f"confidence {index} must be in [0,1]; got {confidence_value!r}"
            )
        support[answer].append(confidence)
    masses = {answer: math.fsum(values) for answer, values in support.items()}
    total = math.fsum(masses.values())
    if total <= 0:
        return WeightedConsensus(None, None, {}, 0.0, sum(map(len, support.values())), invalid_count)
    scores = {answer: mass / total for answer, mass in masses.items()}
    answer = min(scores, key=lambda label: (-scores[label], label))
    return WeightedConsensus(
        answer=answer,
        score=scores[answer],
        scores=scores,
        total_confidence=total,
        valid_count=sum(map(len, support.values())),
        invalid_count=invalid_count,
    )


def _strip_chat_wrapper(prompt: str) -> str:
    text = prompt
    positions = [(text.rfind(marker), marker) for marker in _USER_MARKERS]
    position, marker = max(positions, key=lambda pair: pair[0])
    if position >= 0:
        text = text[position + len(marker) :]
    for end_marker in _END_MARKERS:
        end = text.find(end_marker)
        if end >= 0:
            text = text[:end]
    text = _CONTROL_TOKEN_RE.sub("", text).strip()
    # build_reasoning_prompt adds one outer "Question:".  Dataset handlers for
    # ARC/CSQA add another; remove at most two leading wrappers.
    for _ in range(2):
        updated = re.sub(r"^\s*Question\s*:\s*", "", text, count=1, flags=re.I)
        if updated == text:
            break
        text = updated
    return text.strip()


def parse_question_and_options(record: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    """Recover a candidate stem/options without consulting the gold answer."""

    raw_options = record.get("options", record.get("original_options"))
    question = record.get(
        "original_question",
        record.get("question_stem", record.get("stem")),
    )
    if isinstance(raw_options, Mapping):
        normalised = {str(key).strip().upper(): value for key, value in raw_options.items()}
        labels = [label for label in "ABCDE" if label in normalised]
        options = [str(normalised[label]).strip() for label in labels]
    elif isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes)):
        options = [str(option).strip() for option in raw_options]
    else:
        options = []

    if question is not None and options:
        return str(question).strip(), options

    prompt = record.get("prompt", record.get("input", ""))
    if not isinstance(prompt, str) or not prompt.strip():
        return (str(question).strip() if question else None), options
    user_text = _strip_chat_wrapper(prompt)
    matches = list(_OPTION_LINE_RE.finditer(user_text))
    if not matches:
        return (str(question).strip() if question else user_text), options

    # Only accept a contiguous A.. sequence; this prevents system prompt text
    # or response prose from silently becoming an option block.
    found: dict[str, str] = {}
    first_start: int | None = None
    for match in matches:
        label = match.group(1).upper()
        if label in found:
            continue
        if first_start is None:
            first_start = match.start()
        option_text = _CONTROL_TOKEN_RE.sub("", match.group(2)).strip()
        found[label] = option_text
    labels = [label for label in "ABCDE" if label in found]
    if not (2 <= len(labels) <= 5) or labels != list("ABCDE"[: len(labels)]):
        return (str(question).strip() if question else None), []
    options = [found[label] for label in labels]
    assert first_start is not None
    stem = user_text[:first_start]
    stem = re.sub(r"\n?\s*Options\s*:\s*$", "", stem, flags=re.I).strip()
    return stem or (str(question).strip() if question else None), options


def _iter_json_objects(path: Path) -> Iterator[tuple[dict[str, Any], str]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, record in enumerate(read_jsonl(path), start=1):
            yield record, f"{path}:{line_number}"
        return
    payload = read_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, Mapping):
        yield dict(payload), str(path)
    elif isinstance(payload, list):
        for index, record in enumerate(payload):
            if isinstance(record, Mapping):
                yield dict(record), f"{path}#{index}"
    else:
        raise DiagnosisInputError(f"expected JSON object/list at {path}")


def _source_files(path: str | Path) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file():
        return [source.resolve()]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted((*source.rglob("*.json"), *source.rglob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL artifacts below {source}")
    return [item.resolve() for item in files]


def _pool_consensus(record: Mapping[str, Any]) -> WeightedConsensus | None:
    responses = record.get("responses")
    confidences = record.get("confidence", record.get("confidences"))
    if not isinstance(responses, Sequence) or isinstance(responses, (str, bytes)):
        return None
    if not isinstance(confidences, Sequence) or isinstance(confidences, (str, bytes)):
        return None
    extracted = record.get("answers", record.get("extracted_answers"))
    if isinstance(extracted, Sequence) and not isinstance(extracted, (str, bytes)):
        answers = list(extracted)
    else:
        answers = [extract_response_answer(response) for response in responses]
    return weighted_consensus(answers, list(confidences))


def _explicit_original_consensus(
    record: Mapping[str, Any], recomputed: WeightedConsensus | None
) -> tuple[str | None, float | None, str]:
    answer_value = None
    for field in (
        "most_common_response_c",
        "weighted_consensus_answer",
        "consensus_answer",
        "top_answer",
    ):
        if field in record:
            answer_value = record[field]
            break
    score_value = None
    for field in (
        "consistency_score_c",
        "weighted_consistency",
        "ssc",
        "ssc_score",
    ):
        if field in record:
            score_value = record[field]
            break
    answer = _option_label(answer_value)
    try:
        score = float(score_value) if score_value is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
        score = None
    if answer is not None and score is not None:
        return answer, score, "stored_cats_fields"
    if recomputed is not None:
        return recomputed.answer, recomputed.score, "recomputed_response_pool"
    return None, None, "unavailable"


def identify_high_ssc_wrong_candidates(
    original_cats_path: str | Path,
    *,
    dataset_name: str,
    threshold: float = 0.9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select option-MCQ cases with original SSC > threshold and wrong top answer."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    dataset = _normalise_dataset_name(dataset_name)
    counters: dict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    records_seen = 0

    for source_file in _source_files(original_cats_path):
        for record, source_location in _iter_json_objects(source_file):
            # Metadata/manifests under a directory are not CaTS result rows.
            if not any(
                field in record
                for field in (
                    "responses",
                    "most_common_response_c",
                    "weighted_consensus_answer",
                )
            ):
                counters["ignored_non_result_records"] += 1
                continue
            records_seen += 1
            record_dataset = _normalise_dataset_name(record.get("dataset_name", dataset))
            if record_dataset and record_dataset != dataset:
                counters["excluded_other_dataset"] += 1
                continue

            stem, options = parse_question_and_options(record)
            gold = _option_label(
                record.get(
                    "correct_answer",
                    record.get("gold_original_answer", record.get("gold_answer")),
                ),
                len(options) if options else None,
            )
            recomputed = _pool_consensus(record)
            wrong_answer, ssc, ssc_source = _explicit_original_consensus(record, recomputed)
            if wrong_answer is None or ssc is None:
                counters["excluded_missing_consensus"] += 1
                continue
            if not ssc > threshold:
                counters["excluded_not_high_ssc"] += 1
                continue
            if gold is None:
                counters["excluded_missing_option_gold"] += 1
                continue
            if wrong_answer == gold:
                counters["excluded_correct_consensus"] += 1
                continue
            counters["raw_high_ssc_wrong"] += 1
            if dataset not in SUPPORTED_OPTION_DATASETS:
                counters["excluded_unsupported_dataset"] += 1
                continue
            if stem is None or not (2 <= len(options) <= 5):
                counters["excluded_unparseable_question_options"] += 1
                continue
            source_index = record.get("source_index", record.get("index"))
            if source_index is None:
                # Candidate-mode generation needs a stable integer filename.
                # This counter is deterministic for the ordered source files.
                source_index = records_seen - 1
            question_id = str(
                record.get("question_id")
                or stable_id(dataset, source_index, stem, length=20)
            )
            discrepancy = None
            if recomputed is not None and recomputed.answer is not None:
                if recomputed.answer != wrong_answer or (
                    recomputed.score is not None
                    and not math.isclose(recomputed.score, ssc, abs_tol=1e-8)
                ):
                    discrepancy = {
                        "recomputed_answer": recomputed.answer,
                        "recomputed_ssc": recomputed.score,
                    }
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": stable_id("wrong-consensus", dataset, question_id, length=20),
                    "question_id": question_id,
                    "dataset_name": dataset,
                    "source_index": source_index,
                    "original_question": stem,
                    "options": options,
                    "gold_original_answer": gold,
                    "original_wrong_consensus_answer": wrong_answer,
                    "original_ssc": ssc,
                    "original_ssc_source": ssc_source,
                    "original_prompt": record.get("prompt"),
                    "original_valid_response_count": (
                        recomputed.valid_count if recomputed else None
                    ),
                    "original_invalid_response_count": (
                        recomputed.invalid_count if recomputed else None
                    ),
                    "stored_vs_recomputed_discrepancy": discrepancy,
                    "gold_usage": "diagnostic_selection_only_not_relssc",
                    "source_location": source_location,
                }
            )

    candidates.sort(
        key=lambda item: (
            str(item["dataset_name"]),
            str(item.get("source_index", "")),
            str(item["question_id"]),
        )
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "identify_high_ssc_wrong_consensus",
        "dataset_name": dataset,
        "strict_threshold": threshold,
        "selection_rule": f"original SSC > {threshold} and consensus != gold",
        "records_seen": records_seen,
        "raw_high_ssc_wrong_count": counters["raw_high_ssc_wrong"],
        "generation_ready_candidate_count": len(candidates),
        "counters": dict(sorted(counters.items())),
        "mean_original_ssc": _mean_or_none(item["original_ssc"] for item in candidates),
        "gold_usage": "gold is used only to identify wrong cases",
    }
    return candidates, summary


def _mean_or_none(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _question_files(root: Path) -> list[Path]:
    nested = sorted(root.rglob("questions/*.json"))
    if nested:
        return nested
    return [
        path
        for path in sorted(root.rglob("*.json"))
        if not any(
            token in path.name.lower()
            for token in ("metadata", "manifest", "summary", "stats")
        )
    ]


def _flatten_relational_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    for field in ("samples", "relational_samples"):
        value = payload.get(field)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    # A flattened data JSONL row is itself one sample.
    if "confidence" in payload and any(
        key in payload for key in ("canonicalized_answer", "extracted_answer")
    ):
        return [dict(payload)]
    return []


def _canonicalized_samples(
    samples: Sequence[Mapping[str, Any]], number_of_options: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    labels = tuple("ABCDE"[:number_of_options])
    for sample in samples:
        copied = dict(sample)
        if copied.get("canonicalized_answer") is None and copied.get("extracted_answer") is not None:
            canonicalized = canonicalize_answer(
                copied.get("extracted_answer"),
                copied,
                answer_type="option",
                labels=labels,
            )
            copied.update(canonicalized.to_record_fields())
        copied.setdefault("relation_weight", 1.0)
        copied.setdefault("dependency_weight", 1.0)
        result.append(copied)
    return result


def load_relational_questions(
    relational_root: str | Path,
) -> list[dict[str, Any]]:
    """Load raw question artifacts or flattened JSONL without double counting."""

    root = Path(relational_root).expanduser()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = _question_files(root)
        if not files:
            files = sorted(root.rglob("*.jsonl"))
    else:
        raise FileNotFoundError(root)
    if not files:
        raise FileNotFoundError(f"no relational artifacts below {root}")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for path in files:
        if path.suffix.lower() == ".jsonl":
            payloads: Iterable[Mapping[str, Any]] = read_jsonl(path)
        else:
            payload = read_json(path)
            if isinstance(payload, Mapping):
                payloads = [payload]
            elif isinstance(payload, list):
                payloads = [item for item in payload if isinstance(item, Mapping)]
            else:
                continue
        for payload in payloads:
            samples = _flatten_relational_payload(payload)
            if not samples:
                continue
            first = samples[0]
            dataset = _normalise_dataset_name(
                payload.get("dataset_name", first.get("dataset_name", ""))
            )
            question_id = str(
                payload.get("question_id", first.get("question_id", "")) or ""
            )
            original_question = str(
                payload.get(
                    "original_question", first.get("original_question", "")
                )
                or ""
            )
            source_index = payload.get("source_index", first.get("source_index"))
            fallback = stable_id(dataset, source_index, original_question, length=20)
            key = (dataset, question_id or fallback)
            entry = grouped.setdefault(
                key,
                {
                    "dataset_name": dataset,
                    "question_id": question_id,
                    "source_index": source_index,
                    "original_question": original_question,
                    "samples": [],
                    "artifact_paths": [],
                },
            )
            entry["samples"].extend(samples)
            entry["artifact_paths"].append(str(path.resolve()))
    return list(grouped.values())


def _normalised_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _match_relational_question(
    candidate: Mapping[str, Any], relational: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, str]:
    dataset = _normalise_dataset_name(candidate.get("dataset_name"))
    pool = [item for item in relational if _normalise_dataset_name(item.get("dataset_name")) == dataset]
    candidate_id = str(candidate.get("question_id") or "")
    if candidate_id:
        exact = [item for item in pool if str(item.get("question_id") or "") == candidate_id]
        if len(exact) == 1:
            return exact[0], "question_id"
        if len(exact) > 1:
            raise DiagnosisInputError(f"multiple relational artifacts match question_id={candidate_id}")
    source_index = candidate.get("source_index")
    if source_index is not None:
        exact = [item for item in pool if str(item.get("source_index")) == str(source_index)]
        if len(exact) == 1:
            return exact[0], "source_index"
        if len(exact) > 1:
            raise DiagnosisInputError(
                f"multiple relational artifacts match source_index={source_index}"
            )
    question = _normalised_text(candidate.get("original_question"))
    if question:
        exact = [item for item in pool if _normalised_text(item.get("original_question")) == question]
        if len(exact) == 1:
            return exact[0], "original_question"
        if len(exact) > 1:
            raise DiagnosisInputError("multiple relational artifacts match original_question")
    return None, "unmatched"


def compare_candidates_with_relational_data(
    candidates: Sequence[Mapping[str, Any]],
    relational_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare SSC(wrong) with gold-free RelSSC(wrong) for each candidate."""

    # A dataset can legitimately have no cases above the strict threshold.  In
    # that case diagnosis is complete without loading a model or requiring an
    # otherwise empty relational artifact directory.
    relational_questions = (
        load_relational_questions(relational_root) if candidates else []
    )
    cases: list[dict[str, Any]] = []
    for candidate in candidates:
        matched, match_key = _match_relational_question(candidate, relational_questions)
        base = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate.get("candidate_id"),
            "question_id": candidate.get("question_id"),
            "dataset_name": candidate.get("dataset_name"),
            "source_index": candidate.get("source_index"),
            "gold_original_answer": candidate.get("gold_original_answer"),
            "original_wrong_consensus_answer": candidate.get(
                "original_wrong_consensus_answer"
            ),
            "original_ssc": candidate.get("original_ssc"),
            "gold_usage": "diagnostic_selection_and_posthoc_correctness_only",
        }
        if matched is None:
            cases.append({**base, "status": "unmatched_relational_data", "match_key": match_key})
            continue
        options = candidate.get("options")
        number_of_options = len(options) if isinstance(options, list) else 5
        samples = _canonicalized_samples(matched["samples"], number_of_options)
        relssc = compute_relssc(samples, zero_weight_policy="skip")
        if not relssc.defined:
            cases.append(
                {
                    **base,
                    "status": "undefined_zero_relational_confidence",
                    "match_key": match_key,
                    "relational_sample_count": len(samples),
                    "relational_valid_sample_count": relssc.valid_sample_count,
                    "relational_invalid_sample_count": relssc.invalid_sample_count,
                    "relssc_reason": relssc.reason,
                    "relational_artifact_paths": matched["artifact_paths"],
                }
            )
            continue
        wrong_answer = str(candidate["original_wrong_consensus_answer"])
        original_ssc = float(candidate["original_ssc"])
        wrong_relssc = float(relssc.scores.get(wrong_answer, 0.0))
        absolute_drop = original_ssc - wrong_relssc
        relative_drop = absolute_drop / original_ssc if original_ssc > 0 else None
        top_answer = relssc.top_answer
        top_score = max(relssc.scores.values())
        top_answers = sorted(
            answer
            for answer, score in relssc.scores.items()
            if math.isclose(score, top_score, rel_tol=0.0, abs_tol=1e-12)
        )
        wrong_unique_top = top_answers == [wrong_answer]
        wrong_displaced = wrong_answer not in top_answers
        gold = str(candidate.get("gold_original_answer"))
        relation_ids = {
            str(sample.get("relation_id"))
            for sample in samples
            if sample.get("relation_id") is not None
        }
        cases.append(
            {
                **base,
                "status": "evaluated",
                "match_key": match_key,
                "relssc_wrong_answer": wrong_relssc,
                "relssc_top_answer": top_answer,
                "relssc_top_answers": top_answers,
                "relssc_scores": dict(relssc.scores),
                "absolute_ssc_drop": absolute_drop,
                "relative_ssc_drop": relative_drop,
                # A tie is already evidence that the original uniquely wrong
                # consensus was dispersed.  ``wrong_top_displaced`` is the
                # stricter statistic where the wrong label is below every top.
                "wrong_consensus_broken": not wrong_unique_top,
                "wrong_top_displaced": wrong_displaced,
                "relational_top_is_gold": top_answer == gold,
                "relational_view_count": len(relation_ids),
                "relational_sample_count": len(samples),
                "relational_valid_sample_count": relssc.valid_sample_count,
                "relational_invalid_sample_count": relssc.invalid_sample_count,
                "relational_total_confidence": relssc.total_weight,
                "relational_artifact_paths": matched["artifact_paths"],
                "relssc_inputs_use_gold": False,
            }
        )

    evaluated = [case for case in cases if case["status"] == "evaluated"]
    broken = sum(bool(case["wrong_consensus_broken"]) for case in evaluated)
    displaced = sum(bool(case["wrong_top_displaced"]) for case in evaluated)
    corrected = sum(bool(case["relational_top_is_gold"]) for case in evaluated)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "compare_ssc_with_relssc",
        "candidate_count": len(candidates),
        "evaluated_count": len(evaluated),
        "unmatched_count": sum(case["status"] == "unmatched_relational_data" for case in cases),
        "undefined_relssc_count": sum(
            case["status"] == "undefined_zero_relational_confidence" for case in cases
        ),
        "mean_original_ssc_all_candidates": _mean_or_none(
            candidate.get("original_ssc") for candidate in candidates
        ),
        "mean_original_ssc": _mean_or_none(case["original_ssc"] for case in evaluated),
        "mean_relssc_wrong_answer": _mean_or_none(
            case["relssc_wrong_answer"] for case in evaluated
        ),
        "mean_absolute_ssc_drop": _mean_or_none(
            case["absolute_ssc_drop"] for case in evaluated
        ),
        "mean_relative_ssc_drop": _mean_or_none(
            case["relative_ssc_drop"] for case in evaluated
        ),
        "relative_drop_of_means": (
            None
            if not evaluated
            else _relative_drop_of_means(evaluated)
        ),
        "wrong_consensus_broken_count": broken,
        "wrong_consensus_broken_rate": broken / len(evaluated) if evaluated else None,
        "wrong_top_displaced_count": displaced,
        "wrong_top_displaced_rate": displaced / len(evaluated) if evaluated else None,
        "relational_top_correct_count": corrected,
        "relational_top_correct_rate": corrected / len(evaluated) if evaluated else None,
        "relssc_definition": (
            "sum confidence over canonicalized wrong answer / sum confidence over "
            "all valid relational samples; r_g=d_gi=1"
        ),
        "gold_usage": (
            "gold selects wrong cases and scores posthoc correctness; it is never an "
            "input to RelSSC"
        ),
    }
    return cases, summary


def _relative_drop_of_means(evaluated: Sequence[Mapping[str, Any]]) -> float | None:
    original = _mean_or_none(case["original_ssc"] for case in evaluated)
    relational = _mean_or_none(case["relssc_wrong_answer"] for case in evaluated)
    if original is None or relational is None or original <= 0:
        return None
    return (original - relational) / original


def _format_percent(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _format_float(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def write_summary_markdown(path: str | Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# RelaCaTS-v1 wrong-consensus diagnosis",
        "",
        f"- Candidate high-SSC wrong cases: {summary.get('candidate_count', 0)}",
        f"- Evaluated with relational data: {summary.get('evaluated_count', 0)}",
        f"- Unmatched cases: {summary.get('unmatched_count', 0)}",
        f"- Undefined RelSSC cases: {summary.get('undefined_relssc_count', 0)}",
        f"- Mean original SSC(wrong): {_format_float(summary.get('mean_original_ssc'))}",
        f"- Mean RelSSC(wrong): {_format_float(summary.get('mean_relssc_wrong_answer'))}",
        f"- Mean absolute drop: {_format_float(summary.get('mean_absolute_ssc_drop'))}",
        f"- Mean per-case relative drop: {_format_percent(summary.get('mean_relative_ssc_drop'))}",
        f"- Relative drop of means: {_format_percent(summary.get('relative_drop_of_means'))}",
        f"- Wrong consensus broken: {summary.get('wrong_consensus_broken_count', 0)} "
        f"({_format_percent(summary.get('wrong_consensus_broken_rate'))})",
        f"- Wrong label strictly displaced from top: {summary.get('wrong_top_displaced_count', 0)} "
        f"({_format_percent(summary.get('wrong_top_displaced_rate'))})",
        f"- Relational top answer becomes gold: {summary.get('relational_top_correct_count', 0)} "
        f"({_format_percent(summary.get('relational_top_correct_rate'))})",
        "",
        "Gold answers are used only to select wrong cases and to score post-hoc "
        "correctness. They are not used in RelSSC.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_candidates(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identify = subparsers.add_parser("identify", help="select original high-SSC wrong cases")
    identify.add_argument("--original-cats-file", required=True)
    identify.add_argument("--dataset-name", required=True)
    identify.add_argument("--threshold", type=float, default=0.9)
    identify.add_argument("--candidates-output", required=True)
    identify.add_argument("--summary-output", required=True)

    compare = subparsers.add_parser("compare", help="compare candidates with relational raw")
    compare.add_argument("--candidates-file", required=True)
    compare.add_argument("--relational-root", required=True)
    compare.add_argument("--cases-output", required=True)
    compare.add_argument("--summary-output", required=True)
    compare.add_argument("--markdown-output")

    run = subparsers.add_parser("run", help="identify and compare in one CPU command")
    run.add_argument("--original-cats-file", required=True)
    run.add_argument("--dataset-name", required=True)
    run.add_argument("--threshold", type=float, default=0.9)
    run.add_argument("--relational-root", required=True)
    run.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "identify":
        candidates, summary = identify_high_ssc_wrong_candidates(
            args.original_cats_file,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
        )
        atomic_write_jsonl(args.candidates_output, candidates)
        atomic_write_json(args.summary_output, summary)
        print(
            f"Identified {len(candidates)} generation-ready high-SSC wrong cases; "
            f"wrote {args.candidates_output}"
        )
        return 0
    if args.command == "compare":
        candidates = _read_candidates(args.candidates_file)
        cases, summary = compare_candidates_with_relational_data(
            candidates, args.relational_root
        )
        atomic_write_jsonl(args.cases_output, cases)
        atomic_write_json(args.summary_output, summary)
        if args.markdown_output:
            write_summary_markdown(args.markdown_output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, identify_summary = identify_high_ssc_wrong_candidates(
        args.original_cats_file,
        dataset_name=args.dataset_name,
        threshold=args.threshold,
    )
    atomic_write_jsonl(output_dir / "candidates.jsonl", candidates)
    atomic_write_json(output_dir / "identification_summary.json", identify_summary)
    cases, summary = compare_candidates_with_relational_data(
        candidates, args.relational_root
    )
    atomic_write_jsonl(output_dir / "cases.jsonl", cases)
    atomic_write_json(output_dir / "summary.json", summary)
    write_summary_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
