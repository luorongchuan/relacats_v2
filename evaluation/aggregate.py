"""Strict-denominator CPU aggregation for RelaCaTS and CaTS baselines.

The response pool is never filtered at question level.  Malformed answers,
missing confidence scores, and incomplete budgets are explicitly counted and
remain in the accuracy denominator instead of silently making a dataset look
better.  Test-time relational transformations are intentionally absent in
RelaCaTS-v2.

Dynamic thresholds have an explicit two-stage contract.  A validation run
selects and persists one parameter per model/dataset/method/target-budget; a
test run only reads those persisted parameters.  The diagnostic ``analysis``
phase keeps threshold curves available for unit tests and exploratory work,
but it is deliberately not the CLI default and is labelled non-reportable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from relacats_v2.common import atomic_write_json, read_json, read_jsonl, stable_id
from relacats_v2.evaluation.answer_parsing import (
    extract_dataset_answer,
    parser_version,
)
from relacats_v2.evaluation.method_names import (
    TABLE2_METHOD_ORDER,
    canonical_method_name,
    canonicalize_report_methods,
)


CALIBRATION_SCHEMA_VERSION = "relacats-v2.dynamic-thresholds.1"
EVALUATION_PHASES = ("validation", "test", "analysis")


# Keep the method sets explicit.  The first seven names are the original
# Table-2 baselines; the three ``RelaCaTS-*`` names are the calibrated
# methods trained in this repository.  Older confidence artifacts may still
# contain ``CaTS-*`` labels, which are accepted by ``canonical_method_name``
# in the predictor dispatch below.
FIXED_METHODS = (
    "SC",
    "CISC",
    "Self-Certainty",
    "Best-of-N",
    "RelaCaTS-SC",
)
DYNAMIC_METHODS = (
    "RelaCaTS-ES",
    "ASC",
    "RelaCaTS-ASC",
    "ESC",
    "RASC",
)


# A confidence record produced by the current v1 pipeline only contains the
# calibrated P(Yes) score.  CISC, Self-Certainty, and RASC normally require
# additional *untrained* or token/reasoning-level scores.  We therefore expose
# the source policy in every report instead of silently presenting the
# calibrated score as an exact reproduction of those baselines.  If a caller
# supplies one of the optional fields listed here, the corresponding method is
# evaluated from that field and its status becomes ``exact``.
METHOD_SCORE_FIELDS: dict[str, tuple[str, ...]] = {
    "CISC": (
        "cisc_confidence",
        "untrained_confidence",
        "response_probability",
        "base_confidence",
        "model_confidence",
    ),
    "Self-Certainty": (
        "self_certainty",
        "self_certainty_score",
        "self_certainty_confidence",
    ),
    "RASC": (
        "rasc_score",
        "rasc_confidence",
        "sufficiency_score",
        "reasoning_quality_score",
    ),
}

# Fields that are optional in legacy confidence JSONL but required by one of
# the newly exposed baselines when available.  ``_question_groups`` keeps
# these fields instead of reducing every record to calibrated P(Yes), allowing
# exact CISC/Self-Certainty/RASC artifacts to pass through unchanged.
OPTIONAL_METHOD_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        field
        for fields in METHOD_SCORE_FIELDS.values()
        for field in fields
    )
)
OPTIONAL_METHOD_FIELDS += (
    "self_certainty_token_probabilities",
    "self_certainty_token_logprobs",
    "token_probabilities",
    "token_logprobs",
    "self_certainty_vocab_size",
    "vocab_size",
)

METHOD_METADATA: dict[str, dict[str, Any]] = {
    "SC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "answer majority",
    },
    "CISC": {
        "family": "baseline",
        "implementation_status": "proxy_if_missing_field",
        "score_source": "untrained confidence field; calibrated confidence fallback",
    },
    "Self-Certainty": {
        "family": "baseline",
        "implementation_status": "exact_if_token_scores_else_proxy",
        "score_source": "self-certainty field; calibrated confidence fallback",
    },
    "Best-of-N": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "record confidence",
    },
    "RelaCaTS-SC": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "RelaCaTS-ES": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "ASC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "answer frequency",
    },
    "RelaCaTS-ASC": {
        "family": "RelaCaTS",
        "implementation_status": "exact",
        "score_source": "calibrated confidence",
    },
    "ESC": {
        "family": "baseline",
        "implementation_status": "exact",
        "score_source": "unanimous non-overlapping windows",
    },
    "RASC": {
        "family": "baseline",
        "implementation_status": "exact_if_reasoning_score_else_proxy",
        "score_source": "reasoning/sufficiency score; calibrated confidence fallback",
    },
}


@dataclass(frozen=True)
class AggregateConfig:
    budgets: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    # Match the original CaTS analysis grid (0.00, 0.01, ..., 1.00).
    thresholds: tuple[float, ...] = tuple(index / 100 for index in range(101))
    curve_max_budget: int = 32
    budget_targets: tuple[int, ...] = (16,)
    dynamic_min_valid: int = 2
    # RASC's original implementation fills a small high-quality buffer before
    # stopping.  Keep the capacity configurable while retaining a conservative
    # default for the Table-2 budget protocol.
    rasc_buffer_size: int = 5
    # ESC is parameterized by a non-overlapping window size rather than a
    # confidence threshold.  An empty tuple means “use every size from 2
    # through curve_max_budget”, which keeps the CLI backwards compatible.
    esc_window_sizes: tuple[int, ...] = ()
    # CISC normalizes confidence within each question using a temperature-
    # scaled softmax.  The CISC paper tunes this on held-out questions; expose
    # the value so a reproduction can provide the published/tuned setting.
    cisc_temperature: float = 1.0
    cisc_normalization: str = "softmax"

    def __post_init__(self) -> None:
        if not self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        if 16 not in self.budgets:
            raise ValueError("The fixed budgets must include 16")
        if self.curve_max_budget <= 0:
            raise ValueError("curve_max_budget must be positive")
        if not self.thresholds or any(not 0 <= value <= 1 for value in self.thresholds):
            raise ValueError("thresholds must lie in [0, 1]")
        if any(value <= 0 for value in self.budget_targets):
            raise ValueError("budget_targets must be positive")
        if self.dynamic_min_valid <= 0:
            raise ValueError("dynamic_min_valid must be positive")
        if self.rasc_buffer_size <= 0:
            raise ValueError("rasc_buffer_size must be positive")
        if any(value <= 0 for value in self.esc_window_sizes):
            raise ValueError("esc_window_sizes must contain positive integers")
        if any(value > self.curve_max_budget for value in self.esc_window_sizes):
            raise ValueError("esc_window_sizes cannot exceed curve_max_budget")
        if not math.isfinite(self.cisc_temperature) or self.cisc_temperature <= 0:
            raise ValueError("cisc_temperature must be finite and positive")
        if self.cisc_normalization not in {"softmax", "linear", "none"}:
            raise ValueError(
                "cisc_normalization must be one of: softmax, linear, none"
            )


def _normalise_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    return text.upper() if text else None


def _confidence(record: Mapping[str, Any]) -> float | None:
    value = record.get("confidence")
    return _finite_float(value)


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing/malformed values."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_score(
    record: Mapping[str, Any], fields: Sequence[str]
) -> tuple[float | None, str | None]:
    """Find the first finite optional method score and its field name."""

    for field in fields:
        value = _finite_float(record.get(field))
        if value is not None:
            return value, field
    return None, None


def _self_certainty_from_tokens(record: Mapping[str, Any]) -> float | None:
    """Compute the published Self-Certainty score when token distributions exist.

    The Self-Certainty paper defines ``-(1/nV) * sum log(V p_j)`` over all
    generated positions and vocabulary entries.  Confidence artifacts made by
    early v1 runs do not retain these distributions; this helper intentionally
    returns ``None`` in that case rather than fabricating a token score.
    """

    # A producer may save either probabilities or log-probabilities.  Accept a
    # few unambiguous spellings to make the evaluator useful with exported
    # artifacts from different inference backends.
    values = record.get("self_certainty_token_probabilities")
    log_values = record.get("self_certainty_token_logprobs")
    if values is None:
        values = record.get("token_probabilities")
    if log_values is None:
        log_values = record.get("token_logprobs")

    vocab_raw = record.get("self_certainty_vocab_size", record.get("vocab_size"))
    vocab_size = None
    if vocab_raw is not None:
        try:
            candidate_vocab = int(vocab_raw)
        except (TypeError, ValueError):
            candidate_vocab = 0
        if candidate_vocab > 0:
            vocab_size = candidate_vocab

    rows: Any = values if values is not None else log_values
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    use_logs = values is None
    terms: list[float] = []
    for row in rows:
        # Each token position should contain a full-vocabulary vector.  A
        # scalar is accepted as a one-element vocabulary, which is useful for
        # compact test fixtures but still follows the same equation.
        if isinstance(row, (int, float)):
            vector = [row]
        elif isinstance(row, (list, tuple)):
            vector = list(row)
        elif isinstance(row, Mapping):
            vector = list(row.values())
        else:
            return None
        if not vector:
            return None
        vocab = vocab_size or len(vector)
        # A saved vocabulary size is a contract that each position contains
        # one probability for every vocabulary item.  A shorter vector is
        # normally a top-k/sparse export and cannot be used in the published
        # full-vocabulary equation, so fall back to the explicitly labelled
        # calibrated-confidence proxy instead of fabricating a score.
        if vocab_size is not None and len(vector) != vocab_size:
            return None
        for raw_value in vector:
            value = _finite_float(raw_value)
            if value is None:
                return None
            if use_logs:
                log_probability = value
            else:
                if value <= 0.0:
                    return None
                log_probability = math.log(value)
            # p=0 contributes -infinity and is not a useful finite score for
            # ranking; reject such a sparse/truncated export rather than
            # pretending it is a full-vocabulary Self-Certainty score.
            if not math.isfinite(log_probability):
                return None
            terms.append(-(math.log(vocab) + log_probability))
    return sum(terms) / len(terms) if terms else None


def _method_score(
    record: Mapping[str, Any], method: str
) -> tuple[float | None, str]:
    """Return a method-specific score and whether it is exact or a proxy.

    CISC uses an untrained confidence/probability, Self-Certainty uses its
    token-level certainty score, and RASC uses a reasoning/sufficiency score.
    If those fields are absent (the normal v1 confidence artifact), the only
    available score is calibrated P(Yes); using it is explicitly labelled
    ``proxy`` in the returned source string and report metadata.
    """

    canonical = canonical_method_name(method)
    if canonical == "RelaCaTS-SC" or canonical == "RelaCaTS-ES" or canonical == "RelaCaTS-ASC":
        value = _confidence(record)
        return value, "calibrated_confidence"
    if canonical == "CISC":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["CISC"])
        if value is not None:
            return value, f"exact:{field}"
        return _confidence(record), "proxy:calibrated_confidence"
    if canonical == "Self-Certainty":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["Self-Certainty"])
        if value is not None:
            return value, f"exact:{field}"
        value = _self_certainty_from_tokens(record)
        if value is not None:
            return value, "exact:token_distributions"
        return _confidence(record), "proxy:calibrated_confidence"
    if canonical == "RASC":
        value, field = _optional_score(record, METHOD_SCORE_FIELDS["RASC"])
        if value is not None:
            return value, f"exact:{field}"
        return _confidence(record), "proxy:calibrated_confidence"
    return _confidence(record), "confidence"


def _answer(record: Mapping[str, Any]) -> str | None:
    return _normalise_answer(record.get("extracted_answer", record.get("answer")))


def _winner(scores: Mapping[str, float | int]) -> str | None:
    """Argmax with CaTS-compatible first-observed tie breaking."""

    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


def _majority(records: Sequence[Mapping[str, Any]]) -> str | None:
    counts, _ = _answer_count_state(records)
    return _winner(counts)


def _answer_count_state(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], int]:
    """Return ordinary ASC vote counts and the number of valid answers."""

    counts: dict[str, int] = {}
    valid = 0
    for record in records:
        answer = _answer(record)
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
            valid += 1
    return counts, valid


def _confidence_weighted_vote_state(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], float, int]:
    """Return RelaCaTS-ASC ``V_k``, its denominator, and valid pair count.

    A record contributes only when both its explicit answer and calibrated
    confidence are valid.  Non-negative finite weights are accepted (zero is
    harmless); negative values cannot be probabilities and are ignored.  The
    same state is used by both the stopping rule and final prediction, which
    prevents accidentally reusing ordinary ASC's count-based stopping index.
    """

    scores: dict[str, float] = {}
    total_weight = 0.0
    valid = 0
    for record in records:
        answer = _answer(record)
        confidence = _method_score(record, "RelaCaTS-ASC")[0]
        if answer is None or confidence is None or confidence < 0.0:
            continue
        scores[answer] = scores.get(answer, 0.0) + confidence
        total_weight += confidence
        valid += 1
    return scores, total_weight, valid


def _vote_ratio(
    scores: Mapping[str, float | int], denominator: float | int
) -> tuple[str | None, float]:
    leader = _winner(scores)
    if leader is None or denominator <= 0:
        return leader, 0.0
    return leader, float(scores[leader]) / float(denominator)


def _asc_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    """Ordinary ASC final vote: unweighted answer frequency."""

    scores, _ = _answer_count_state(records)
    return _winner(scores)


def _relacats_asc_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    """RelaCaTS-ASC final vote: confidence-weighted ``V_k(a)``."""

    scores, _, _ = _confidence_weighted_vote_state(records)
    return _winner(scores)


def _weighted_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    return _weighted_vote_with(records, lambda record: _confidence(record))


def _weighted_vote_with(
    records: Sequence[Mapping[str, Any]],
    score_getter: Callable[[Mapping[str, Any]], float | None],
) -> str | None:
    scores: dict[str, float] = {}
    for record in records:
        answer = _answer(record)
        score = score_getter(record)
        if answer is None or score is None:
            continue
        scores[answer] = scores.get(answer, 0.0) + score
    return _winner(scores)


def _cisc_vote(
    records: Sequence[Mapping[str, Any]],
    temperature: float = 1.0,
    normalization: str = "softmax",
) -> str | None:
    """Confidence-informed self-consistency (CISC).

    The released CISC implementation applies a temperature-scaled softmax to
    per-response confidence and sums utilities by answer.  Multiplying all
    utilities by a common normalization is unnecessary for the argmax, so we
    use a numerically stable shifted exponential here.  ``linear`` and
    ``none`` are accepted for compatibility with the released CISC utility;
    the paper's recommended/default protocol is ``softmax``.  Missing scores
    are skipped; if every optional score is absent ``_method_score`` provides
    the explicitly documented calibrated-confidence proxy.
    """

    candidates: list[tuple[str, float]] = []
    for record in records:
        answer = _answer(record)
        score, _ = _method_score(record, "CISC")
        if answer is not None and score is not None:
            candidates.append((answer, score))
    if not candidates:
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("CISC temperature must be finite and positive")
    if normalization == "softmax":
        # CISC scores are probabilities in the original protocol.  For
        # exported logits/certainty values, softmax remains well-defined and
        # preserves the intended confidence ordering.
        maximum = max(score for _, score in candidates)
        weights = [
            (answer, math.exp((score - maximum) / temperature))
            for answer, score in candidates
        ]
    elif normalization == "linear":
        minimum = min(score for _, score in candidates)
        weights = [
            (answer, temperature * (score - minimum) + 1.0)
            for answer, score in candidates
        ]
    elif normalization == "none":
        weights = list(candidates)
    else:
        raise ValueError(
            "CISC normalization must be one of: softmax, linear, none"
        )
    totals: dict[str, float] = {}
    for answer, weight in weights:
        totals[answer] = totals.get(answer, 0.0) + weight
    return _winner(totals)


def _self_certainty_vote(records: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [
        record
        for record in records
        if _answer(record) is not None and _method_score(record, "Self-Certainty")[0] is not None
    ]
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda record: _method_score(record, "Self-Certainty")[0],
    )
    return _answer(selected)


def _fixed_predict(
    method: str,
    records: Sequence[Mapping[str, Any]],
    budget: int,
    cisc_temperature: float = 1.0,
    cisc_normalization: str = "softmax",
) -> tuple[str | None, int, str]:
    if len(records) < budget:
        return None, len(records), "insufficient_samples"
    prefix = records[:budget]
    method = canonical_method_name(method)
    if method == "SC":
        prediction = _majority(prefix)
    elif method == "CISC":
        prediction = _cisc_vote(prefix, cisc_temperature, cisc_normalization)
    elif method == "Self-Certainty":
        prediction = _self_certainty_vote(prefix)
    elif method == "Best-of-N":
        candidates = [
            record
            for record in prefix
            if _answer(record) is not None and _confidence(record) is not None
        ]
        prediction = (
            _answer(max(candidates, key=lambda item: _confidence(item)))
            if candidates
            else None
        )
    elif method == "RelaCaTS-SC":
        prediction = _weighted_vote_with(
            prefix, lambda record: _method_score(record, "RelaCaTS-SC")[0]
        )
    else:
        raise ValueError(f"Unknown fixed method: {method}")
    return prediction, budget, "ok" if prediction is not None else "no_valid_answer"


def _dynamic_predict(
    method: str,
    records: Sequence[Mapping[str, Any]],
    threshold: float,
    max_budget: int,
    min_valid: int,
    rasc_buffer_size: int = 5,
) -> tuple[str | None, int, str]:
    method = canonical_method_name(method)
    prefix = records[:max_budget]
    if method == "ASC":
        for index, record in enumerate(prefix):
            scores, valid = _answer_count_state(prefix[: index + 1])
            if valid >= min_valid:
                leader, ratio = _vote_ratio(scores, valid)
                if ratio >= threshold:
                    return leader, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _asc_vote(prefix)
    elif method == "RelaCaTS-ES":
        for index, record in enumerate(prefix):
            answer = _answer(record)
            confidence = _method_score(record, "RelaCaTS-ES")[0]
            if answer is not None and confidence is not None and confidence >= threshold:
                return answer, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _weighted_vote_with(
            prefix, lambda record: _method_score(record, "RelaCaTS-ES")[0]
        )
    elif method == "RelaCaTS-ASC":
        for index, record in enumerate(prefix):
            scores_float, total, valid = _confidence_weighted_vote_state(
                prefix[: index + 1]
            )
            if valid >= min_valid:
                leader, ratio = _vote_ratio(scores_float, total)
                if ratio >= threshold:
                    return leader, index + 1, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _relacats_asc_vote(prefix)
    elif method == "ESC":
        # ESC (Early-Stopping Self-Consistency) samples complete, sequential,
        # non-overlapping windows.  A window can trigger stopping only when
        # every response in it has the same valid answer.  If no window is
        # unanimous, vote over the complete windows and never charge a
        # trailing partial window.
        window_size = int(round(threshold))
        if window_size < 2:
            # Backwards-compatible callers may still pass a normalized value
            # in [0, 1].  Map it to a useful window while making the emitted
            # curves use integer sizes (see ``evaluate_records``).
            window_size = max(2, int(round(float(threshold) * max_budget)))
        window_size = min(window_size, max_budget)
        usable = (len(prefix) // window_size) * window_size
        sampled = prefix[:usable]
        for start in range(0, usable, window_size):
            window = sampled[start : start + window_size]
            answers = [_answer(record) for record in window]
            if answers and all(answer is not None for answer in answers) and len(set(answers)) == 1:
                return answers[0], start + window_size, "early_stop"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _majority(sampled)
        used = usable
        return prediction, used, "ok" if prediction is not None else "no_valid_answer"
    elif method == "RASC":
        # RASC uses a high-quality buffer: retain responses whose reasoning /
        # sufficiency score clears the threshold, stop when the buffer reaches
        # capacity, then vote among buffered answers.  The current v1 records
        # do not include RASC's learned feature score, so ``_method_score``
        # falls back to calibrated confidence and the report marks this as a
        # proxy.  Artifacts carrying ``rasc_score`` (or an accepted alias) use
        # the exact same control flow with that score.
        capacity = max(1, int(rasc_buffer_size))
        buffer: list[Mapping[str, Any]] = []
        for index, record in enumerate(prefix):
            answer = _answer(record)
            score = _method_score(record, "RASC")[0]
            if answer is None or score is None or score < threshold:
                continue
            buffer.append(record)
            if len(buffer) >= capacity:
                prediction = _weighted_vote_with(
                    buffer, lambda item: _method_score(item, "RASC")[0]
                )
                return prediction, index + 1, "early_stop" if prediction is not None else "no_valid_answer"
        if len(records) < max_budget:
            return None, len(records), "insufficient_samples"
        prediction = _weighted_vote_with(
            prefix, lambda item: _method_score(item, "RASC")[0]
        )
    else:
        raise ValueError(f"Unknown dynamic method: {method}")
    return prediction, max_budget, "ok" if prediction is not None else "no_valid_answer"


def _question_groups(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str | None], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_by_question: dict[str, str | None] = {}
    seen_samples: dict[str, tuple[Any, ...]] = {}
    invalid_answers = 0
    missing_confidence = 0
    yes_missing = 0
    total_input = 0
    input_splits: set[str] = set()
    input_datasets: set[str] = set()
    strict_response_reparsed = 0
    legacy_answer_fallback = 0
    reparsed_answer_disagreements = 0

    for row_index, raw in enumerate(records):
        total_input += 1
        if raw.get("split") is not None:
            input_splits.add(str(raw["split"]).strip().lower())
        if raw.get("dataset_name") is not None:
            input_datasets.add(str(raw["dataset_name"]).strip())
        question_id = str(
            raw.get("question_id")
            or f"legacy:{stable_id(raw.get('prompt', ''), length=20)}"
        )
        generation_index = int(raw.get("generation_index", len(grouped[question_id])))
        sample_id = str(
            raw.get("sample_id")
            or stable_id(question_id, generation_index, row_index, length=24)
        )
        dataset_name = str(raw.get("dataset_name") or "")
        if "response" in raw and raw.get("response") is not None:
            extracted_answer = extract_dataset_answer(
                dataset_name,
                str(raw.get("response", "")),
                answer_type=(
                    str(raw["answer_type"])
                    if raw.get("answer_type") is not None
                    else None
                ),
            )
            strict_response_reparsed += 1
            if _normalise_answer(extracted_answer) != _normalise_answer(
                raw.get("extracted_answer", raw.get("answer"))
            ):
                reparsed_answer_disagreements += 1
        else:
            extracted_answer = raw.get("extracted_answer", raw.get("answer"))
            legacy_answer_fallback += 1
        slim = {
            "sample_id": sample_id,
            "question_id": question_id,
            "generation_index": generation_index,
            "dataset_name": raw.get("dataset_name"),
            "correct_answer": raw.get("correct_answer"),
            "extracted_answer": extracted_answer,
            "confidence": raw.get("confidence"),
            "confidence_valid": raw.get("confidence_valid"),
            "yes_token_found_top20": raw.get("yes_token_found_top20"),
        }
        for field in OPTIONAL_METHOD_FIELDS:
            if field in raw:
                slim[field] = raw[field]
        signature = (
            question_id,
            generation_index,
            _normalise_answer(slim["correct_answer"]),
            _normalise_answer(slim["extracted_answer"]),
            _confidence(slim),
            tuple(
                (field, repr(slim.get(field)))
                for field in OPTIONAL_METHOD_FIELDS
                if field in slim
            ),
        )
        if sample_id in seen_samples:
            if seen_samples[sample_id] != signature:
                raise ValueError(f"Conflicting duplicate sample_id: {sample_id}")
            continue
        seen_samples[sample_id] = signature
        gold = _normalise_answer(slim["correct_answer"])
        if question_id in gold_by_question and gold_by_question[question_id] != gold:
            raise ValueError(f"Conflicting gold answers for question {question_id}")
        gold_by_question[question_id] = gold
        grouped[question_id].append(slim)
        if _answer(slim) is None:
            invalid_answers += 1
        if _confidence(slim) is None:
            missing_confidence += 1
        if slim.get("yes_token_found_top20") is False:
            yes_missing += 1

    duplicate_generation_indices = 0
    for question_id, question_records in grouped.items():
        question_records.sort(key=lambda item: (item["generation_index"], item["sample_id"]))
        indices = [record["generation_index"] for record in question_records]
        duplicate_generation_indices += len(indices) - len(set(indices))
    diagnostics = {
        "input_records": total_input,
        "unique_samples": len(seen_samples),
        "duplicate_samples_ignored": total_input - len(seen_samples),
        "duplicate_generation_indices": duplicate_generation_indices,
        "invalid_extracted_answers": invalid_answers,
        "missing_or_nonfinite_confidence": missing_confidence,
        "yes_token_missing_from_top20": yes_missing,
        "input_splits": sorted(input_splits),
        "input_datasets": sorted(input_datasets),
        "strict_response_reparsed_records": strict_response_reparsed,
        "legacy_extracted_answer_fallback_records": legacy_answer_fallback,
        "reparsed_answer_disagreements": reparsed_answer_disagreements,
        "answer_parser_versions": {
            name: parser_version(name) for name in sorted(input_datasets)
        },
    }
    return dict(grouped), gold_by_question, diagnostics


Predictor = Callable[[Sequence[Mapping[str, Any]]], tuple[str | None, int, str]]


def _score_method(
    question_ids: Sequence[str],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    gold_by_question: Mapping[str, str | None],
    predictor: Predictor,
) -> dict[str, Any]:
    correct = 0
    invalid = 0
    insufficient = 0
    invalid_gold = 0
    early_stops = 0
    used_total = 0
    used_observed_total = 0
    valid_samples = 0
    observed_questions = 0
    for question_id in question_ids:
        question_records = grouped.get(question_id, ())
        gold = gold_by_question.get(question_id)
        prediction, used, status = predictor(question_records)
        used_total += used
        valid_samples += sum(
            1 for record in question_records[:used] if _answer(record) is not None
        )
        if question_id in grouped:
            observed_questions += 1
            used_observed_total += used
        if gold is None:
            invalid_gold += 1
        if prediction is None:
            invalid += 1
        if status == "insufficient_samples":
            insufficient += 1
        if status == "early_stop":
            early_stops += 1
        if prediction is not None and gold is not None and prediction == gold:
            correct += 1
    total = len(question_ids)
    invalid_samples = used_total - valid_samples
    actual_avg_samples = used_total / total if total else 0.0
    return {
        "questions_total": total,
        "questions_observed": observed_questions,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100.0 * correct / total if total else 0.0,
        # ``actual_avg_samples`` is the canonical v2 field.  Keep the old
        # spelling as an exact alias so existing table scripts remain usable.
        "actual_avg_samples": actual_avg_samples,
        "avg_samples_used": actual_avg_samples,
        "avg_samples_used_observed": (
            used_observed_total / observed_questions if observed_questions else 0.0
        ),
        "generated_samples": used_total,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "invalid_rate": invalid_samples / used_total if used_total else 0.0,
        "invalid_predictions": invalid,
        "insufficient_sample_questions": insufficient,
        "invalid_gold_questions": invalid_gold,
        "early_stop_questions": early_stops,
    }


def _method_runtime_metadata(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Annotate whether optional baseline scores were present in the input.

    This is deliberately data-driven: a report made from the ordinary v1
    confidence JSONL will say ``proxy`` for CISC/Self-Certainty/RASC, while a
    richer artifact carrying native fields is marked ``exact``.  Consumers can
    therefore distinguish a genuine baseline reproduction from a fair,
    explicitly labelled fallback run.
    """

    rows = [record for records in grouped.values() for record in records]
    metadata: dict[str, dict[str, Any]] = {
        method: dict(values) for method, values in METHOD_METADATA.items()
    }
    for method in ("CISC", "Self-Certainty", "RASC"):
        native_fields: set[str] = set()
        native_count = 0
        for record in rows:
            if method == "Self-Certainty":
                scalar_fields = METHOD_SCORE_FIELDS[method]
                scalar_value, scalar_field = _optional_score(record, scalar_fields)
                token_value = _self_certainty_from_tokens(record)
                if scalar_value is not None and scalar_field is not None:
                    native_fields.add(scalar_field)
                    native_count += 1
                elif token_value is not None:
                    native_fields.add("token_distributions")
                    native_count += 1
            else:
                value, field = _optional_score(record, METHOD_SCORE_FIELDS[method])
                if value is not None and field is not None:
                    native_fields.add(field)
                    native_count += 1
        metadata[method]["native_score_records"] = native_count
        metadata[method]["native_score_fields"] = sorted(native_fields)
        metadata[method]["implementation_status"] = (
            "exact" if native_count else "proxy"
        )
    return metadata


def _control_parameter(row: Mapping[str, Any]) -> float | int:
    """Return the actual dynamic-method control parameter from a curve row."""

    if row.get("parameter_type") == "window_size":
        return int(row["window_size"])
    return float(row["threshold"])


def _select_validation_budget_matches(
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    budget_targets: Sequence[int],
    *,
    require_at_or_below: bool = True,
) -> list[dict[str, Any]]:
    """Select parameters from validation curves without exceeding the target.

    Accuracy is intentionally absent from the selection key.  Among rows at
    or below the requested validation average, choose the closest one.  This
    makes the budget contract one-sided: a parameter whose validation cost is
    already above 16 can never be selected for a target of 16.
    """

    matches: list[dict[str, Any]] = []
    for target in budget_targets:
        for method in DYNAMIC_METHODS:
            feasible = (
                [
                    row
                    for row in curves[method]
                    if float(row["actual_avg_samples"]) <= float(target) + 1e-12
                ]
                if require_at_or_below
                else list(curves[method])
            )
            if not feasible:
                minimum = min(
                    float(row["actual_avg_samples"]) for row in curves[method]
                )
                raise ValueError(
                    f"No validation parameter for {method} satisfies average "
                    f"budget <= {target}; minimum observed average is {minimum:.6f}"
                )
            selected = min(
                feasible,
                key=lambda row: (
                    (
                        float(target) - float(row["actual_avg_samples"])
                        if require_at_or_below
                        else abs(float(target) - float(row["actual_avg_samples"]))
                    ),
                    float(_control_parameter(row)),
                ),
            )
            matches.append(
                {
                    **selected,
                    "row_type": "dynamic_budget_match",
                    "budget": target,
                    "budget_target": target,
                    "budget_gap": float(selected["actual_avg_samples"]) - target,
                    "selection_split": "validation",
                    "selection_rule": (
                        "closest validation actual average at or below target; "
                        "accuracy not used"
                    ),
                    "budget_compliant": (
                        float(selected["actual_avg_samples"]) <= float(target)
                    ),
                }
            )
    return matches


def build_threshold_calibration(
    validation_report: Mapping[str, Any],
    *,
    model_id: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Build a model/dataset-scoped threshold artifact from validation only."""

    if validation_report.get("evaluation_phase") != "validation":
        raise ValueError("Threshold calibration can only be built from validation")
    if not model_id.strip() or not dataset_name.strip():
        raise ValueError("model_id and dataset_name are required for calibration")
    selections = []
    for row in validation_report.get("dynamic_budget_matches", ()):
        selections.append(
            {
                "method": canonical_method_name(str(row["method"])),
                "budget_target": int(row["budget_target"]),
                "parameter_type": str(row["parameter_type"]),
                "threshold": row.get("threshold"),
                "window_size": row.get("window_size"),
                # The parameter was calibrated with this hard execution cap.
                # Persist it as provenance so a later test run cannot silently
                # revert to the diagnostic curve's larger max budget.
                "budget_cap": int(row.get("budget_cap", row["budget_target"])),
                # Keep a descriptive alias for consumers that distinguish the
                # calibration-side cap from the test-side execution cap.
                "validation_budget_cap": int(
                    row.get("budget_cap", row["budget_target"])
                ),
                "validation_actual_avg_samples": float(row["actual_avg_samples"]),
                "validation_valid_samples": int(row["valid_samples"]),
                "validation_invalid_rate": float(row["invalid_rate"]),
            }
        )
    expected = {
        (method, int(target))
        for method in DYNAMIC_METHODS
        for target in validation_report["config"]["budget_targets"]
    }
    observed = {
        (str(row["method"]), int(row["budget_target"])) for row in selections
    }
    if observed != expected:
        raise ValueError(
            "Validation report does not contain exactly one selection for every "
            f"method/target: expected={sorted(expected)}, observed={sorted(observed)}"
        )
    config = validation_report["config"]
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibrated_on": "validation",
        "model_id": model_id,
        "dataset_name": dataset_name,
        "protocol": (
            "parameters selected by validation sample cost only; each target "
            "uses a hard execution cap"
        ),
        "curve_max_budget": int(config["curve_max_budget"]),
        "dynamic_min_valid": int(config["dynamic_min_valid"]),
        "rasc_buffer_size": int(config["rasc_buffer_size"]),
        "budget_targets": [int(value) for value in config["budget_targets"]],
        "budget_cap_policy": "dynamic budget matches use their target as a hard cap",
        "selections": selections,
    }


def _validate_threshold_calibration(
    calibration: Mapping[str, Any],
    *,
    model_id: str,
    dataset_name: str,
    config: AggregateConfig,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate scope/protocol and index a persisted calibration artifact."""

    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported dynamic-threshold calibration schema")
    if calibration.get("calibrated_on") != "validation":
        raise ValueError("Test thresholds must have calibrated_on='validation'")
    if calibration.get("model_id") != model_id:
        raise ValueError(
            f"Threshold model mismatch: {calibration.get('model_id')!r} != {model_id!r}"
        )
    if calibration.get("dataset_name") != dataset_name:
        raise ValueError(
            "Threshold dataset mismatch: "
            f"{calibration.get('dataset_name')!r} != {dataset_name!r}"
        )
    expected_protocol = {
        "curve_max_budget": config.curve_max_budget,
        "dynamic_min_valid": config.dynamic_min_valid,
        "rasc_buffer_size": config.rasc_buffer_size,
        "budget_targets": list(config.budget_targets),
    }
    for key, expected_value in expected_protocol.items():
        if calibration.get(key) != expected_value:
            raise ValueError(
                f"Threshold protocol mismatch for {key}: "
                f"{calibration.get(key)!r} != {expected_value!r}"
            )
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in calibration.get("selections", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("Every threshold selection must be a JSON object")
        method = canonical_method_name(str(raw.get("method", "")))
        target = int(raw.get("budget_target"))
        key = (method, target)
        if key in indexed:
            raise ValueError(f"Duplicate threshold selection: {key}")
        if method not in DYNAMIC_METHODS or target not in config.budget_targets:
            raise ValueError(f"Unexpected threshold selection: {key}")
        expected_budget_cap = min(config.curve_max_budget, target)
        recorded_budget_cap = raw.get(
            "validation_budget_cap", raw.get("budget_cap")
        )
        if recorded_budget_cap is not None:
            try:
                recorded_budget_cap = int(recorded_budget_cap)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid validation budget cap for {key}: "
                    f"{recorded_budget_cap!r}"
                ) from exc
            if recorded_budget_cap != expected_budget_cap:
                raise ValueError(
                    f"Threshold budget-cap mismatch for {key}: "
                    f"{recorded_budget_cap} != {expected_budget_cap}"
                )
        parameter_type = raw.get("parameter_type")
        if method == "ESC":
            if parameter_type != "window_size":
                raise ValueError("ESC calibration must contain a window_size")
            parameter = int(raw.get("window_size"))
            if not 2 <= parameter <= config.curve_max_budget:
                raise ValueError(f"Invalid ESC window size: {parameter}")
        else:
            if parameter_type != "confidence_threshold":
                raise ValueError(f"{method} calibration must contain a threshold")
            parameter = float(raw.get("threshold"))
            if not math.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
                raise ValueError(f"Invalid {method} threshold: {parameter}")
        indexed[key] = {**dict(raw), "control_parameter": parameter}
    expected_keys = {
        (method, target)
        for method in DYNAMIC_METHODS
        for target in config.budget_targets
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            "Threshold artifact must contain exactly one selection per "
            f"method/target; missing={sorted(expected_keys - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected_keys)}"
        )
    return indexed


def _validate_evaluation_split(
    diagnostics: Mapping[str, Any], phase: str
) -> None:
    """Prevent validation/test role confusion in formal aggregation phases."""

    splits = set(diagnostics.get("input_splits", ()))
    if phase == "validation":
        if not splits or not splits <= {"validation", "valid", "val", "dev"}:
            raise ValueError(
                "Threshold calibration requires records explicitly labelled as a "
                f"validation/dev split; observed splits={sorted(splits)}"
            )
    elif phase == "test":
        if not splits or splits != {"test"}:
            raise ValueError(
                "Test aggregation requires records explicitly labelled split='test'; "
                f"observed splits={sorted(splits)}"
            )


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    config: AggregateConfig | None = None,
    expected_question_ids: Sequence[str] | None = None,
    expected_question_count: int | None = None,
    *,
    phase: str = "analysis",
    threshold_calibration: Mapping[str, Any] | None = None,
    model_id: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate confidence records entirely on CPU.

    ``expected_question_ids`` is the strongest strict-denominator contract.  If
    only a count is known, missing questions are represented as explicit empty
    questions and therefore score as incorrect rather than disappearing.
    """

    config = config or AggregateConfig()
    if phase not in EVALUATION_PHASES:
        raise ValueError(f"phase must be one of {EVALUATION_PHASES}, got {phase!r}")
    grouped, gold_by_question, diagnostics = _question_groups(records)
    _validate_evaluation_split(diagnostics, phase)
    calibration_index: dict[tuple[str, int], dict[str, Any]] | None = None
    if phase == "test":
        if threshold_calibration is None:
            raise ValueError(
                "Test aggregation requires a validation threshold artifact; "
                "test-time threshold selection is forbidden"
            )
        if not model_id or not dataset_name:
            raise ValueError("Test aggregation requires model_id and dataset_name")
        input_datasets = set(diagnostics.get("input_datasets", ()))
        if input_datasets != {dataset_name}:
            raise ValueError(
                f"Input dataset mismatch: records={sorted(input_datasets)}, "
                f"requested={dataset_name!r}"
            )
        calibration_index = _validate_threshold_calibration(
            threshold_calibration,
            model_id=model_id,
            dataset_name=dataset_name,
            config=config,
        )
    elif threshold_calibration is not None:
        raise ValueError("threshold_calibration is only valid in phase='test'")
    observed_ids = sorted(grouped)
    if expected_question_ids is None:
        question_ids = list(observed_ids)
    else:
        question_ids = list(dict.fromkeys(str(item) for item in expected_question_ids))
        unexpected = sorted(set(observed_ids) - set(question_ids))
        if unexpected:
            raise ValueError(
                f"Found {len(unexpected)} question IDs outside expected_question_ids"
            )
    if expected_question_count is not None:
        if expected_question_count < len(question_ids):
            raise ValueError(
                f"expected_question_count={expected_question_count} is smaller than "
                f"the {len(question_ids)} known questions"
            )
        missing_count = expected_question_count - len(question_ids)
        question_ids.extend(
            f"__missing_question_{index:08d}" for index in range(missing_count)
        )

    sample_counts = [len(grouped[question_id]) for question_id in observed_ids]
    diagnostics.update(
        {
            "questions_total_denominator": len(question_ids),
            "questions_observed": len(observed_ids),
            "questions_missing_entirely": len(question_ids) - len(observed_ids),
            "questions_without_valid_answer": sum(
                1
                for question_id in observed_ids
                if not any(_answer(record) is not None for record in grouped[question_id])
            ),
            "min_samples_per_observed_question": min(sample_counts) if sample_counts else 0,
            "max_samples_per_observed_question": max(sample_counts) if sample_counts else 0,
        }
    )

    fixed_rows: list[dict[str, Any]] = []
    for budget in config.budgets:
        for method in FIXED_METHODS:
            metrics = _score_method(
                question_ids,
                grouped,
                gold_by_question,
                lambda rows, method=method, budget=budget: _fixed_predict(
                    method,
                    rows,
                    budget,
                    config.cisc_temperature,
                    config.cisc_normalization,
                ),
            )
            fixed_rows.append(
                {
                    "row_type": "fixed_budget",
                    "method": method,
                    "budget": budget,
                    "budget_cap": budget,
                    "threshold": None,
                    **metrics,
                }
            )

    curves: dict[str, list[dict[str, Any]]] = {
        method: [] for method in DYNAMIC_METHODS
    }

    def score_dynamic(
        method: str,
        parameter: float | int,
        *,
        budget_cap: int | None = None,
    ) -> dict[str, Any]:
        # ``curve_max_budget`` controls how much of the diagnostic curve is
        # exposed.  A requested target budget is a separate execution
        # contract: both validation calibration and held-out test must use
        # the same hard cap, otherwise a threshold selected at <=16 on one
        # partition can consume 32 responses on another partition.
        execution_cap = config.curve_max_budget
        if budget_cap is not None:
            execution_cap = min(execution_cap, int(budget_cap))
        if execution_cap <= 0:
            raise ValueError("dynamic budget cap must be positive")
        metrics = _score_method(
            question_ids,
            grouped,
            gold_by_question,
            lambda rows, method=method, parameter=parameter: _dynamic_predict(
                method,
                rows,
                parameter,
                execution_cap,
                config.dynamic_min_valid,
                config.rasc_buffer_size,
            ),
        )
        return {
            "row_type": "threshold_curve",
            "method": method,
            "budget": None,
            "budget_cap": execution_cap,
            "threshold": parameter,
            "parameter_type": (
                "window_size" if method == "ESC" else "confidence_threshold"
            ),
            "window_size": int(parameter) if method == "ESC" else None,
            **metrics,
        }

    def dynamic_parameters(
        method: str, *, budget_cap: int | None = None
    ) -> Sequence[float | int]:
        """Return the control grid for a diagnostic or target-capped curve."""

        cap = config.curve_max_budget
        if budget_cap is not None:
            cap = min(cap, int(budget_cap))
        if method != "ESC":
            return config.thresholds
        sizes = config.esc_window_sizes or tuple(range(2, cap + 1))
        # A window larger than the execution cap is equivalent to the cap in
        # ``_dynamic_predict``.  Dropping those aliases keeps calibration
        # deterministic and prevents a parameter outside the requested budget
        # from being persisted accidentally.
        filtered = tuple(int(size) for size in sizes if int(size) <= cap)
        if filtered:
            return filtered
        # ``target=1`` is not a normal Table-2 setting, but retaining one
        # candidate makes the API total and lets the predictor clamp it
        # consistently instead of failing with an empty calibration curve.
        return (cap,)

    if phase == "test":
        assert calibration_index is not None
        budget_matches: list[dict[str, Any]] = []
        for target in config.budget_targets:
            for method in DYNAMIC_METHODS:
                calibration_row = calibration_index[(method, target)]
                selected = score_dynamic(
                    method,
                    calibration_row["control_parameter"],
                    budget_cap=target,
                )
                curves[method].append(selected)
                budget_matches.append(
                    {
                        **selected,
                        "row_type": "dynamic_budget_match",
                        "budget": target,
                        "budget_target": target,
                        "budget_gap": selected["actual_avg_samples"] - target,
                        "selection_split": "validation",
                        "selection_rule": "read fixed parameter from validation artifact",
                        "validation_actual_avg_samples": calibration_row.get(
                            "validation_actual_avg_samples"
                        ),
                        "budget_compliant": selected["actual_avg_samples"] <= target,
                    }
                )
    else:
        for method in DYNAMIC_METHODS:
            parameters = dynamic_parameters(method)
            for parameter in parameters:
                curves[method].append(score_dynamic(method, parameter))

        # Select each target from a curve evaluated with that target as the
        # hard cap.  The full ``curves`` above remain diagnostic-only; using
        # them for calibration would reintroduce the 32-sample overrun that
        # this protocol is designed to prevent.
        budget_matches = []
        for target in config.budget_targets:
            target_curves = {
                method: [
                    score_dynamic(method, parameter, budget_cap=target)
                    for parameter in dynamic_parameters(method, budget_cap=target)
                ]
                for method in DYNAMIC_METHODS
            }
            budget_matches.extend(
                _select_validation_budget_matches(
                    target_curves,
                    (target,),
                    require_at_or_below=phase == "validation",
                )
            )
        if phase == "analysis":
            for row in budget_matches:
                row["selection_split"] = "analysis"
                row["selection_rule"] = (
                    "diagnostic-only inline curve selection; not reportable as test"
                )

    return {
        "schema_version": 2,
        # Keep the report namespace and display order explicit.  This avoids
        # downstream table builders having to infer whether a ``CaTS-*``
        # spelling refers to an original baseline or to one of the trained
        # RelaCaTS methods.
        "evaluation_namespace": "RelaCaTS",
        "method_order": list(TABLE2_METHOD_ORDER),
        "protocol": "RelaCaTS-v2 unified evaluator without test-time relational views",
        "evaluation_phase": phase,
        "reportable": phase in {"validation", "test"},
        "model_id": model_id,
        "dataset_name": dataset_name,
        "config": asdict(config),
        "diagnostics": diagnostics,
        "method_metadata": _method_runtime_metadata(grouped),
        "fixed_budget_results": fixed_rows,
        "threshold_curves": curves,
        "dynamic_budget_matches": budget_matches,
    }


def _discover_confidence_files(inputs: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            files.append(path.resolve())
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        direct = path / "confidence.jsonl"
        if direct.is_file():
            files.append(direct.resolve())
            continue
        merged = sorted(path.rglob("confidence.jsonl"))
        if merged:
            files.extend(item.resolve() for item in merged)
            continue
        chunks = sorted(path.rglob("chunks/chunk-*.jsonl"))
        if chunks:
            raise ValueError(
                f"Confidence chunks exist below {path}, but no merged confidence.jsonl; "
                "the confidence stage did not finish, so refusing a partial report"
            )
        raise FileNotFoundError(f"No confidence JSONL artifacts below {path}")
    unique = list(dict.fromkeys(files))
    if not unique:
        raise ValueError("No confidence inputs provided")
    _validate_confidence_manifests(unique)
    return unique


def _validate_confidence_manifests(files: Sequence[Path]) -> None:
    """Reject incomplete or partially supplied confidence shard sets.

    A report made from one of two GPU shards would otherwise have a deceptively
    high accuracy because absent questions disappear from the denominator.  The
    confidence stage writes a manifest for every shard, so validate it before
    any CPU aggregation.  Legacy single-shard files without metadata remain
    supported.
    """

    shard_groups: dict[Path, tuple[int, set[int]]] = {}
    for path in files:
        artifact_dir = path.parent
        manifest_path = artifact_dir / "confidence_manifest.json"
        metadata_path = artifact_dir / "confidence_metadata.json"
        if not manifest_path.exists() and not metadata_path.exists():
            continue
        if not manifest_path.exists() or not metadata_path.exists():
            raise ValueError(
                f"Incomplete confidence artifact metadata beside {path}; "
                "rerun confidence calculation or provide a new output directory"
            )
        try:
            manifest = read_json(manifest_path)
            metadata = read_json(metadata_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid confidence metadata beside {path}") from exc
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            raise ValueError(f"Confidence metadata must be JSON objects: {path}")
        if manifest.get("complete") is not True:
            raise ValueError(f"Confidence manifest is incomplete: {manifest_path}")
        expected = manifest.get("expected_samples")
        actual = manifest.get("samples")
        if expected is not None and actual != expected:
            raise ValueError(
                f"Confidence sample count mismatch in {manifest_path}: "
                f"{actual} != {expected}"
            )
        # Do not trust a stale manifest alone: a truncated JSONL with an old
        # manifest would otherwise make missing questions disappear from the
        # denominator.  Recompute the count, question set, and producer digest.
        observed_samples = 0
        observed_questions: set[str] = set()
        observed_digest = hashlib.sha256()
        observed_sample_ids: set[str] = set()
        try:
            for record in read_jsonl(path):
                sample_id = str(record.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"record without sample_id in {path}")
                if sample_id in observed_sample_ids:
                    raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
                observed_sample_ids.add(sample_id)
                question_id = str(record.get("question_id", ""))
                if not question_id:
                    raise ValueError(f"record without question_id in {path}")
                observed_questions.add(question_id)
                observed_digest.update(sample_id.encode("utf-8"))
                observed_digest.update(b"\0")
                observed_samples += 1
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid confidence JSONL: {path}") from exc
        if actual is not None and observed_samples != actual:
            raise ValueError(
                f"Confidence file/manifest count mismatch in {path}: "
                f"{observed_samples} != {actual}"
            )
        if manifest.get("questions") is not None and len(observed_questions) != manifest["questions"]:
            raise ValueError(
                f"Confidence question count mismatch in {path}: "
                f"{len(observed_questions)} != {manifest['questions']}"
            )
        expected_questions = manifest.get("expected_questions")
        if (
            expected_questions is not None
            and manifest.get("questions") is not None
            and int(expected_questions) != int(manifest["questions"])
        ):
            raise ValueError(
                f"Confidence expected question count mismatch in {manifest_path}: "
                f"{manifest['questions']} != {expected_questions}"
            )
        digest = manifest.get("sample_id_sha256")
        if digest is not None and observed_digest.hexdigest() != digest:
            raise ValueError(f"Confidence sample digest mismatch in {path}")
        try:
            num_shards = int(metadata.get("num_shards", 1))
            shard_index = int(metadata.get("shard_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid shard metadata in {metadata_path}") from exc
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(f"Invalid shard identity in {metadata_path}")
        # The parent of shard-xxxxx-of-yyyyy is the natural group.  For a
        # single artifact generated with num_shards=1, grouping by itself
        # avoids imposing a sibling directory convention on legacy outputs.
        group = artifact_dir.parent.resolve() if num_shards > 1 else artifact_dir.resolve()
        if group in shard_groups:
            previous_n, indices = shard_groups[group]
            if previous_n != num_shards:
                raise ValueError(f"Conflicting num_shards under {group}")
            indices.add(shard_index)
        else:
            shard_groups[group] = (num_shards, {shard_index})

    for group, (num_shards, indices) in shard_groups.items():
        if num_shards > 1 and indices != set(range(num_shards)):
            missing = sorted(set(range(num_shards)) - indices)
            raise ValueError(
                f"Missing confidence shards under {group}: {missing}; "
                "do not report a partial denominator"
            )


def _manifest_expected_questions(files: Sequence[Path]) -> int | None:
    """Infer a strict question denominator from complete confidence shards."""

    groups: dict[Path, dict[str, Any]] = {}
    for path in files:
        artifact_dir = path.parent
        manifest_path = artifact_dir / "confidence_manifest.json"
        metadata_path = artifact_dir / "confidence_metadata.json"
        if not manifest_path.is_file() or not metadata_path.is_file():
            return None
        manifest = read_json(manifest_path)
        metadata = read_json(metadata_path)
        if not isinstance(manifest, dict) or not isinstance(metadata, dict):
            return None
        expected = manifest.get("expected_questions")
        if expected is None:
            return None
        num_shards = int(metadata.get("num_shards", 1))
        shard_index = int(metadata.get("shard_index", 0))
        group = artifact_dir.parent.resolve() if num_shards > 1 else artifact_dir.resolve()
        state = groups.setdefault(
            group,
            {
                "num_shards": num_shards,
                "indices": set(),
                "expected": {},
                "disjoint": True,
            },
        )
        state["indices"].add(shard_index)
        state["expected"][shard_index] = int(expected)
        state["disjoint"] = state["disjoint"] and bool(
            metadata.get("responses_already_sharded", False)
        )

    if not groups:
        return None
    total = 0
    for state in groups.values():
        expected_values = list(state["expected"].values())
        if state["num_shards"] > 1 and state["disjoint"]:
            total += sum(expected_values)
        else:
            # In the legacy modulo-by-sample protocol every confidence shard
            # contains every question, so the maximum is the correct count.
            total += max(expected_values)
    return total


def _iter_files(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        yield from read_jsonl(path)


def _flat_rows(report: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    yield from report["fixed_budget_results"]
    for rows in report["threshold_curves"].values():
        yield from rows
    yield from report["dynamic_budget_matches"]


CSV_COLUMNS = (
    "row_type",
    "method",
    "budget",
    "budget_target",
    "budget_cap",
    "threshold",
    "parameter_type",
    "window_size",
    "questions_total",
    "questions_observed",
    "correct",
    "accuracy",
    "accuracy_percent",
    "actual_avg_samples",
    "avg_samples_used",
    "avg_samples_used_observed",
    "generated_samples",
    "valid_samples",
    "invalid_samples",
    "invalid_rate",
    "invalid_predictions",
    "insufficient_sample_questions",
    "invalid_gold_questions",
    "early_stop_questions",
    "budget_gap",
    "budget_compliant",
    "selection_split",
    "selection_rule",
    "validation_actual_avg_samples",
)


def _markdown(report: Mapping[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    lines = [
        "# RelaCaTS-v2 evaluation",
        "",
        f"Evaluation phase: **{report.get('evaluation_phase', 'unknown')}**. ",
        f"Reportable: **{report.get('reportable', False)}**.",
        "",
        "Test-time relational transformations: **disabled** (original CaTS protocol).",
        "Method labels: original baselines (`SC`, `CISC`, `Self-Certainty`, `Best-of-N`, `ASC`, `ESC`, `RASC`) plus `RelaCaTS-SC`, `RelaCaTS-ES`, and `RelaCaTS-ASC`.",
        "",
        "## Coverage and invalid outputs",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key in (
        "questions_total_denominator",
        "questions_observed",
        "questions_missing_entirely",
        "unique_samples",
        "invalid_extracted_answers",
        "missing_or_nonfinite_confidence",
        "yes_token_missing_from_top20",
        "questions_without_valid_answer",
        "duplicate_samples_ignored",
        "duplicate_generation_indices",
        "strict_response_reparsed_records",
        "legacy_extracted_answer_fallback_records",
        "reparsed_answer_disagreements",
    ):
        lines.append(f"| {key} | {diagnostics.get(key, 0)} |")

    lines.extend(
        [
            "",
            "Accuracy always uses `questions_total_denominator`; invalid or missing "
            "questions are not dropped.",
            "",
        "## Fixed-budget results",
            "",
            "| Method | Budget | Accuracy | Actual avg | Valid samples | Invalid rate | Invalid predictions | Insufficient |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["fixed_budget_results"]:
        lines.append(
            f"| {row['method']} | {row['budget']} | "
            f"{row['accuracy_percent']:.2f}% | {row['actual_avg_samples']:.3f} | "
            f"{row['valid_samples']} | {100.0 * row['invalid_rate']:.2f}% | "
            f"{row['invalid_predictions']} | {row['insufficient_sample_questions']} |"
        )

    lines.extend(
        [
            "",
            "## Method implementation and score sources",
            "",
            "CISC, Self-Certainty, and RASC require optional native confidence, "
            "token-certainty, or reasoning-score fields.  When those fields are "
            "absent, this report uses calibrated P(Yes) only as an explicitly "
            "labelled proxy.",
            "",
            "| Method | Status | Native records | Native fields |",
            "|---|---|---:|---|",
        ]
    )
    for method in (*FIXED_METHODS, *DYNAMIC_METHODS):
        metadata = report.get("method_metadata", {}).get(method, {})
        native_fields = ", ".join(metadata.get("native_score_fields", [])) or "—"
        lines.append(
            f"| {method} | {metadata.get('implementation_status', 'unknown')} | "
            f"{metadata.get('native_score_records', '—')} | {native_fields} |"
        )

    lines.extend(
        [
            "",
            "## Dynamic methods at requested average budgets",
            "",
            "Validation selects each parameter by sample cost only, with the "
            "requested target as a hard execution cap. Test reports only consume "
            "the persisted model/dataset/method-specific parameter and the same "
            "cap; they never search a test curve. ESC uses an integer "
            "non-overlapping window size; other dynamic methods use a confidence "
            "threshold.",
            "",
            "| Method | Target | Threshold/window | Accuracy | Actual avg | Valid samples | Invalid rate | Gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["dynamic_budget_matches"]:
        parameter = (
            row.get("window_size")
            if row.get("parameter_type") == "window_size"
            else row.get("threshold")
        )
        if parameter is None:
            parameter = "—"
        lines.append(
            f"| {row['method']} | {row['budget_target']} | "
            f"{float(parameter):.4f} | {row['accuracy_percent']:.2f}% | "
            f"{row['actual_avg_samples']:.3f} | {row['valid_samples']} | "
            f"{100.0 * row['invalid_rate']:.2f}% | {row['budget_gap']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Threshold curves",
            "",
            "Default confidence threshold grid: `0.00` through `1.00` in steps of "
            "`0.01`; ESC additionally reports window sizes `2..curve_max_budget`.",
            "",
            "| Method | Threshold/window | Accuracy | Avg used | Early stops | Invalid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in DYNAMIC_METHODS:
        # ``run_aggregation`` canonicalizes legacy ``CaTS-*`` labels at the
        # report boundary, so look up either spelling while older in-memory
        # reports remain readable.
        canonical = canonical_method_name(method)
        rows = report["threshold_curves"].get(method)
        if rows is None:
            rows = report["threshold_curves"].get(canonical, [])
        for row in rows:
            parameter = (
                row.get("window_size")
                if row.get("parameter_type") == "window_size"
                else row.get("threshold")
            )
            if parameter is None:
                parameter = "—"
            lines.append(
                f"| {method} | {float(parameter):.4f} | "
                f"{row['accuracy_percent']:.2f}% | {row['avg_samples_used']:.3f} | "
                f"{row['early_stop_questions']} | {row['invalid_predictions']} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    # Keep the writer safe for callers that construct/restore a report
    # directly (rather than going through ``run_aggregation``).  Normalizing
    # twice is harmless and guarantees that every persisted format uses the
    # canonical RelaCaTS labels.
    report = canonicalize_report_methods(report)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "evaluation.json"
    csv_path = output / "evaluation.csv"
    markdown_path = output / "evaluation.md"
    atomic_write_json(json_path, report)

    temporary_csv = output / ".evaluation.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in _flat_rows(report):
            writer.writerow(row)
    temporary_csv.replace(csv_path)

    temporary_markdown = output / ".evaluation.md.tmp"
    temporary_markdown.write_text(_markdown(report), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def run_aggregation(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    config: AggregateConfig | None = None,
    expected_question_count: int | None = None,
    *,
    phase: str = "analysis",
    threshold_file: str | Path | None = None,
    model_id: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    files = _discover_confidence_files(inputs)
    if expected_question_count is None:
        expected_question_count = _manifest_expected_questions(files)
    config = config or AggregateConfig()
    threshold_path = (
        Path(threshold_file).expanduser().resolve()
        if threshold_file is not None
        else None
    )
    calibration: Mapping[str, Any] | None = None
    if phase == "test":
        if threshold_path is None or not threshold_path.is_file():
            raise ValueError(
                "Test aggregation requires an existing --threshold-file produced "
                "by a validation run"
            )
        calibration = read_json(threshold_path)
        if not isinstance(calibration, Mapping):
            raise ValueError("Threshold artifact must be a JSON object")
    elif phase == "validation":
        if threshold_path is None:
            raise ValueError("Validation aggregation requires --threshold-file output")
        if threshold_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite threshold calibration: {threshold_path}"
            )
    elif phase == "analysis":
        if threshold_path is not None:
            raise ValueError("Diagnostic analysis does not read or write thresholds")
    else:
        raise ValueError(f"phase must be one of {EVALUATION_PHASES}, got {phase!r}")
    report = evaluate_records(
        _iter_files(files),
        config=config,
        expected_question_count=expected_question_count,
        phase=phase,
        threshold_calibration=calibration,
        model_id=model_id,
        dataset_name=dataset_name,
    )
    # Normalize legacy internal labels at the report boundary.  This keeps
    # older confidence artifacts and callers that still pass ``CaTS-*``
    # aliases compatible, while every newly persisted JSON/CSV/Markdown report
    # exposes the unambiguous ``RelaCaTS-*`` names.
    report = canonicalize_report_methods(report)
    report["input_files"] = [str(path) for path in files]
    if phase == "validation":
        if not model_id or not dataset_name:
            raise ValueError("Validation aggregation requires model_id and dataset_name")
        observed_datasets = set(report["diagnostics"].get("input_datasets", ()))
        if observed_datasets != {dataset_name}:
            raise ValueError(
                f"Input dataset mismatch: records={sorted(observed_datasets)}, "
                f"requested={dataset_name!r}"
            )
        calibration_document = build_threshold_calibration(
            report, model_id=model_id, dataset_name=dataset_name
        )
        assert threshold_path is not None
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(threshold_path, calibration_document)
        report["threshold_calibration_file"] = str(threshold_path)
    elif phase == "test":
        assert threshold_path is not None
        report["threshold_calibration_file"] = str(threshold_path)
    output_path = Path(output_dir).expanduser().resolve()
    # Add output paths before serialising evaluation.json.  Previously these
    # fields were appended only after write_reports(), so the returned Python
    # object had them while the persisted JSON silently did not.
    report["output_files"] = {
        "json": str(output_path / "evaluation.json"),
        "csv": str(output_path / "evaluation.csv"),
        "markdown": str(output_path / "evaluation.md"),
    }
    write_reports(report, output_path)
    return report


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return values


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated float list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only Table-2 evaluation: SC/CISC/Self-Certainty/Best-of-N/"
            "ASC/ESC/RASC baselines plus RelaCaTS-SC/ES/ASC"
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="Confidence artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--phase",
        choices=EVALUATION_PHASES,
        default="test",
        help=(
            "validation writes --threshold-file; test only reads it; analysis "
            "is diagnostic and not a reportable test result"
        ),
    )
    parser.add_argument(
        "--threshold-file",
        help="Validation output or existing test-time threshold artifact",
    )
    parser.add_argument(
        "--model-id",
        help="Stable model tag used to scope validation thresholds",
    )
    parser.add_argument(
        "--dataset-name",
        help="Dataset tag used to scope validation thresholds",
    )
    parser.add_argument("--budgets", type=_parse_ints, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument(
        "--thresholds",
        type=_parse_floats,
        default=tuple(index / 100 for index in range(101)),
    )
    parser.add_argument("--curve-max-budget", type=int, default=32)
    parser.add_argument("--budget-targets", type=_parse_ints, default=(16,))
    parser.add_argument("--dynamic-min-valid", type=int, default=2)
    parser.add_argument("--rasc-buffer-size", type=int, default=5)
    parser.add_argument(
        "--esc-window-sizes",
        type=_parse_ints,
        default=(),
        help="ESC non-overlapping window sizes (default: every size 2..curve-max-budget)",
    )
    parser.add_argument("--cisc-temperature", type=float, default=1.0)
    parser.add_argument(
        "--cisc-normalization",
        choices=("softmax", "linear", "none"),
        default="softmax",
    )
    parser.add_argument(
        "--expected-questions",
        type=int,
        help="Strict denominator override; entirely missing questions count wrong",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AggregateConfig(
        budgets=tuple(args.budgets),
        thresholds=tuple(args.thresholds),
        curve_max_budget=args.curve_max_budget,
        budget_targets=tuple(args.budget_targets),
        dynamic_min_valid=args.dynamic_min_valid,
        rasc_buffer_size=args.rasc_buffer_size,
        esc_window_sizes=tuple(args.esc_window_sizes),
        cisc_temperature=args.cisc_temperature,
        cisc_normalization=args.cisc_normalization,
    )
    report = run_aggregation(
        args.input,
        args.output_dir,
        config=config,
        expected_question_count=args.expected_questions,
        phase=args.phase,
        threshold_file=args.threshold_file,
        model_id=args.model_id,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(report["output_files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
