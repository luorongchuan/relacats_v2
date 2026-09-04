"""Answer normalization and relational canonicalization.

This module is intentionally independent of model inference.  It consumes an
already extracted answer and relation metadata, then either returns a valid
answer in the *original* answer space or an explicit invalid state.  Invalid
extractions are never turned into a vote for a sentinel such as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence

from .relational_views import (
    DEFAULT_OPTION_LABELS,
    InvalidPermutationError,
    OptionPermutation,
)


class CanonicalizationStatus(str, Enum):
    """Machine-readable outcome of normalization/canonicalization."""

    VALID = "valid"
    MISSING_ANSWER = "missing_answer"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_ANSWER_SPACE = "out_of_answer_space"
    UNSUPPORTED_RELATION = "unsupported_relation"


class InvalidAnswerError(ValueError):
    """Raised in strict mode when an extracted answer cannot be canonicalized."""

    def __init__(self, result: "CanonicalizationResult") -> None:
        self.result = result
        super().__init__(
            f"cannot canonicalize answer {result.raw_answer!r}: "
            f"{result.status.value}: {result.reason}"
        )


@dataclass(frozen=True)
class AnswerNormalizationResult:
    """Normalized answer or an explicit non-voting invalid state."""

    raw_answer: Any
    normalized_answer: str | None
    status: CanonicalizationStatus
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is CanonicalizationStatus.VALID


@dataclass(frozen=True)
class CanonicalizationResult:
    """Result of mapping a transformed answer to the original answer space."""

    raw_answer: Any
    normalized_transformed_answer: str | None
    canonicalized_answer: str | None
    status: CanonicalizationStatus
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is CanonicalizationStatus.VALID

    def require_valid(self) -> str:
        """Return the canonical answer or raise ``InvalidAnswerError``."""

        if not self.valid or self.canonicalized_answer is None:
            raise InvalidAnswerError(self)
        return self.canonicalized_answer

    def to_record_fields(self) -> dict[str, Any]:
        """Fields suitable for one generated relational-sample record."""

        return {
            "extracted_answer": self.normalized_transformed_answer,
            "canonicalized_answer": self.canonicalized_answer,
            "canonicalization_status": self.status.value,
            "canonicalization_error": self.reason,
            "is_valid_answer": self.valid,
        }


_PREFIX_RE = re.compile(
    r"^\s*(?:(?:final\s+)?answer|option|choice)\s*(?:is\s*)?[:=]?\s*",
    flags=re.IGNORECASE,
)
_OPTION_TOKEN_RE = re.compile(r"^[A-Za-z]$")
_OPTION_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _strip_answer_wrapper(value: str) -> str:
    candidate = _PREFIX_RE.sub("", value, count=1).strip()
    if (
        len(candidate) >= 2
        and (candidate[0], candidate[-1]) in {("(", ")"), ("[", "]")}
    ):
        candidate = candidate[1:-1].strip()
    # Extraction utilities commonly retain a final full stop or colon.
    return candidate.rstrip(".:;").strip()


def _option_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label).strip().upper() for label in labels)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"option labels must be a non-empty unique sequence; got {labels!r}")
    if any(label not in DEFAULT_OPTION_LABELS for label in normalized):
        raise ValueError(f"option labels must be drawn from A--E; got {normalized!r}")
    return normalized


def normalize_option_answer(
    answer: Any,
    *,
    labels: Sequence[str] = DEFAULT_OPTION_LABELS,
    numeric_base: int = 1,
) -> AnswerNormalizationResult:
    """Normalize a multiple-choice label, including numeric option indices.

    Letters are normalized to upper-case A--E.  Integer-like answers are
    interpreted as option indices (1->A by default; set ``numeric_base=0`` for
    zero-based datasets).  The function deliberately expects an *extracted*
    answer and will not fish a letter out of arbitrary free-form text.
    """

    chosen_labels = _option_labels(labels)
    if numeric_base not in (0, 1):
        raise ValueError(f"numeric_base must be 0 or 1; got {numeric_base!r}")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.MISSING_ANSWER,
            reason="answer extraction returned no value",
        )
    if isinstance(answer, bool):
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.INVALID_FORMAT,
            reason="boolean values are not option answers",
        )

    if isinstance(answer, int):
        token = str(answer)
    elif isinstance(answer, float):
        if not math.isfinite(answer) or not answer.is_integer():
            token = ""
        else:
            token = str(int(answer))
    elif isinstance(answer, Decimal):
        token = str(answer) if answer.is_finite() and answer == answer.to_integral() else ""
    elif isinstance(answer, str):
        token = _strip_answer_wrapper(answer)
    else:
        token = ""

    if _OPTION_TOKEN_RE.fullmatch(token):
        normalized = token.upper()
        if normalized in chosen_labels:
            return AnswerNormalizationResult(
                raw_answer=answer,
                normalized_answer=normalized,
                status=CanonicalizationStatus.VALID,
            )
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.OUT_OF_ANSWER_SPACE,
            reason=f"option {normalized!r} is outside {chosen_labels!r}",
        )

    if _OPTION_NUMBER_RE.fullmatch(token):
        index = int(Decimal(token)) - numeric_base
        if 0 <= index < len(chosen_labels):
            return AnswerNormalizationResult(
                raw_answer=answer,
                normalized_answer=chosen_labels[index],
                status=CanonicalizationStatus.VALID,
            )
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.OUT_OF_ANSWER_SPACE,
            reason=(
                f"numeric option {token!r} is outside the {numeric_base}-based "
                f"range for {len(chosen_labels)} options"
            ),
        )

    return AnswerNormalizationResult(
        raw_answer=answer,
        normalized_answer=None,
        status=CanonicalizationStatus.INVALID_FORMAT,
        reason="expected one extracted option letter or integer option index",
    )


def normalize_numeric_answer(answer: Any) -> AnswerNormalizationResult:
    """Normalize a scalar numeric answer to a stable decimal string.

    Examples: ``"Answer: $1,000.00" -> "1000"`` and ``-0.0 -> "0"``.
    Fractions, percentages, units, and free-form text are intentionally not
    guessed; dataset-specific extraction should handle them first.
    """

    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.MISSING_ANSWER,
            reason="answer extraction returned no value",
        )
    if isinstance(answer, bool):
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.INVALID_FORMAT,
            reason="boolean values are not numeric answers",
        )

    if isinstance(answer, Decimal):
        decimal_value = answer
    elif isinstance(answer, (int, float)):
        if isinstance(answer, float) and not math.isfinite(answer):
            decimal_value = Decimal("NaN")
        else:
            # str avoids importing the binary floating-point tail.
            decimal_value = Decimal(str(answer))
    elif isinstance(answer, str):
        token = _strip_answer_wrapper(answer).replace(",", "").replace("$", "")
        if not _NUMBER_RE.fullmatch(token):
            return AnswerNormalizationResult(
                raw_answer=answer,
                normalized_answer=None,
                status=CanonicalizationStatus.INVALID_FORMAT,
                reason="expected one extracted scalar number",
            )
        try:
            decimal_value = Decimal(token)
        except InvalidOperation:
            decimal_value = Decimal("NaN")
    else:
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.INVALID_FORMAT,
            reason=f"unsupported numeric answer type {type(answer).__name__}",
        )

    if not decimal_value.is_finite():
        return AnswerNormalizationResult(
            raw_answer=answer,
            normalized_answer=None,
            status=CanonicalizationStatus.INVALID_FORMAT,
            reason="numeric answer must be finite",
        )
    if decimal_value == 0:
        normalized = "0"
    else:
        normalized = format(decimal_value.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
    return AnswerNormalizationResult(
        raw_answer=answer,
        normalized_answer=normalized,
        status=CanonicalizationStatus.VALID,
    )


def normalize_answer(
    answer: Any,
    *,
    answer_type: str = "option",
    labels: Sequence[str] = DEFAULT_OPTION_LABELS,
    numeric_base: int = 1,
) -> AnswerNormalizationResult:
    """Dispatch to option-label or scalar-number normalization."""

    normalized_type = str(answer_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_type in {"option", "option_letter", "multiple_choice", "letter"}:
        return normalize_option_answer(
            answer, labels=labels, numeric_base=numeric_base
        )
    if normalized_type in {"number", "numeric", "scalar"}:
        return normalize_numeric_answer(answer)
    raise ValueError(f"unsupported answer_type {answer_type!r}")


def _invalid_canonicalization(
    normalization: AnswerNormalizationResult,
) -> CanonicalizationResult:
    return CanonicalizationResult(
        raw_answer=normalization.raw_answer,
        normalized_transformed_answer=normalization.normalized_answer,
        canonicalized_answer=None,
        status=normalization.status,
        reason=normalization.reason,
    )


def _finish(
    result: CanonicalizationResult, strict: bool
) -> CanonicalizationResult:
    if strict and not result.valid:
        raise InvalidAnswerError(result)
    return result


def _resolve_option_permutation(
    relation: OptionPermutation | Mapping[str, Any] | None,
    labels: Sequence[str],
) -> OptionPermutation:
    if isinstance(relation, OptionPermutation):
        return relation
    if relation is None:
        return OptionPermutation.identity(len(tuple(labels)))
    if not isinstance(relation, Mapping):
        raise InvalidPermutationError(
            "option relation must be OptionPermutation, mapping metadata, or None"
        )
    relation_type = str(relation.get("relation_type", "option_permutation")).lower()
    if relation_type in {"invariant", "none"}:
        return OptionPermutation.identity(len(tuple(labels)))
    if relation_type == "identity" and not any(
        key in relation
        for key in ("permutation", "forward_mapping", "inverse_permutation", "inverse_mapping")
    ):
        metadata = dict(relation)
        metadata["option_labels"] = list(labels)
        return OptionPermutation.from_metadata(metadata)
    if relation_type not in {"identity", "option_permutation"}:
        raise InvalidPermutationError(
            f"unsupported option relation_type {relation_type!r}"
        )
    return OptionPermutation.from_metadata(relation)


def canonicalize_answer(
    transformed_answer: Any,
    relation: OptionPermutation | Mapping[str, Any] | None = None,
    *,
    answer_type: str = "option",
    labels: Sequence[str] = DEFAULT_OPTION_LABELS,
    numeric_base: int = 1,
    strict: bool = False,
) -> CanonicalizationResult:
    """Canonicalize an extracted transformed answer into original space.

    For option permutations this *always* applies the inverse mapping
    (transformed -> original).  For scalar numeric answers only identity or
    invariant relations are supported in v1.
    """

    normalized_type = str(answer_type).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_type in {"option", "option_letter", "multiple_choice", "letter"}:
        permutation = _resolve_option_permutation(relation, labels)
        normalization = normalize_option_answer(
            transformed_answer,
            labels=permutation.labels,
            numeric_base=numeric_base,
        )
        if not normalization.valid:
            return _finish(_invalid_canonicalization(normalization), strict)
        assert normalization.normalized_answer is not None
        canonicalized = permutation.inverse_answer(normalization.normalized_answer)
        return CanonicalizationResult(
            raw_answer=transformed_answer,
            normalized_transformed_answer=normalization.normalized_answer,
            canonicalized_answer=canonicalized,
            status=CanonicalizationStatus.VALID,
        )

    if normalized_type in {"number", "numeric", "scalar"}:
        relation_type = "identity"
        if isinstance(relation, Mapping):
            relation_type = str(relation.get("relation_type", "identity")).lower()
        elif isinstance(relation, OptionPermutation):
            relation_type = "option_permutation"
        if relation_type not in {"identity", "invariant", "none"}:
            result = CanonicalizationResult(
                raw_answer=transformed_answer,
                normalized_transformed_answer=None,
                canonicalized_answer=None,
                status=CanonicalizationStatus.UNSUPPORTED_RELATION,
                reason=(
                    "RelaCaTS-v1 only canonicalizes scalar numbers under "
                    "identity/invariant relations"
                ),
            )
            return _finish(result, strict)
        normalization = normalize_numeric_answer(transformed_answer)
        if not normalization.valid:
            return _finish(_invalid_canonicalization(normalization), strict)
        return CanonicalizationResult(
            raw_answer=transformed_answer,
            normalized_transformed_answer=normalization.normalized_answer,
            canonicalized_answer=normalization.normalized_answer,
            status=CanonicalizationStatus.VALID,
        )

    raise ValueError(f"unsupported answer_type {answer_type!r}")


def canonicalize_answer_value(
    transformed_answer: Any,
    relation: OptionPermutation | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> str | None:
    """Convenience wrapper returning the canonical value or ``None``.

    Callers that need auditability should retain the full
    ``CanonicalizationResult`` instead.
    """

    return canonicalize_answer(transformed_answer, relation, **kwargs).canonicalized_answer
