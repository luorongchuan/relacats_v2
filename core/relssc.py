"""Pure RelaCaTS-v1 relational soft self-consistency (RelSSC).

V1 deliberately implements only equation (20) with relation reliability
``r_g = 1`` and dependency weight ``d_gi = 1``.  All valid samples from all
views share one confidence-weighted denominator; the function does not first
average within each view.  Invalid/canonicalization-failed answers are skipped
rather than being grouped into an artificial ``None`` answer class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

from .canonicalization import CanonicalizationResult, CanonicalizationStatus


class RelSSCError(ValueError):
    """Base class for invalid RelSSC input or configuration."""


class InvalidConfidenceError(RelSSCError):
    """Raised when a valid answer has a non-probability confidence."""


class UnsupportedV1WeightError(RelSSCError):
    """Raised when v1 receives relation/dependency weights other than one."""


class ZeroTotalWeightError(RelSSCError):
    """Raised when no positive confidence mass remains after invalid filtering."""


class ZeroWeightPolicy(str, Enum):
    """How a question with zero valid confidence mass is handled."""

    RAISE = "raise"
    SKIP = "skip"


@dataclass(frozen=True)
class RelSSCSample:
    """Minimal per-response input to RelSSC."""

    canonicalized_answer: str | None
    confidence: float | None
    relation_id: str | None = None
    valid: bool = True
    relation_weight: float = 1.0
    dependency_weight: float = 1.0


@dataclass(frozen=True)
class RelSSCResult:
    """Question-level RelSSC scores and aligned per-response targets."""

    scores: Mapping[str, float]
    weighted_support: Mapping[str, float]
    targets: tuple[float | None, ...]
    total_weight: float
    valid_sample_count: int
    invalid_sample_count: int
    positive_weight_sample_count: int
    defined: bool
    reason: str | None = None

    def score(self, canonicalized_answer: str) -> float | None:
        """Return RelSSC(answer), or ``None`` when the question was skipped."""

        if not self.defined:
            return None
        return self.scores.get(str(canonicalized_answer).strip())

    @property
    def top_answer(self) -> str | None:
        """Highest-support answer with deterministic lexical tie breaking."""

        if not self.defined or not self.scores:
            return None
        return min(self.scores, key=lambda answer: (-self.scores[answer], answer))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "weighted_support": dict(self.weighted_support),
            "targets": list(self.targets),
            "total_weight": self.total_weight,
            "valid_sample_count": self.valid_sample_count,
            "invalid_sample_count": self.invalid_sample_count,
            "positive_weight_sample_count": self.positive_weight_sample_count,
            "defined": self.defined,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _PreparedSample:
    answer: str | None
    confidence: float | None
    valid: bool
    relation_weight: float
    dependency_weight: float


_INVALID_SENTINELS = {"", "none", "null", "n/a", "na", "invalid", "<invalid>"}


def _as_finite_weight(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RelSSCError(f"sample {index} {name} must be numeric; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RelSSCError(
            f"sample {index} {name} must be finite and non-negative; got {value!r}"
        )
    return result


def _canonical_answer_and_validity(value: Any) -> tuple[str | None, bool]:
    if isinstance(value, CanonicalizationResult):
        if not value.valid or value.canonicalized_answer is None:
            return None, False
        return value.canonicalized_answer, True
    if value is None or isinstance(value, bool):
        return None, False
    answer = str(value).strip()
    if answer.lower() in _INVALID_SENTINELS:
        return None, False
    if len(answer) == 1 and answer.upper() in "ABCDE":
        answer = answer.upper()
    return answer, True


def _mapping_value(
    record: Mapping[str, Any], names: Sequence[str], default: Any
) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def _coerce_valid_flag(value: Any, index: int) -> bool:
    """Parse serialized validity flags without Python's ``bool('false')`` trap."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, Real) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    token = str(value).strip().lower()
    if token in {"true", "yes", "1", "valid"}:
        return True
    if token in {"false", "no", "0", "invalid", "none", "null"}:
        return False
    raise RelSSCError(
        f"sample {index} validity flag must be boolean-like; got {value!r}"
    )


def _prepare_sample(sample: RelSSCSample | Mapping[str, Any], index: int) -> _PreparedSample:
    if isinstance(sample, RelSSCSample):
        answer_value: Any = sample.canonicalized_answer
        confidence_value: Any = sample.confidence
        explicit_valid = sample.valid
        relation_weight_value: Any = sample.relation_weight
        dependency_weight_value: Any = sample.dependency_weight
        canonicalization_status: Any = None
    elif isinstance(sample, Mapping):
        answer_value = _mapping_value(
            sample,
            ("canonicalized_answer", "canonical_answer"),
            None,
        )
        confidence_value = sample.get("confidence")
        explicit_valid = _coerce_valid_flag(
            _mapping_value(sample, ("is_valid_answer", "valid"), True), index
        )
        relation_weight_value = _mapping_value(
            sample,
            ("relation_weight", "relation_reliability", "r_g", "r"),
            1.0,
        )
        dependency_weight_value = _mapping_value(
            sample,
            ("dependency_weight", "d_gi", "d"),
            1.0,
        )
        canonicalization_status = sample.get("canonicalization_status")
    else:
        raise RelSSCError(
            f"sample {index} must be RelSSCSample or a mapping; "
            f"got {type(sample).__name__}"
        )

    relation_weight = _as_finite_weight(
        relation_weight_value, "relation_weight", index
    )
    dependency_weight = _as_finite_weight(
        dependency_weight_value, "dependency_weight", index
    )
    answer, answer_valid = _canonical_answer_and_validity(answer_value)

    if isinstance(canonicalization_status, CanonicalizationStatus):
        status_valid = canonicalization_status is CanonicalizationStatus.VALID
    elif canonicalization_status is None:
        status_valid = True
    else:
        status_valid = str(canonicalization_status).strip().lower() == "valid"
    valid = explicit_valid and answer_valid and status_valid
    if not valid:
        return _PreparedSample(
            answer=None,
            confidence=None,
            valid=False,
            relation_weight=relation_weight,
            dependency_weight=dependency_weight,
        )

    if isinstance(confidence_value, bool) or not isinstance(confidence_value, Real):
        raise InvalidConfidenceError(
            f"sample {index} confidence must be numeric for a valid answer; "
            f"got {confidence_value!r}"
        )
    confidence = float(confidence_value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidConfidenceError(
            f"sample {index} confidence must be a finite probability in [0, 1]; "
            f"got {confidence_value!r}"
        )
    return _PreparedSample(
        answer=answer,
        confidence=confidence,
        valid=True,
        relation_weight=relation_weight,
        dependency_weight=dependency_weight,
    )


def _coerce_zero_weight_policy(
    policy: ZeroWeightPolicy | str,
) -> ZeroWeightPolicy:
    if isinstance(policy, ZeroWeightPolicy):
        return policy
    try:
        return ZeroWeightPolicy(str(policy).strip().lower())
    except ValueError as exc:
        raise RelSSCError(
            f"zero_weight_policy must be 'raise' or 'skip'; got {policy!r}"
        ) from exc


def compute_relssc(
    samples: Iterable[RelSSCSample | Mapping[str, Any]],
    *,
    zero_weight_policy: ZeroWeightPolicy | str = ZeroWeightPolicy.RAISE,
    enforce_v1_weights: bool = True,
) -> RelSSCResult:
    """Compute confidence-weighted RelSSC over *all* relational samples.

    For each answer ``a`` this computes::

        sum_{g,i} confidence[g,i] * I(canonical[g,i] == a)
        -----------------------------------------------------
                   sum_{g,i} confidence[g,i]

    after excluding invalid answers.  V1 requires every ``r_g`` and ``d_gi``
    to equal one.  Set ``zero_weight_policy='skip'`` to return an explicitly
    undefined result (all targets ``None``); the default raises so callers
    cannot accidentally train on fabricated zero labels.
    """

    if isinstance(samples, (str, bytes, Mapping)):
        raise RelSSCError("samples must be an iterable of sample records")
    raw_samples = tuple(samples)
    prepared = tuple(_prepare_sample(sample, index) for index, sample in enumerate(raw_samples))

    if enforce_v1_weights:
        for index, sample in enumerate(prepared):
            if not math.isclose(sample.relation_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise UnsupportedV1WeightError(
                    f"sample {index} relation_weight={sample.relation_weight}; "
                    "RelaCaTS-v1 requires r_g=1"
                )
            if not math.isclose(sample.dependency_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise UnsupportedV1WeightError(
                    f"sample {index} dependency_weight={sample.dependency_weight}; "
                    "RelaCaTS-v1 requires d_gi=1"
                )

    weighted_terms: dict[str, list[float]] = {}
    all_weights: list[float] = []
    valid_count = 0
    invalid_count = 0
    positive_count = 0
    for sample in prepared:
        if not sample.valid:
            invalid_count += 1
            continue
        assert sample.answer is not None and sample.confidence is not None
        valid_count += 1
        weight = (
            sample.confidence * sample.relation_weight * sample.dependency_weight
        )
        if weight > 0:
            positive_count += 1
        all_weights.append(weight)
        weighted_terms.setdefault(sample.answer, []).append(weight)

    total_weight = math.fsum(all_weights)
    policy = _coerce_zero_weight_policy(zero_weight_policy)
    if total_weight <= 0:
        reason = (
            "no positive confidence mass remains after excluding invalid answers; "
            "skip this original question"
        )
        if policy is ZeroWeightPolicy.RAISE:
            raise ZeroTotalWeightError(reason)
        return RelSSCResult(
            scores={},
            weighted_support={answer: math.fsum(terms) for answer, terms in weighted_terms.items()},
            targets=tuple(None for _ in prepared),
            total_weight=0.0,
            valid_sample_count=valid_count,
            invalid_sample_count=invalid_count,
            positive_weight_sample_count=positive_count,
            defined=False,
            reason=reason,
        )

    support = {
        answer: math.fsum(terms) for answer, terms in weighted_terms.items()
    }
    scores = {answer: weight / total_weight for answer, weight in support.items()}
    targets = tuple(
        scores[sample.answer] if sample.valid and sample.answer is not None else None
        for sample in prepared
    )
    return RelSSCResult(
        scores=scores,
        weighted_support=support,
        targets=targets,
        total_weight=total_weight,
        valid_sample_count=valid_count,
        invalid_sample_count=invalid_count,
        positive_weight_sample_count=positive_count,
        defined=True,
    )


def attach_relssc_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    target_field: str = "relssc",
    consistency_field: str = "relational_consistency",
    zero_weight_policy: ZeroWeightPolicy | str = ZeroWeightPolicy.RAISE,
) -> tuple[dict[str, Any], ...]:
    """Return copied records with per-response RelSSC targets attached.

    The input mappings are not modified.  Invalid response records receive
    ``None`` and should be skipped by the training dataset builder.
    """

    result = compute_relssc(records, zero_weight_policy=zero_weight_policy)
    output: list[dict[str, Any]] = []
    for record, target in zip(records, result.targets):
        copied = dict(record)
        copied[target_field] = target
        copied[consistency_field] = target
        output.append(copied)
    return tuple(output)
