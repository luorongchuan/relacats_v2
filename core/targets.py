"""Gold-free confidence targets for RelaCaTS v2.

The calibration target and the criterion used to admit examples to the
generation (causal-LM) task are deliberately separate:

* calibration may use SSC, RelSSC replacement, or the v2 residual target;
* generation examples are *always* selected using the original-view SSC.

Keeping these calculations in a small, pure module makes that distinction
testable without loading a model or tokenizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence


TARGET_MODES = frozenset({"ssc", "relssc_replace", "residual"})
DEFAULT_TARGET_MODE = "residual"
DEFAULT_LAMBDA_REL = 0.5

_INVALID_ANSWERS = {"", "none", "null", "n/a", "na", "invalid", "<invalid>"}


@dataclass(frozen=True)
class SSCTargetContext:
    """Question-level inputs required by the three v2 target modes."""

    scores: Mapping[str, float]
    valid_identity_samples: int
    invalid_identity_samples: int
    relation_valid_ratio: float
    valid_relation_samples: int
    total_relation_samples: int
    defined: bool

    def score(self, answer: Any) -> float | None:
        if not self.defined:
            return None
        canonical = _canonical_answer(answer)
        if canonical is None:
            return None
        return float(self.scores.get(canonical, 0.0))


def _canonical_answer(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    answer = str(value).strip()
    if answer.lower() in _INVALID_ANSWERS:
        return None
    if len(answer) == 1 and answer.upper() in "ABCDE":
        answer = answer.upper()
    return answer


def _boolean_like(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
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
    raise ValueError(f"validity flag must be boolean-like; got {value!r}")


def _is_valid_answer(record: Mapping[str, Any]) -> bool:
    answer = _canonical_answer(
        record.get("canonicalized_answer", record.get("canonical_answer"))
    )
    if answer is None:
        return False
    if not _boolean_like(
        record.get("is_valid_answer", record.get("valid")), default=True
    ):
        return False
    status = record.get("canonicalization_status")
    status_value = getattr(status, "value", status)
    return status_value is None or str(status_value).strip().lower() == "valid"


def _is_identity(record: Mapping[str, Any]) -> bool:
    """Recognize the original question view without guessing transformed views."""

    relation_type = record.get("relation_type")
    if relation_type is not None:
        return str(relation_type).strip().lower() == "identity"
    if record.get("view_index") is not None:
        try:
            return int(record["view_index"]) == 0
        except (TypeError, ValueError):
            return False
    if record.get("relation_id") is not None:
        return str(record["relation_id"]).strip().lower() == "g0"
    # Legacy SSC-only records have no relation metadata and are, by
    # construction, original-view samples.
    return True


def _is_duplicate_view(record: Mapping[str, Any]) -> bool:
    return _boolean_like(record.get("is_duplicate_view"), default=False)


def compute_ssc_target_context(
    records: Sequence[Mapping[str, Any]],
) -> SSCTargetContext:
    """Compute count-based SSC and relation parse-validity for one question.

    SSC is based only on valid responses from the identity/original view.  The
    relation validity ratio uses non-identity, non-duplicate relation samples.
    With no relational view (for example GSM8K/SVAMP 1x32), the ratio is zero,
    so the residual target reduces exactly to SSC.
    """

    identity = [record for record in records if _is_identity(record)]
    relation = [
        record
        for record in records
        if not _is_identity(record) and not _is_duplicate_view(record)
    ]
    counts = Counter(
        _canonical_answer(
            record.get("canonicalized_answer", record.get("canonical_answer"))
        )
        for record in identity
        if _is_valid_answer(record)
    )
    counts.pop(None, None)
    valid_identity = sum(counts.values())
    invalid_identity = len(identity) - valid_identity
    scores = (
        {answer: count / valid_identity for answer, count in counts.items()}
        if valid_identity
        else {}
    )
    valid_relation = sum(_is_valid_answer(record) for record in relation)
    relation_ratio = valid_relation / len(relation) if relation else 0.0
    return SSCTargetContext(
        scores=scores,
        valid_identity_samples=valid_identity,
        invalid_identity_samples=invalid_identity,
        relation_valid_ratio=relation_ratio,
        valid_relation_samples=valid_relation,
        total_relation_samples=len(relation),
        defined=valid_identity > 0,
    )


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0,1]; got {value!r}")
    return result


def resolve_confidence_target(
    *,
    ssc: Any,
    relssc: Any,
    relation_valid_ratio: Any,
    target_mode: str = DEFAULT_TARGET_MODE,
    lambda_rel: float = DEFAULT_LAMBDA_REL,
) -> float:
    """Resolve one calibration target according to the requested v2 mode."""

    mode = str(target_mode).strip().lower()
    if mode not in TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}; got {target_mode!r}"
        )
    ssc_value = _probability(ssc, "ssc")
    relssc_value = _probability(relssc, "relssc")
    ratio = _probability(relation_valid_ratio, "relation_valid_ratio")
    if isinstance(lambda_rel, bool) or not isinstance(lambda_rel, Real):
        raise ValueError(f"lambda_rel must be finite and non-negative; got {lambda_rel!r}")
    relation_scale = float(lambda_rel)
    if not math.isfinite(relation_scale) or relation_scale < 0:
        raise ValueError(f"lambda_rel must be finite and non-negative; got {lambda_rel!r}")

    if mode == "ssc":
        return ssc_value
    if mode == "relssc_replace":
        return relssc_value

    penalty = max(ssc_value - relssc_value, 0.0)
    target = ssc_value - relation_scale * ratio * penalty
    return min(0.99, max(0.01, target))


def attach_v2_target_inputs(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], SSCTargetContext]:
    """Return copies annotated with original SSC and relation-validity fields."""

    context = compute_ssc_target_context(records)
    output: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        copied["ssc"] = context.score(
            record.get("canonicalized_answer", record.get("canonical_answer"))
        ) if _is_valid_answer(record) else None
        copied["relation_valid_ratio"] = context.relation_valid_ratio
        copied["question_ssc_scores"] = dict(context.scores)
        copied["question_valid_identity_samples"] = context.valid_identity_samples
        copied["question_invalid_identity_samples"] = context.invalid_identity_samples
        copied["question_valid_relation_samples"] = context.valid_relation_samples
        copied["question_total_relation_samples"] = context.total_relation_samples
        output.append(copied)
    return tuple(output), context
