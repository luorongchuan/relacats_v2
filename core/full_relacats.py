"""Full RelaCaTS theory utilities.

This module implements the components that are intentionally absent from the
minimal v1/v2 target-only pipeline:

* dependency correction ``d_i = |B_i|^{-beta}``;
* relation/dependency-weighted RelSSC;
* the consensus-fragility pseudo-label from Eq. (24);
* effective test-time weights ``q_i (1-f_i) d_i``;
* the STOP / SAMPLE / INTERVENE controller from Eqs. (34)--(37).

The theory document specifies *what* a dependency block is, but not a concrete
clustering algorithm.  To keep the repository dependency-free and reproducible,
``annotate_dependency_weights`` uses a deterministic lexical-Jaccard fallback
when an upstream ``dependency_cluster_id`` / ``strategy_cluster_id`` is absent.
Projects with a stronger semantic clusterer can write those IDs upstream and
this module will respect them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


_INVALID_ANSWERS = {"", "none", "null", "n/a", "na", "invalid", "<invalid>"}
_FINAL_ANSWER_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:\*\*)?answer(?:\*\*)?\s*[:\-].*$"
)
_TOKEN_RE = re.compile(r"[a-zA-Z]+|[-+*/=<>]|\d+(?:\.\d+)?")


class FullRelaCaTSError(ValueError):
    """Invalid full-RelaCaTS input or configuration."""


class ControllerAction(str, Enum):
    STOP = "STOP"
    SAMPLE = "SAMPLE"
    INTERVENE = "INTERVENE"


@dataclass(frozen=True)
class DependencySummary:
    beta: float
    similarity_threshold: float
    cluster_sizes: Mapping[str, int]
    valid_sample_count: int
    invalid_sample_count: int

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_sizes)

    @property
    def effective_cluster_mass(self) -> float:
        """Sum of cluster coefficients m^(1-beta), matching Eq. (90)."""

        return math.fsum(
            size ** (1.0 - self.beta) for size in self.cluster_sizes.values()
        )


@dataclass(frozen=True)
class FragilityResult:
    scores: Mapping[str, float]
    targets: tuple[float | None, ...]
    identity_support: Mapping[str, float]
    relational_support: Mapping[str, float]
    view_supports: Mapping[str, Mapping[str, float]]
    view_weights: Mapping[str, float]
    defined: bool
    reason: str | None = None


@dataclass(frozen=True)
class EffectiveVoteState:
    scores: Mapping[str, float]
    total_weight: float
    leader: str | None
    support_ratio: float
    leader_fragility: float
    valid_sample_count: int


@dataclass(frozen=True)
class ControllerState:
    action: ControllerAction
    leader: str | None
    support_ratio: float
    leader_fragility: float
    valid_sample_count: int
    total_weight: float


def _finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FullRelaCaTSError(f"{name} must be numeric; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FullRelaCaTSError(f"{name} must be in [0,1]; got {value!r}")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FullRelaCaTSError(f"{name} must be numeric; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FullRelaCaTSError(f"{name} must be finite and non-negative; got {value!r}")
    return result


def _canonical_answer(record: Mapping[str, Any]) -> str | None:
    value = record.get("canonicalized_answer", record.get("canonical_answer"))
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if text.lower() in _INVALID_ANSWERS:
        return None
    if len(text) == 1 and text.upper() in "ABCDE":
        text = text.upper()
    return text


def _bool_like(value: Any, default: bool = True) -> bool:
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
    raise FullRelaCaTSError(f"validity flag must be boolean-like; got {value!r}")


def _valid_record(record: Mapping[str, Any]) -> bool:
    if _canonical_answer(record) is None:
        return False
    if not _bool_like(record.get("is_valid_answer", record.get("valid")), True):
        return False
    status = record.get("canonicalization_status")
    status_value = getattr(status, "value", status)
    return status_value is None or str(status_value).strip().lower() == "valid"


def _strategy_tokens(text: Any) -> frozenset[str]:
    value = _FINAL_ANSWER_RE.sub(" ", str(text or "")).lower()
    return frozenset(_TOKEN_RE.findall(value))


def strategy_similarity(left: Any, right: Any) -> float:
    """Deterministic Jaccard fallback for approximate strategy similarity."""

    a = _strategy_tokens(left)
    b = _strategy_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _explicit_cluster_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("dependency_cluster_id", record.get("strategy_cluster_id"))
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def annotate_dependency_weights(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float = 0.5,
    similarity_threshold: float = 0.86,
    answer_sensitive: bool = True,
) -> tuple[tuple[dict[str, Any], ...], DependencySummary]:
    """Attach dependency blocks and ``d_i=|B_i|^{-beta}`` to copied records.

    Explicit cluster IDs are authoritative.  Remaining valid samples are
    greedily matched to the first/highest-similarity representative above the
    configured threshold; by default only samples supporting the same
    canonical answer may share a fallback block.
    """

    beta_value = _finite_probability(beta, "beta")
    threshold = _finite_probability(similarity_threshold, "similarity_threshold")
    copied = [dict(record) for record in records]
    assigned: list[str | None] = [None] * len(copied)
    representatives: dict[str, int] = {}
    cluster_order: list[str] = []
    next_index = 0

    for index, record in enumerate(copied):
        if not _valid_record(record):
            continue
        explicit = _explicit_cluster_id(record)
        if explicit is not None:
            assigned[index] = explicit
            if explicit not in representatives:
                representatives[explicit] = index
                cluster_order.append(explicit)
            continue

        answer = _canonical_answer(record)
        text = record.get("response", record.get("reasoning", record.get("text", "")))
        best_cluster: str | None = None
        best_similarity = -1.0
        for cluster_id in cluster_order:
            rep_index = representatives[cluster_id]
            rep = copied[rep_index]
            if _explicit_cluster_id(rep) is not None:
                # Explicit upstream IDs define a block but do not absorb
                # unlabelled records unless their text independently matches.
                pass
            if answer_sensitive and _canonical_answer(rep) != answer:
                continue
            rep_text = rep.get("response", rep.get("reasoning", rep.get("text", "")))
            similarity = strategy_similarity(text, rep_text)
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        if best_cluster is None:
            while True:
                candidate = f"B{next_index:04d}"
                next_index += 1
                if candidate not in representatives:
                    break
            best_cluster = candidate
            representatives[best_cluster] = index
            cluster_order.append(best_cluster)
        assigned[index] = best_cluster

    cluster_sizes: dict[str, int] = {}
    for cluster_id in assigned:
        if cluster_id is not None:
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    valid_count = 0
    invalid_count = 0
    output: list[dict[str, Any]] = []
    for record, cluster_id in zip(copied, assigned):
        if cluster_id is None:
            invalid_count += 1
            record["dependency_cluster_id"] = None
            record["dependency_cluster_size"] = 0
            record["dependency_weight"] = 0.0
        else:
            valid_count += 1
            size = cluster_sizes[cluster_id]
            record["dependency_cluster_id"] = cluster_id
            record["dependency_cluster_size"] = size
            record["dependency_weight"] = size ** (-beta_value)
        output.append(record)

    return tuple(output), DependencySummary(
        beta=beta_value,
        similarity_threshold=threshold,
        cluster_sizes=cluster_sizes,
        valid_sample_count=valid_count,
        invalid_sample_count=invalid_count,
    )


def compute_relssc_full(
    records: Sequence[Mapping[str, Any]],
    *,
    zero_weight_policy: str = "skip",
):
    """Compute Eq. (20) using per-sample ``r_g d_gi C_gi`` weights."""

    # Local import avoids a module-level cycle through core.__init__.
    from .relssc import compute_relssc

    return compute_relssc(
        records,
        zero_weight_policy=zero_weight_policy,
        enforce_v1_weights=False,
    )


def _relation_id(record: Mapping[str, Any]) -> str:
    value = record.get("relation_id")
    if value is not None:
        return str(value)
    view = record.get("view_index")
    return f"g{int(view)}" if view is not None else "g0"


def _is_identity(record: Mapping[str, Any]) -> bool:
    relation_type = record.get("relation_type")
    if relation_type is not None and str(relation_type).strip().lower() == "identity":
        return True
    if record.get("view_index") is not None:
        try:
            return int(record["view_index"]) == 0
        except (TypeError, ValueError):
            return False
    return _relation_id(record).strip().lower() == "g0"


def _view_relation_weight(records: Sequence[Mapping[str, Any]]) -> float:
    weights = [
        _finite_nonnegative(record.get("relation_weight", 1.0), "relation_weight")
        for record in records
    ]
    if not weights:
        return 0.0
    # Deterministic relations normally have r_g=1.  Averaging is robust to
    # serialized per-sample reliability estimates while keeping r_g in [0,1].
    weight = math.fsum(weights) / len(weights)
    if weight > 1.0:
        raise FullRelaCaTSError(f"relation_weight must be <=1 for fragility; got {weight}")
    return weight


def compute_fragility(
    records: Sequence[Mapping[str, Any]],
    *,
    lambda_v: float = 0.5,
) -> FragilityResult:
    """Compute the Eq. (24) consensus-fragility pseudo-label.

    ``s_g`` uses ``d_i C_i`` inside each relation view.  ``s_R`` and its
    variance use the relation reliability ``r_g``.  Invalid answers and views
    with zero positive mass are excluded; the identity view must remain
    defined for the label to be meaningful.
    """

    if isinstance(lambda_v, bool) or not isinstance(lambda_v, Real):
        raise FullRelaCaTSError("lambda_v must be numeric")
    lambda_value = float(lambda_v)
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise FullRelaCaTSError("lambda_v must be finite and non-negative")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    identity_ids: set[str] = set()
    for record in records:
        relation_id = _relation_id(record)
        grouped.setdefault(relation_id, []).append(record)
        if _is_identity(record):
            identity_ids.add(relation_id)

    view_supports: dict[str, dict[str, float]] = {}
    view_weights: dict[str, float] = {}
    all_answers: set[str] = set()
    for relation_id, view_records in grouped.items():
        denominator = 0.0
        support: dict[str, float] = {}
        for record in view_records:
            if not _valid_record(record):
                continue
            answer = _canonical_answer(record)
            assert answer is not None
            confidence = _finite_probability(record.get("confidence"), "confidence")
            dependency = _finite_nonnegative(
                record.get("dependency_weight", 1.0), "dependency_weight"
            )
            weight = dependency * confidence
            denominator += weight
            support[answer] = support.get(answer, 0.0) + weight
            all_answers.add(answer)
        if denominator <= 0.0:
            continue
        view_supports[relation_id] = {
            answer: value / denominator for answer, value in support.items()
        }
        view_weights[relation_id] = _view_relation_weight(view_records)

    identity_defined = sorted(identity_ids & set(view_supports))
    if not identity_defined:
        return FragilityResult(
            scores={},
            targets=tuple(None for _ in records),
            identity_support={},
            relational_support={},
            view_supports=view_supports,
            view_weights=view_weights,
            defined=False,
            reason="no valid positive-mass identity view",
        )
    identity_id = identity_defined[0]
    identity_support = view_supports[identity_id]

    active_views = [
        relation_id
        for relation_id in view_supports
        if view_weights.get(relation_id, 0.0) > 0.0
    ]
    relation_denominator = math.fsum(view_weights[relation_id] for relation_id in active_views)
    if relation_denominator <= 0.0:
        return FragilityResult(
            scores={},
            targets=tuple(None for _ in records),
            identity_support=identity_support,
            relational_support={},
            view_supports=view_supports,
            view_weights=view_weights,
            defined=False,
            reason="no positive relation reliability mass",
        )

    relational_support: dict[str, float] = {}
    for answer in all_answers:
        relational_support[answer] = math.fsum(
            view_weights[relation_id] * view_supports[relation_id].get(answer, 0.0)
            for relation_id in active_views
        ) / relation_denominator

    fragility: dict[str, float] = {}
    for answer in all_answers:
        s_r = relational_support.get(answer, 0.0)
        variance = math.fsum(
            view_weights[relation_id]
            * (view_supports[relation_id].get(answer, 0.0) - s_r) ** 2
            for relation_id in active_views
        ) / relation_denominator
        value = max(identity_support.get(answer, 0.0) - s_r, 0.0)
        value += lambda_value * math.sqrt(max(variance, 0.0))
        fragility[answer] = min(1.0, max(0.0, value))

    targets = tuple(
        fragility.get(_canonical_answer(record)) if _valid_record(record) else None
        for record in records
    )
    return FragilityResult(
        scores=fragility,
        targets=targets,
        identity_support=dict(identity_support),
        relational_support=relational_support,
        view_supports=view_supports,
        view_weights=view_weights,
        defined=True,
    )


def attach_full_targets(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float = 0.5,
    similarity_threshold: float = 0.86,
    lambda_v: float = 0.5,
) -> tuple[tuple[dict[str, Any], ...], Any, FragilityResult, DependencySummary]:
    """One-shot offline label construction for Eqs. (19)--(25)."""

    weighted, dependency = annotate_dependency_weights(
        records,
        beta=beta,
        similarity_threshold=similarity_threshold,
    )
    relssc = compute_relssc_full(weighted, zero_weight_policy="skip")
    if not relssc.defined:
        return weighted, relssc, compute_fragility(weighted, lambda_v=lambda_v), dependency
    with_relssc: list[dict[str, Any]] = []
    for record, target in zip(weighted, relssc.targets):
        copied = dict(record)
        copied["relssc"] = target
        copied["relational_consistency"] = target
        with_relssc.append(copied)
    fragility = compute_fragility(with_relssc, lambda_v=lambda_v)
    output: list[dict[str, Any]] = []
    for record, target in zip(with_relssc, fragility.targets):
        copied = dict(record)
        copied["fragility_target"] = target
        copied["consensus_fragility"] = target
        output.append(copied)
    return tuple(output), relssc, fragility, dependency


def fragility_suffix(model_name: str | None = None) -> str:
    """Instruction ``I_frag`` for the same-LM fragility query in Eq. (27)."""

    del model_name
    return (
        "Is the confidence in the preceding answer fragile under valid "
        "relation-preserving transformations? Answer Yes or No:"
    )


def _prediction_fields(record: Mapping[str, Any]) -> tuple[str | None, float | None, float | None]:
    answer_value = record.get("extracted_answer", record.get("answer", record.get("canonicalized_answer")))
    answer = None if answer_value is None else str(answer_value).strip()
    if not answer:
        return None, None, None
    try:
        confidence = float(record.get("confidence"))
        fragility = float(record.get("fragility", record.get("predicted_fragility")))
    except (TypeError, ValueError):
        return answer, None, None
    if not (math.isfinite(confidence) and 0.0 <= confidence <= 1.0):
        return answer, None, None
    if not (math.isfinite(fragility) and 0.0 <= fragility <= 1.0):
        return answer, None, None
    return answer, confidence, fragility


def effective_vote_state(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float = 0.5,
    similarity_threshold: float = 0.86,
) -> EffectiveVoteState:
    """Compute Eqs. (34)--(36) from predicted q/f on a sampled prefix."""

    weighted, _ = annotate_dependency_weights(
        records,
        beta=beta,
        similarity_threshold=similarity_threshold,
        answer_sensitive=True,
    )
    scores: dict[str, float] = {}
    total = 0.0
    valid = 0
    leader_fragility_num: dict[str, float] = {}
    leader_fragility_den: dict[str, float] = {}
    for record in weighted:
        answer, confidence, fragility = _prediction_fields(record)
        if answer is None or confidence is None or fragility is None:
            continue
        dependency = _finite_nonnegative(
            record.get("dependency_weight", 1.0), "dependency_weight"
        )
        vote_weight = dependency * confidence * (1.0 - fragility)
        scores[answer] = scores.get(answer, 0.0) + vote_weight
        total += vote_weight
        q_weight = dependency * confidence
        leader_fragility_num[answer] = leader_fragility_num.get(answer, 0.0) + q_weight * fragility
        leader_fragility_den[answer] = leader_fragility_den.get(answer, 0.0) + q_weight
        valid += 1
    leader = max(scores, key=scores.__getitem__) if scores else None
    ratio = scores[leader] / total if leader is not None and total > 0.0 else 0.0
    if leader is None or leader_fragility_den.get(leader, 0.0) <= 0.0:
        mean_fragility = 1.0
    else:
        mean_fragility = (
            leader_fragility_num[leader] / leader_fragility_den[leader]
        )
    return EffectiveVoteState(
        scores=scores,
        total_weight=total,
        leader=leader,
        support_ratio=ratio,
        leader_fragility=mean_fragility,
        valid_sample_count=valid,
    )


def controller_state(
    records: Sequence[Mapping[str, Any]],
    *,
    tau_support: float,
    tau_fragility: float,
    beta: float = 0.5,
    similarity_threshold: float = 0.86,
    min_valid: int = 2,
) -> ControllerState:
    """Return STOP/SAMPLE/INTERVENE exactly from Eq. (37)."""

    tau_s = _finite_probability(tau_support, "tau_support")
    tau_f = _finite_probability(tau_fragility, "tau_fragility")
    if min_valid <= 0:
        raise FullRelaCaTSError("min_valid must be positive")
    state = effective_vote_state(
        records,
        beta=beta,
        similarity_threshold=similarity_threshold,
    )
    if state.valid_sample_count < min_valid or state.support_ratio < tau_s:
        action = ControllerAction.SAMPLE
    elif state.leader_fragility <= tau_f:
        action = ControllerAction.STOP
    else:
        action = ControllerAction.INTERVENE
    return ControllerState(
        action=action,
        leader=state.leader,
        support_ratio=state.support_ratio,
        leader_fragility=state.leader_fragility,
        valid_sample_count=state.valid_sample_count,
        total_weight=state.total_weight,
    )
