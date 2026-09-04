"""Runtime wrappers for the full RelaCaTS controller.

Offline label records use ``canonicalized_answer`` while evaluation artifacts
use ``extracted_answer``.  These wrappers bridge that schema difference before
calling the theory-core implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .full_relacats import (
    ControllerState,
    EffectiveVoteState,
    controller_state as _controller_state,
    effective_vote_state as _effective_vote_state,
)


def _normalize(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        if copied.get("canonicalized_answer") is None:
            copied["canonicalized_answer"] = copied.get(
                "extracted_answer", copied.get("answer")
            )
        copied.setdefault("is_valid_answer", copied.get("canonicalized_answer") is not None)
        output.append(copied)
    return output


def effective_vote_state(
    records: Sequence[Mapping[str, Any]],
    *,
    beta: float = 0.5,
    similarity_threshold: float = 0.86,
) -> EffectiveVoteState:
    return _effective_vote_state(
        _normalize(records),
        beta=beta,
        similarity_threshold=similarity_threshold,
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
    return _controller_state(
        _normalize(records),
        tau_support=tau_support,
        tau_fragility=tau_fragility,
        beta=beta,
        similarity_threshold=similarity_threshold,
        min_valid=min_valid,
    )
