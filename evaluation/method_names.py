"""Canonical method labels used by RelaCaTS evaluation reports.

The evaluator historically called the three methods introduced by this
project ``CaTS-SC``, ``CaTS-ES`` and ``CaTS-ASC``.  That spelling is ambiguous
once the report also contains the original CaTS baselines.  New reports must
use the ``RelaCaTS-*`` labels, while readers of older JSON/CSV artifacts should
still be able to normalize the legacy spelling.

The functions in this module are deliberately independent of the aggregation
implementation.  They can therefore be used by report writers, downstream
analysis scripts, and compatibility importers without changing the numerical
semantics of a method.
"""

from __future__ import annotations

from typing import Any, Mapping


# Original CaTS/Table-2 baselines.  Keep this order aligned with the paper's
# table wherever a report needs a stable display order.
BASELINE_METHODS: tuple[str, ...] = (
    "SC",
    "CISC",
    "Self-Certainty",
    "Best-of-N",
    "ASC",
    "ESC",
    "RASC",
)

# Methods trained/evaluated by this repository.  Their labels intentionally
# carry the project name so they cannot be mistaken for original CaTS rows.
RELACATS_METHODS: tuple[str, ...] = (
    "RelaCaTS-SC",
    "RelaCaTS-ES",
    "RelaCaTS-ASC",
)

# Historical labels emitted by the first v1 evaluator and by the upstream
# Table-2 helper.  These are input aliases only; report writers must emit the
# names in ``RELACATS_METHODS``.
LEGACY_RELACATS_METHODS: tuple[str, ...] = (
    "CaTS-SC",
    "CaTS-ES",
    "CaTS-ASC",
    "SC w/ Conf.",
    "Early Stopping",
    "ASC w/ Conf.",
)

ALL_METHODS: tuple[str, ...] = BASELINE_METHODS + RELACATS_METHODS

# Preferred Table-2-style row ordering.  This is only presentation metadata;
# callers may choose a different order for curves or ablation tables.
TABLE2_METHOD_ORDER: tuple[str, ...] = (
    "SC",
    "CISC",
    "Self-Certainty",
    "RelaCaTS-SC",
    "Best-of-N",
    "RelaCaTS-ES",
    "ASC",
    "RelaCaTS-ASC",
    "ESC",
    "RASC",
)


# Legacy spellings accepted on input.  Do not emit these names in newly
# written reports.  A few separator/case variants are included because old
# hand-written summaries used underscores or lower-case labels.
METHOD_ALIASES: dict[str, str] = {
    "CaTS-SC": "RelaCaTS-SC",
    "CaTS-ES": "RelaCaTS-ES",
    "CaTS-ASC": "RelaCaTS-ASC",
    "cats_sc": "RelaCaTS-SC",
    "cats_es": "RelaCaTS-ES",
    "cats_asc": "RelaCaTS-ASC",
    # Lower-case hyphenated spellings occur in a few hand-written JSON
    # summaries.  Canonicalization is case-insensitive, but the explicit
    # entries keep those spellings distinct from the baseline ``ASC`` token.
    "cats-sc": "RelaCaTS-SC",
    "cats-es": "RelaCaTS-ES",
    "cats-asc": "RelaCaTS-ASC",
    "relacats_sc": "RelaCaTS-SC",
    "relacats_es": "RelaCaTS-ES",
    "relacats_asc": "RelaCaTS-ASC",
    "relacats-sc": "RelaCaTS-SC",
    "relacats-es": "RelaCaTS-ES",
    "relacats-asc": "RelaCaTS-ASC",
    # Names used by the original repository's Table-2 aggregation helper.
    "SC w/ Conf.": "RelaCaTS-SC",
    "Early Stopping": "RelaCaTS-ES",
    "ASC w/ Conf.": "RelaCaTS-ASC",
    "sc_conf": "RelaCaTS-SC",
    "early_stopping": "RelaCaTS-ES",
    "asc_conf": "RelaCaTS-ASC",
}

_ALIASES_CASEFOLD: dict[str, str] = {
    key.casefold(): value for key, value in METHOD_ALIASES.items()
}


def canonical_method_name(name: Any) -> str:
    """Return the canonical display label for *name*.

    Unknown labels are returned unchanged (apart from surrounding whitespace),
    which keeps this helper forward-compatible with future baselines.  Only
    the three legacy project-method spellings are rewritten; plain ``ASC``
    remains the original baseline and is never silently changed.
    """

    text = str(name).strip()
    if not text:
        return text
    return _ALIASES_CASEFOLD.get(text.casefold(), text)


def canonicalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one report row and normalize its ``method`` field if present."""

    result = dict(row)
    if "method" in result and result["method"] is not None:
        result["method"] = canonical_method_name(result["method"])
    return result


def canonicalize_report_methods(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compatibility-normalized copy of an aggregate report.

    This helper handles the standard report collections (fixed rows, dynamic
    rows, and threshold curves) without assuming any particular algorithm.  It
    is useful when loading a pre-RelaCaTS report for comparison; current report
    writers should canonicalize before serializing instead.
    """

    result = dict(report)
    for field in ("fixed_budget_results", "dynamic_budget_matches"):
        rows = result.get(field)
        if isinstance(rows, list):
            result[field] = [canonicalize_row(row) for row in rows]
    # ``method_metadata`` was added after the first v1 reports.  Normalize its
    # method-keyed entries as well when loading an older report, while leaving
    # scalar metadata untouched.
    metadata = result.get("method_metadata")
    if isinstance(metadata, Mapping):
        normalized_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            canonical = canonical_method_name(key)
            # A mixed report can contain both a legacy alias and its new
            # spelling.  Preserve fields from both entries instead of letting
            # whichever key happens to be visited last silently win.
            if canonical in normalized_metadata:
                previous = normalized_metadata[canonical]
                if isinstance(previous, Mapping) and isinstance(value, Mapping):
                    merged = dict(previous)
                    merged.update(value)
                    normalized_metadata[canonical] = merged
                    continue
            normalized_metadata[canonical] = value
        result["method_metadata"] = normalized_metadata

    method_order = result.get("method_order")
    if isinstance(method_order, list):
        result["method_order"] = list(
            dict.fromkeys(canonical_method_name(item) for item in method_order)
        )

    curves = result.get("threshold_curves")
    if isinstance(curves, Mapping):
        normalized_curves: dict[str, Any] = {}
        for key, rows in curves.items():
            canonical = canonical_method_name(key)
            normalized_rows = (
                [canonicalize_row(row) for row in rows]
                if isinstance(rows, list)
                else rows
            )
            # If a mixed report contains both a legacy and canonical key,
            # merge list-valued curves instead of silently dropping one.
            if canonical in normalized_curves:
                previous = normalized_curves[canonical]
                if isinstance(previous, list) and isinstance(normalized_rows, list):
                    normalized_curves[canonical] = previous + normalized_rows
                    continue
            normalized_curves[canonical] = normalized_rows
        result["threshold_curves"] = normalized_curves
    return result


__all__ = [
    "BASELINE_METHODS",
    "RELACATS_METHODS",
    "LEGACY_RELACATS_METHODS",
    "ALL_METHODS",
    "TABLE2_METHOD_ORDER",
    "METHOD_ALIASES",
    "canonical_method_name",
    "canonicalize_row",
    "canonicalize_report_methods",
]
