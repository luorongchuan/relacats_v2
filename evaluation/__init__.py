"""RelaCaTS-v1 test-time evaluation.

RelaCaTS-v1 deliberately uses the unmodified CaTS test-time protocol: the
relational transformations are used to construct training targets, but are
not applied during evaluation.  The public CPU API is exposed here; the two
vLLM stages live in :mod:`generate_responses` and
:mod:`calculate_confidence` and keep their heavy imports lazy.
"""

__all__ = [
    "AggregateConfig",
    "evaluate_records",
    "run_aggregation",
    "write_reports",
    # Public naming helpers keep report consumers from depending on the
    # aggregator's private constants.  Legacy ``CaTS-*`` labels are accepted
    # on input but canonical output uses ``RelaCaTS-*``.
    "BASELINE_METHODS",
    "RELACATS_METHODS",
    "LEGACY_RELACATS_METHODS",
    "ALL_METHODS",
    "TABLE2_METHOD_ORDER",
    "canonical_method_name",
    "canonicalize_report_methods",
]


def __getattr__(name: str):
    """Lazily expose the CPU API without pre-importing CLI modules."""

    if name in {"BASELINE_METHODS", "RELACATS_METHODS",
                "LEGACY_RELACATS_METHODS", "ALL_METHODS", "TABLE2_METHOD_ORDER",
                "canonical_method_name", "canonicalize_report_methods"}:
        from relacats_v2.evaluation import method_names

        return getattr(method_names, name)
    if name in {"AggregateConfig", "evaluate_records", "run_aggregation", "write_reports"}:
        from relacats_v2.evaluation import aggregate

        return getattr(aggregate, name)
    raise AttributeError(name)
