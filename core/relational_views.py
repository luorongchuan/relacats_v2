"""Relational views for multiple-choice questions.

The direction of every permutation in this module is deliberately explicit:

``permutation`` / ``forward_mapping``
    original answer space -> transformed answer space (``phi_g``).

``inverse_permutation`` / ``inverse_mapping``
    transformed answer space -> original answer space (``phi_g^{-1}``).

Internally, ``forward_indices[i]`` is the transformed position of the option
that was originally at position ``i``.  Keeping one convention throughout is
important: treating a transformed-order list as a forward permutation is a
subtle but destructive source of incorrect pseudo-labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import random
from typing import Any, Mapping, Sequence


DEFAULT_OPTION_LABELS: tuple[str, ...] = tuple("ABCDE")


class RelationalViewError(ValueError):
    """Base class for invalid relational-view configuration."""


class InvalidPermutationError(RelationalViewError):
    """Raised when a permutation is malformed or directionally inconsistent."""


class SamplingBudgetError(RelationalViewError):
    """Raised when the relational sampling budget is not the CaTS budget."""


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SamplingBudgetError(f"{name} must be a positive integer; got {value!r}")
    return value


def validate_sampling_budget(
    num_views: int = 4,
    samples_per_view: int = 8,
    total_budget: int = 32,
    *,
    relation_mode: str = "option_permutation",
    allow_nonstandard_budget: bool = False,
) -> int:
    """Validate and return the total number of responses per original sample.

    RelaCaTS-v1 uses exactly the original CaTS teacher budget (32 responses).
    Ordinary option-MCQ tasks use four views times eight responses; the
    two-option WinoGrande adapter uses the two unique views (identity and swap)
    with 16 responses each.  Scalar tasks such as GSM8K/SVAMP have no option
    space to permute, so their explicitly named ``identity`` mode uses one
    identity view with 32 responses.  The default remains the
    option-permutation mode, preserving the original API; formal profile
    enforcement is performed by the CLI so older smoke callers can still
    request another product-compatible budget.
    """

    num_views = _require_positive_int(num_views, "num_views")
    samples_per_view = _require_positive_int(samples_per_view, "samples_per_view")
    total_budget = _require_positive_int(total_budget, "total_budget")
    mode = str(relation_mode).strip().lower().replace("-", "_")
    aliases = {
        "option": "option_permutation",
        "mcq": "option_permutation",
        "numeric": "identity",
        "scalar": "identity",
        "identity_only": "identity",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"option_permutation", "identity"}:
        raise SamplingBudgetError(
            f"unsupported relation_mode={relation_mode!r}; expected "
            "option_permutation or identity"
        )
    actual = num_views * samples_per_view
    if actual != total_budget:
        raise SamplingBudgetError(
            "relational sampling budget mismatch: "
            f"num_views({num_views}) * samples_per_view({samples_per_view}) "
            f"= {actual}, expected total_budget={total_budget}"
        )
    if (
        mode == "identity"
        and not allow_nonstandard_budget
        and (num_views, samples_per_view, total_budget) != (1, 32, 32)
    ):
        raise SamplingBudgetError(
            "identity numeric RelaCaTS-v1 requires 1 view x 32 responses = 32; "
            f"got {num_views}x{samples_per_view}={total_budget}"
        )
    return actual


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label).strip().upper() for label in labels)
    if not 2 <= len(normalized) <= len(DEFAULT_OPTION_LABELS):
        raise InvalidPermutationError(
            f"RelaCaTS-v1 supports 2--5 options (A--E); got {len(normalized)}"
        )
    if len(set(normalized)) != len(normalized):
        raise InvalidPermutationError(f"option labels must be unique; got {normalized!r}")
    if any(label not in DEFAULT_OPTION_LABELS for label in normalized):
        raise InvalidPermutationError(
            f"option labels must be drawn from A--E; got {normalized!r}"
        )
    return normalized


def _validate_forward_indices(
    forward_indices: Sequence[int], number_of_options: int
) -> tuple[int, ...]:
    result = tuple(forward_indices)
    if len(result) != number_of_options:
        raise InvalidPermutationError(
            "permutation length must equal the number of option labels: "
            f"{len(result)} != {number_of_options}"
        )
    if any(isinstance(index, bool) or not isinstance(index, int) for index in result):
        raise InvalidPermutationError("permutation indices must be integers")
    if set(result) != set(range(number_of_options)):
        raise InvalidPermutationError(
            "permutation must be a bijection over option indices; "
            f"got {result!r}"
        )
    return result


def _coerce_label(label: Any, labels: tuple[str, ...], direction: str) -> str:
    if not isinstance(label, str):
        raise InvalidPermutationError(
            f"{direction} answer label must be a string in {labels!r}; got {label!r}"
        )
    normalized = label.strip().upper()
    if normalized not in labels:
        raise InvalidPermutationError(
            f"{direction} answer label {label!r} is outside {labels!r}"
        )
    return normalized


@dataclass(frozen=True)
class OptionPermutation:
    """A bijection between original and transformed option-label spaces.

    ``forward_indices[original_index] == transformed_index``.  Therefore
    ``forward_answer`` implements ``phi_g`` and ``inverse_answer`` implements
    ``phi_g^{-1}``.
    """

    labels: tuple[str, ...]
    forward_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        labels = _validate_labels(self.labels)
        indices = _validate_forward_indices(self.forward_indices, len(labels))
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "forward_indices", indices)

    @classmethod
    def identity(cls, number_of_options: int = 4) -> "OptionPermutation":
        if isinstance(number_of_options, bool) or not isinstance(number_of_options, int):
            raise InvalidPermutationError("number_of_options must be an integer")
        labels = _validate_labels(DEFAULT_OPTION_LABELS[:number_of_options])
        return cls(labels=labels, forward_indices=tuple(range(number_of_options)))

    @classmethod
    def from_transformed_order(
        cls,
        transformed_order: Sequence[int],
        labels: Sequence[str] | None = None,
    ) -> "OptionPermutation":
        """Build from original indices listed in transformed display order.

        For example, ``[2, 0, 3, 1]`` means transformed A displays original C,
        transformed B displays original A, and so on.  Its forward mapping is
        therefore ``A->B, B->D, C->A, D->C``.
        """

        order = tuple(transformed_order)
        chosen_labels = _validate_labels(
            labels if labels is not None else DEFAULT_OPTION_LABELS[: len(order)]
        )
        _validate_forward_indices(order, len(chosen_labels))
        forward = [0] * len(order)
        for transformed_index, original_index in enumerate(order):
            forward[original_index] = transformed_index
        return cls(labels=chosen_labels, forward_indices=tuple(forward))

    @classmethod
    def from_forward_mapping(
        cls,
        mapping: Mapping[str, str],
        labels: Sequence[str] | None = None,
    ) -> "OptionPermutation":
        """Build from an explicit original-label -> transformed-label mapping."""

        if not isinstance(mapping, Mapping) or not mapping:
            raise InvalidPermutationError("forward mapping must be a non-empty mapping")
        if labels is None:
            label_set = {str(key).strip().upper() for key in mapping}
            chosen_labels = tuple(
                label for label in DEFAULT_OPTION_LABELS if label in label_set
            )
        else:
            chosen_labels = tuple(labels)
        chosen_labels = _validate_labels(chosen_labels)

        normalized_mapping = {
            str(key).strip().upper(): str(value).strip().upper()
            for key, value in mapping.items()
        }
        if set(normalized_mapping) != set(chosen_labels):
            raise InvalidPermutationError(
                "forward mapping keys must exactly match option labels; "
                f"got {tuple(normalized_mapping)!r}, expected {chosen_labels!r}"
            )
        try:
            indices = tuple(
                chosen_labels.index(normalized_mapping[label]) for label in chosen_labels
            )
        except ValueError as exc:
            raise InvalidPermutationError(
                "forward mapping values must be valid option labels"
            ) from exc
        return cls(labels=chosen_labels, forward_indices=indices)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "OptionPermutation":
        """Load and cross-check serialized relation metadata.

        At least one of ``permutation`` (forward) or ``inverse_permutation``
        must be present.  If both are present, they must be exact inverses.
        """

        if not isinstance(metadata, Mapping):
            raise InvalidPermutationError("relation metadata must be a mapping")
        forward = metadata.get("permutation", metadata.get("forward_mapping"))
        inverse = metadata.get("inverse_permutation", metadata.get("inverse_mapping"))
        raw_labels = metadata.get("option_labels")

        if forward is None and inverse is None:
            relation_type = str(metadata.get("relation_type", "")).lower()
            if relation_type == "identity" and raw_labels is not None:
                labels = _validate_labels(raw_labels)
                return cls(labels=labels, forward_indices=tuple(range(len(labels))))
            raise InvalidPermutationError(
                "relation metadata needs permutation (original->transformed) "
                "or inverse_permutation (transformed->original)"
            )

        if forward is not None:
            if not isinstance(forward, Mapping):
                raise InvalidPermutationError("metadata permutation must be a mapping")
            result = cls.from_forward_mapping(forward, labels=raw_labels)
        else:
            if not isinstance(inverse, Mapping):
                raise InvalidPermutationError("metadata inverse_permutation must be a mapping")
            inverse_normalized = {
                str(key).strip().upper(): str(value).strip().upper()
                for key, value in inverse.items()
            }
            # Swap keys and values to obtain original -> transformed.
            forward_from_inverse = {
                original: transformed
                for transformed, original in inverse_normalized.items()
            }
            result = cls.from_forward_mapping(forward_from_inverse, labels=raw_labels)

        if inverse is not None:
            if not isinstance(inverse, Mapping):
                raise InvalidPermutationError("metadata inverse_permutation must be a mapping")
            normalized_inverse = {
                str(key).strip().upper(): str(value).strip().upper()
                for key, value in inverse.items()
            }
            if normalized_inverse != result.inverse_mapping:
                raise InvalidPermutationError(
                    "permutation and inverse_permutation are directionally inconsistent"
                )
        return result

    @property
    def inverse_indices(self) -> tuple[int, ...]:
        """Original index at each transformed index."""

        inverse = [0] * len(self.forward_indices)
        for original_index, transformed_index in enumerate(self.forward_indices):
            inverse[transformed_index] = original_index
        return tuple(inverse)

    @property
    def forward_mapping(self) -> dict[str, str]:
        """Return ``phi_g``: original label -> transformed label."""

        return {
            self.labels[original]: self.labels[transformed]
            for original, transformed in enumerate(self.forward_indices)
        }

    @property
    def inverse_mapping(self) -> dict[str, str]:
        """Return ``phi_g^{-1}``: transformed label -> original label."""

        return {
            self.labels[transformed]: self.labels[original]
            for transformed, original in enumerate(self.inverse_indices)
        }

    def forward_answer(self, original_answer: str) -> str:
        """Map an original-space answer into the transformed answer space."""

        label = _coerce_label(original_answer, self.labels, "original-space")
        return self.forward_mapping[label]

    def inverse_answer(self, transformed_answer: str) -> str:
        """Map a transformed-space answer back into the original answer space."""

        label = _coerce_label(transformed_answer, self.labels, "transformed-space")
        return self.inverse_mapping[label]

    def permute_options(self, original_options: Sequence[str]) -> tuple[str, ...]:
        """Place original option texts at their transformed positions."""

        options = tuple(str(option) for option in original_options)
        if len(options) != len(self.labels):
            raise InvalidPermutationError(
                f"received {len(options)} options for {len(self.labels)} labels"
            )
        transformed: list[str | None] = [None] * len(options)
        for original_index, transformed_index in enumerate(self.forward_indices):
            transformed[transformed_index] = options[original_index]
        # Bijection validation guarantees every entry was populated.
        return tuple(option for option in transformed if option is not None)

    @property
    def is_identity(self) -> bool:
        return self.forward_indices == tuple(range(len(self.forward_indices)))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize with both mapping directions to make data self-checking."""

        return {
            "option_labels": list(self.labels),
            "permutation": self.forward_mapping,
            "inverse_permutation": self.inverse_mapping,
            "forward_indices": list(self.forward_indices),
            "inverse_indices": list(self.inverse_indices),
        }


def render_multiple_choice_question(
    question_stem: str,
    options: Sequence[str],
    labels: Sequence[str] | None = None,
    options_header: str = "Options:",
) -> str:
    """Render a stable ARC/MathQA-style multiple-choice prompt."""

    option_tuple = tuple(str(option) for option in options)
    chosen_labels = _validate_labels(
        labels if labels is not None else DEFAULT_OPTION_LABELS[: len(option_tuple)]
    )
    if len(option_tuple) != len(chosen_labels):
        raise RelationalViewError(
            f"received {len(option_tuple)} options for {len(chosen_labels)} labels"
        )
    stem = str(question_stem).rstrip()
    if not stem:
        raise RelationalViewError("question_stem must not be empty")
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(chosen_labels, option_tuple)
    )
    header = str(options_header).strip()
    return f"{stem}\n{header}\n{option_lines}" if header else f"{stem}\n{option_lines}"


@dataclass(frozen=True)
class RelationalView:
    """One identity or option-permuted view of an original question."""

    relation_id: str
    relation_type: str
    original_question: str
    transformed_question: str
    original_options: tuple[str, ...]
    transformed_options: tuple[str, ...]
    # Numeric GSM8K/SVAMP identity views have no option space.  ``None`` is an
    # explicit representation of that fact; option-MCQ views always carry an
    # OptionPermutation instance.
    option_permutation: OptionPermutation | None
    samples_per_view: int
    answer_type: str = "option letter"
    relation_mode: str = "option_permutation"
    # Two-option callers can request an explicit legacy/synthetic repeat via
    # ``allow_repeated_views``; this flag keeps such a repeat auditable.  The
    # built-in WinoGrande profile uses two unique views and therefore never
    # sets it.
    is_duplicate_view: bool = False

    @property
    def permutation(self) -> dict[str, str]:
        """Original-space -> transformed-space answer mapping."""

        return (
            self.option_permutation.forward_mapping
            if self.option_permutation is not None
            else {}
        )

    @property
    def inverse_permutation(self) -> dict[str, str]:
        """Transformed-space -> original-space answer mapping."""

        return (
            self.option_permutation.inverse_mapping
            if self.option_permutation is not None
            else {}
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.option_permutation is not None:
            metadata.update(self.option_permutation.to_metadata())
        metadata.update(
            {
                "relation_id": self.relation_id,
                "relation_type": self.relation_type,
                "samples_per_view": self.samples_per_view,
                "answer_type": self.answer_type,
                "relation_mode": self.relation_mode,
                "is_duplicate_view": self.is_duplicate_view,
            }
        )
        return metadata


def generate_identity_views(
    question: str,
    *,
    samples_per_view: int = 32,
    total_budget: int = 32,
    seed: int = 42,
    answer_type: str = "number",
    allow_nonstandard_budget: bool = False,
) -> tuple[RelationalView, ...]:
    """Return the single identity view used by scalar-answer datasets.

    GSM8K and SVAMP have no finite option-label space, so option permutation is
    undefined.  They still use the same 32-response teacher budget as CaTS,
    represented as ``g0`` with ``relation_mode='identity_only'``.  The ``seed`` is
    accepted for API symmetry and validated, although it does not affect an
    identity transform.
    """

    validate_sampling_budget(
        1,
        samples_per_view,
        total_budget,
        relation_mode="identity",
        allow_nonstandard_budget=allow_nonstandard_budget,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RelationalViewError(f"seed must be an integer; got {seed!r}")
    normalized_type = str(answer_type).strip().lower().replace("-", "_")
    if normalized_type not in {"number", "numeric", "scalar"}:
        raise RelationalViewError(
            "identity-only views are reserved for scalar numeric answers; "
            f"got answer_type={answer_type!r}"
        )
    text = str(question).strip()
    if not text:
        raise RelationalViewError("question must not be empty")
    return (
        RelationalView(
            relation_id="g0",
            relation_type="identity",
            original_question=text,
            transformed_question=text,
            original_options=tuple(),
            transformed_options=tuple(),
            option_permutation=None,
            samples_per_view=samples_per_view,
            answer_type="number",
            relation_mode="identity_only",
        ),
    )


def _coerce_options(
    options: Sequence[str] | Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(options, Mapping):
        normalized = {
            str(label).strip().upper(): str(text) for label, text in options.items()
        }
        labels = tuple(label for label in DEFAULT_OPTION_LABELS if label in normalized)
        labels = _validate_labels(labels)
        if set(normalized) != set(labels):
            raise RelationalViewError(
                "option mapping keys must be unique labels drawn from A--E"
            )
        return labels, tuple(normalized[label] for label in labels)

    if isinstance(options, (str, bytes)):
        raise RelationalViewError("options must be a sequence of option texts, not a string")
    option_tuple = tuple(str(option) for option in options)
    labels = _validate_labels(DEFAULT_OPTION_LABELS[: len(option_tuple)])
    return labels, option_tuple


def generate_option_permutation_views(
    question_stem: str,
    options: Sequence[str] | Mapping[str, str],
    *,
    num_views: int = 4,
    samples_per_view: int = 8,
    total_budget: int = 32,
    seed: int = 42,
    options_header: str = "Options:",
    allow_repeated_views: bool = False,
) -> tuple[RelationalView, ...]:
    """Generate one identity and distinct option-permutation views.

    The result is deterministic for a given seed.  The identity is always
    ``g0``; ``g1`` onward are non-identity permutations.  Views are unique, so
    asking for more than ``n!`` views is rejected instead of duplicating a
    supposedly independent relation, unless ``allow_repeated_views`` is
    explicitly enabled by a legacy or synthetic caller.  The built-in
    WinoGrande adapter does not enable it: WinoGrande uses the two unique
    views (identity and swap) with 16 responses per view.
    """

    validate_sampling_budget(
        num_views,
        samples_per_view,
        total_budget,
        relation_mode="option_permutation",
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RelationalViewError(f"seed must be an integer; got {seed!r}")

    labels, original_options = _coerce_options(options)
    unique_permutations = math.factorial(len(labels))
    if num_views > unique_permutations and not allow_repeated_views:
        raise RelationalViewError(
            f"cannot create {num_views} distinct views from {len(labels)} options; "
            f"at most {math.factorial(len(labels))} permutations exist"
        )

    identity_order = tuple(range(len(labels)))
    non_identity_orders = [
        order for order in permutations(range(len(labels))) if order != identity_order
    ]
    random.Random(seed).shuffle(non_identity_orders)
    available_orders = [identity_order, *non_identity_orders]
    if num_views <= len(available_orders):
        selected_orders = available_orders[:num_views]
    else:
        selected_orders = [
            available_orders[index % len(available_orders)]
            for index in range(num_views)
        ]

    original_question = render_multiple_choice_question(
        question_stem, original_options, labels, options_header
    )
    views: list[RelationalView] = []
    seen_orders: set[tuple[int, ...]] = set()
    for index, transformed_order in enumerate(selected_orders):
        option_permutation = OptionPermutation.from_transformed_order(
            transformed_order, labels
        )
        transformed_options = option_permutation.permute_options(original_options)
        transformed_question = render_multiple_choice_question(
            question_stem, transformed_options, labels, options_header
        )
        views.append(
            RelationalView(
                relation_id=f"g{index}",
                relation_type="identity" if index == 0 else "option_permutation",
                original_question=original_question,
                transformed_question=transformed_question,
                original_options=original_options,
                transformed_options=transformed_options,
                option_permutation=option_permutation,
                samples_per_view=samples_per_view,
                answer_type="option letter",
                relation_mode="option_permutation",
                is_duplicate_view=transformed_order in seen_orders,
            )
        )
        seen_orders.add(transformed_order)
    return tuple(views)
