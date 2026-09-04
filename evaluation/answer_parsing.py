"""Strict answer extraction for RelaCaTS-v2 evaluation.

Model responses are parsed only from an explicit ``Answer:`` (or
``Final Answer:``) field. In particular, this module never falls back to a
dataset handler for a model response: several upstream numeric handlers scan
the whole chain-of-thought and return its last number, which can silently turn
an unfinished response into a seemingly valid answer.

Dataset handlers remain useful for *trusted ground-truth strings* (for
example, GSM8K stores its target after ``####``). That deliberately separate
operation is exposed as :func:`extract_gold_answer`.
"""

from __future__ import annotations

import re
from typing import Any, Literal


STRICT_EXPLICIT_ANSWER_PARSER_VERSION = "explicit-final-answer-v3"

# Backward-compatible exports for callers that record these names in artifact
# metadata. All response datasets now use one parser and therefore one
# version; the aliases are intentionally identical.
UPSTREAM_HANDLER_PARSER_VERSION = STRICT_EXPLICIT_ANSWER_PARSER_VERSION
MATHQA_PARSER_VERSION = STRICT_EXPLICIT_ANSWER_PARSER_VERSION

_OPTION_DATASETS = {
    "arc_challenge",
    "arc_easy",
    "aqua_rat",
    "commonsense_qa",
    "gpqa",
    "logiqa",
    "math_qa",
    "mathqa",
    "openbookqa",
    "reclor",
    "sciq",
    "winogrande",
}
_NUMERIC_DATASETS = {"gsm8k", "object_counting", "svamp"}

# The value is deliberately restricted to the same physical line. This
# prevents an empty ``Answer:`` in an explanation from claiming an unrelated
# number or letter several paragraphs later.
_ANSWER_MARKER = r"(?<![A-Za-z0-9_])(?:\*{1,2})?(?:final\s+)?answer\s*:(?:\*{1,2})?"
_ANSWER_FIELD_RE = re.compile(
    rf"(?im){_ANSWER_MARKER}[ \t]*(?P<value>.*?)(?={_ANSWER_MARKER}|$)"
)
_BOXED_RE = re.compile(r"^\\boxed\s*\{\s*([^{}]+?)\s*\}", re.IGNORECASE)
_DIRECT_OPTION_RE = re.compile(
    r"^(?:\*{1,2})?\s*(?:\(\s*)?([A-Ea-e])(?:\s*\))?(?:\*{1,2})?(?=$|[\s.,;:!?])"
)
_PAREN_OPTION_RE = re.compile(r"\(\s*([A-Ea-e])\s*\)")
_DIRECT_OPTION_INDEX_RE = re.compile(r"^\(?\s*([1-5])\s*\)?(?=$|[\s.,;:!?])")
_NUMERIC_RE = re.compile(
    r"^[ \t]*(?P<number>[-+]?(?:[$£€][ \t]*)?(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
    r")(?:[eE][-+]?\d+)?)(?![\d,])"
)

_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def _normalise_name(dataset_name: str) -> str:
    return str(dataset_name).strip().lower().replace("-", "_")


def parser_version(dataset_name: str) -> str:
    """Return the single v2 response-parser protocol version."""

    del dataset_name
    return STRICT_EXPLICIT_ANSWER_PARSER_VERSION


def _answer_kind(
    dataset_name: str, answer_type: str | None
) -> Literal["option", "number", "unknown"]:
    if answer_type:
        normalized_type = str(answer_type).strip().lower()
        if "option" in normalized_type or "letter" in normalized_type:
            return "option"
        if "number" in normalized_type or "numeric" in normalized_type:
            return "number"
    normalized_name = _normalise_name(dataset_name)
    if normalized_name in _OPTION_DATASETS:
        return "option"
    if normalized_name in _NUMERIC_DATASETS:
        return "number"
    return "unknown"


def _unbox(value: str) -> str:
    stripped = value.strip()
    match = _BOXED_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def _parse_option_field(value: str, allow_numeric_label: bool = True) -> str | None:
    unboxed = _unbox(value)
    direct = _DIRECT_OPTION_RE.match(unboxed)
    if direct:
        return direct.group(1).upper()

    # MathQA generations sometimes put a computed scalar before the selected
    # option, e.g. ``Answer: 0.036 (A)``. The parenthesized option is still an
    # explicit part of the Answer field, so it is safe to consume.
    numeric_prefix = _NUMERIC_RE.match(unboxed)
    if numeric_prefix:
        parenthesized = _PAREN_OPTION_RE.search(unboxed, numeric_prefix.end())
        if parenthesized:
            return parenthesized.group(1).upper()

    if allow_numeric_label:
        numeric_label = _DIRECT_OPTION_INDEX_RE.match(unboxed)
        if numeric_label:
            return chr(ord("A") + int(numeric_label.group(1)) - 1)
    return None


def _number_words_to_int(value: str) -> int | None:
    """Parse a leading, explicit English integer phrase."""

    tokens = re.findall(r"[A-Za-z]+", value.replace("-", " ").lower())
    if not tokens:
        return None

    sign = 1
    position = 0
    if tokens[0] in {"minus", "negative"}:
        sign = -1
        position = 1
    if position >= len(tokens):
        return None

    current = 0
    total = 0
    consumed_number = False
    while position < len(tokens):
        token = tokens[position]
        if token == "and" and consumed_number:
            position += 1
            continue
        if token in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[token]
            consumed_number = True
        elif token in _TENS:
            current += _TENS[token]
            consumed_number = True
        elif token == "hundred" and consumed_number:
            current = max(current, 1) * 100
        elif token in _SCALES and consumed_number:
            total += max(current, 1) * _SCALES[token]
            current = 0
        else:
            break
        position += 1
    return sign * (total + current) if consumed_number else None


def _parse_numeric_field(value: str) -> str | None:
    unboxed = _unbox(value)
    match = _NUMERIC_RE.match(unboxed)
    if match:
        return (
            match.group("number")
            .replace(",", "")
            .replace("$", "")
            .replace("£", "")
            .replace("€", "")
            .replace(" ", "")
        )
    words = _number_words_to_int(unboxed)
    return str(words) if words is not None else None


def _coerce_numeric_for_dataset(dataset_name: str, value: str) -> Any:
    normalized_name = _normalise_name(dataset_name)
    if normalized_name == "object_counting":
        try:
            numeric = float(value)
        except ValueError:
            return None
        return int(numeric) if numeric.is_integer() else numeric
    if normalized_name == "gsm8k" and value.endswith(".00"):
        return value[:-3]
    return value


def extract_explicit_answer(
    dataset_name: str,
    text: str,
    *,
    answer_type: str | None = None,
) -> Any:
    """Return the last valid answer from an explicit Answer field only."""

    kind = _answer_kind(dataset_name, answer_type)
    fields = [match.group("value") for match in _ANSWER_FIELD_RE.finditer(str(text))]
    for value in reversed(fields):
        if kind in {"option", "unknown"}:
            option = _parse_option_field(value)
            if option is not None:
                return option
        if kind in {"number", "unknown"}:
            numeric = _parse_numeric_field(value)
            if numeric is not None:
                return _coerce_numeric_for_dataset(dataset_name, numeric)
    return None


def extract_mathqa_option_answer(text: str) -> str | None:
    """Compatibility wrapper for the unified strict option parser."""

    answer = extract_explicit_answer("math_qa", text, answer_type="option letter")
    return str(answer) if answer is not None else None


def _normalise_arc_label(label: str) -> str:
    normalized = str(label).strip().upper()
    if normalized in {"1", "2", "3", "4", "5"}:
        return chr(ord("A") + int(normalized) - 1)
    return normalized


def extract_dataset_answer(
    dataset_name: str,
    text: str,
    handler: Any | None = None,
    *,
    answer_type: str | None = None,
) -> Any:
    """Parse a model response without consulting a free-form handler.

    ``handler`` remains in the signature for source compatibility but is
    intentionally ignored. Call :func:`extract_gold_answer` for trusted gold
    annotations that use a dataset-native representation.
    """

    del handler
    return extract_explicit_answer(dataset_name, text, answer_type=answer_type)


def extract_gold_answer(dataset_name: str, text: str, handler: Any) -> Any:
    """Extract a trusted dataset annotation using its native handler."""

    normalized_name = _normalise_name(dataset_name)
    # MathQA gold strings already use an explicit option label, and its
    # upstream handler is narrower than the unified parser.
    if normalized_name in {"math_qa", "mathqa"}:
        explicit = extract_mathqa_option_answer(text)
        if explicit is not None:
            return explicit
    extracted = handler.extract_answer(text)
    if extracted is not None and normalized_name in {"arc_challenge", "arc_easy"}:
        return _normalise_arc_label(str(extracted))
    return extracted


__all__ = [
    "MATHQA_PARSER_VERSION",
    "STRICT_EXPLICIT_ANSWER_PARSER_VERSION",
    "UPSTREAM_HANDLER_PARSER_VERSION",
    "extract_dataset_answer",
    "extract_explicit_answer",
    "extract_gold_answer",
    "extract_mathqa_option_answer",
    "parser_version",
]
