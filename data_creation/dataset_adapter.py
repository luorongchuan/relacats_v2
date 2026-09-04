"""Adapters from the original CaTS datasets to RelaCaTS example records.

Most of the CaTS training tasks are multiple-choice questions.  RelaCaTS-v1
uses the option content and the gold position separately for those tasks so it
can build deterministic option-permutation views.  GSM8K and SVAMP are
different: they have scalar numeric answers, for which v1 currently keeps a
single identity view and spends the full 32-response budget on that view.

This module deliberately keeps source loading in the original
``utils.dataset_loader``.  It only normalises the source schemas and exposes a
small, model-independent example interface; generation and training code can
then dispatch on ``answer_type``/``relation_mode`` without guessing from a
dataset name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias

from relacats_v2.common import stable_id
from relacats_v2.core.canonicalization import normalize_numeric_answer
from utils.dataset_loader import get_dataset


"""The nine datasets used by the original CaTS training configuration.

The two legacy entries at the end are retained for backwards compatibility
with the earlier RelaCaTS-v1 Table-2/evaluation code.  New training scripts
should use :data:`TRAIN_DATASETS` explicitly.
"""
TRAIN_DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "gsm8k",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "svamp",
    "winogrande",
)

MCQ_DATASETS = (
    "arc_easy",
    "commonsense_qa",
    "logiqa",
    "openbookqa",
    "reclor",
    "sciq",
    "winogrande",
    # Existing v1 evaluation/data-generation support.  They are not part of
    # the nine-task CaTS training mixture, but keeping them here avoids a
    # backwards-incompatible change for callers that already use them.
    "arc_challenge",
    "math_qa",
)

NUMERIC_DATASETS = ("gsm8k", "svamp")
SUPPORTED_DATASETS = (*TRAIN_DATASETS, "arc_challenge", "math_qa")

AnswerType = Literal["option letter", "number"]
RelationMode = Literal["option_permutation", "identity_only"]


def option_labels(count: int) -> tuple[str, ...]:
    if not 2 <= count <= 26:
        raise ValueError(f"Option count must be in [2, 26], got {count}")
    return tuple(chr(ord("A") + index) for index in range(count))


@dataclass(frozen=True)
class MCQExample:
    dataset_name: str
    split: str
    source_index: int
    question_id: str
    stem: str
    options: tuple[str, ...]
    correct_index: int

    def __post_init__(self) -> None:
        if not self.stem.strip():
            raise ValueError("MCQ stem cannot be empty")
        if len(self.options) < 2:
            raise ValueError("MCQ must have at least two options")
        if any(not option.strip() for option in self.options):
            raise ValueError("MCQ options cannot be empty")
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError("correct_index is outside the option list")

    @property
    def labels(self) -> tuple[str, ...]:
        return option_labels(len(self.options))

    @property
    def answer_type(self) -> AnswerType:
        """Canonical answer kind consumed by the shared generator."""

        return "option letter"

    @property
    def relation_mode(self) -> RelationMode:
        return "option_permutation"

    @property
    def num_views(self) -> int:
        # WinoGrande has exactly two answer choices.  There are only two
        # distinct option permutations (identity and swap), so its full
        # 32-response budget is split across those two views rather than
        # repeating either permutation.
        return 2 if self.dataset_name.strip().lower() == "winogrande" else 4

    @property
    def samples_per_view(self) -> int:
        return 16 if self.dataset_name.strip().lower() == "winogrande" else 8

    @property
    def total_budget(self) -> int:
        return 32

    @property
    def is_numeric(self) -> bool:
        return False

    @property
    def is_multiple_choice(self) -> bool:
        return True

    @property
    def allow_repeated_views(self) -> bool:
        """Whether this example permits duplicate relation views.

        RelaCaTS-v1 does not permit duplicate views for WinoGrande: its
        formal profile is identity x16 plus swap x16.  The property remains in
        the public example interface for compatibility with older callers,
        but is now false for every built-in MCQ dataset.
        """

        return False

    @property
    def correct_answer(self) -> str:
        return self.labels[self.correct_index]

    def render(self, transformed_to_original: Sequence[str] | None = None) -> str:
        """Render options in new-label order.

        ``transformed_to_original[new_position]`` gives the original label whose
        content is placed at that transformed position.  This is the inverse
        permutation used during canonicalization.
        """

        labels = self.labels
        if transformed_to_original is None:
            transformed_to_original = labels
        if set(transformed_to_original) != set(labels):
            raise ValueError(
                "transformed_to_original must be a permutation of the option labels"
            )
        original_options = dict(zip(labels, self.options))
        option_block = "\n".join(
            f"{new_label}. {original_options[old_label]}"
            for new_label, old_label in zip(labels, transformed_to_original)
        )
        return f"{self.stem.rstrip()}\nOptions:\n{option_block}\n"


# GSM8K stores the target in a rationale string ending in ``#### <number>``;
# SVAMP generally stores a number directly, but mirrors occasionally serialise
# it as a string.  Keep the extraction intentionally conservative: prefer the
# GSM8K delimiter and otherwise use the final standalone numeric token.
_GSM8K_FINAL_RE = re.compile(
    r"####\s*([-+]?(?:\$?\d[\d,]*(?:\.\d+)?|\$?\.\d+))\s*$",
    flags=re.MULTILINE,
)
_NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?(?:\$?\d[\d,]*(?:\.\d+)?|\$?\.\d+)(?:[eE][-+]?\d+)?"
)


def _normalise_numeric_value(value: Any, *, field_name: str = "answer") -> str:
    """Return the same canonical decimal representation used by core.

    Keeping gold answers in canonical form is important for numeric RelSSC:
    ``42``, ``42.0`` and ``$42.00`` must be one answer class.  We deliberately
    do not evaluate fractions or arithmetic expressions here; those need a
    dataset-specific parser and should be reported as conversion failures.
    """

    result = normalize_numeric_answer(value)
    if result.valid and result.normalized_answer is not None:
        return result.normalized_answer
    raise ValueError(
        f"{field_name} must be one finite scalar number; got {value!r}"
    )


def _extract_gsm8k_gold(value: Any) -> str:
    """Extract GSM8K's final ``####`` answer and canonicalise it."""

    if value is None:
        raise ValueError("GSM8K answer is missing")
    # Numeric values are accepted for local mirrors that pre-extract targets.
    if not isinstance(value, str):
        return _normalise_numeric_value(value, field_name="GSM8K answer")
    text = value.strip()
    final_match = _GSM8K_FINAL_RE.search(text)
    if final_match:
        return _normalise_numeric_value(final_match.group(1), field_name="GSM8K answer")
    # A few converted copies omit ``####``.  Taking the final token mirrors the
    # original CaTS handler while still rejecting prose with no number.
    tokens = _NUMERIC_TOKEN_RE.findall(text)
    if not tokens:
        raise ValueError(f"GSM8K answer has no numeric target: {value!r}")
    return _normalise_numeric_value(tokens[-1], field_name="GSM8K answer")


@dataclass(frozen=True)
class NumericExample:
    """One scalar-answer question using RelaCaTS-v1's identity-only fallback.

    Numeric tasks cannot safely undergo an option permutation.  They therefore
    expose the same identifying fields as :class:`MCQExample`, plus explicit
    generation metadata that downstream code can use to request one identity
    view with 32 samples.  ``options`` is an empty tuple for duck-typed callers
    that expect the attribute, while ``labels`` stays empty and no option
    canonicalisation is attempted.
    """

    dataset_name: str
    split: str
    source_index: int
    question_id: str
    stem: str
    correct_answer: str

    # These class-level defaults are intentionally visible as instance
    # attributes too, so a generic generator can inspect either form.
    answer_type: AnswerType = "number"
    relation_mode: RelationMode = "identity_only"
    num_views: int = 1
    samples_per_view: int = 32
    total_budget: int = 32

    def __post_init__(self) -> None:
        if not str(self.stem).strip():
            raise ValueError("Numeric stem cannot be empty")
        if self.answer_type != "number":
            raise ValueError("NumericExample answer_type must be 'number'")
        if self.relation_mode != "identity_only":
            raise ValueError("NumericExample relation_mode must be 'identity_only'")
        if (self.num_views, self.samples_per_view, self.total_budget) != (1, 32, 32):
            raise ValueError(
                "NumericExample identity fallback requires 1 view x 32 responses = 32"
            )
        object.__setattr__(
            self,
            "correct_answer",
            _normalise_numeric_value(self.correct_answer, field_name="correct_answer"),
        )

    @property
    def options(self) -> tuple[str, ...]:
        """Empty option tuple for a common MCQ/numeric duck-typed interface."""

        return ()

    @property
    def labels(self) -> tuple[str, ...]:
        return ()

    @property
    def is_numeric(self) -> bool:
        return True

    @property
    def is_multiple_choice(self) -> bool:
        return False

    def render(self, transformed_to_original: Sequence[str] | None = None) -> str:
        """Render the unchanged identity view; permutations are not supported."""

        if transformed_to_original not in (None, (), []):
            raise ValueError("NumericExample supports identity view only")
        return f"{self.stem.rstrip()}\n"


DatasetExample: TypeAlias = MCQExample | NumericExample


def _normalise_source_label(value: Any) -> str:
    return str(value).strip().upper()


def _position_from_labels(labels: Sequence[Any], answer: Any) -> int:
    normalised = [_normalise_source_label(label) for label in labels]
    target = _normalise_source_label(answer)
    if target in normalised:
        return normalised.index(target)
    # Some mirrors of ARC expose answerKey as a 1-based numeric string even
    # though the choice labels are A/B/C/D.  Treat that representation as an
    # option position only after the direct-label lookup, so a dataset that
    # genuinely uses numeric labels keeps its own mapping.
    if re.fullmatch(r"[1-9][0-9]*", target):
        position = int(target) - 1
        if 0 <= position < len(labels):
            return position
    raise ValueError(f"Answer label {answer!r} not found in labels {labels!r}")


def _parse_mathqa_options(raw: str) -> tuple[list[str], list[str]]:
    matches = list(re.finditer(r"(?:^|,\s*)([A-Ea-e])\s*\)", raw))
    if len(matches) < 2:
        raise ValueError(f"Could not parse MathQA options: {raw!r}")
    labels: list[str] = []
    values: list[str] = []
    for index, match in enumerate(matches):
        labels.append(match.group(1).upper())
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        value = raw[match.end() : end].strip().strip(",").strip()
        if not value:
            raise ValueError(f"Empty MathQA option in {raw!r}")
        values.append(value)
    return labels, values


def _source_id(row: dict[str, Any], index: int) -> Any:
    """Find a stable source identifier across Hub mirrors."""

    for key in ("id", "idx", "index", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return index


def _question_id(
    dataset_name: str,
    split: str,
    source_id: Any,
    stem: str,
    *parts: str,
) -> str:
    return (
        f"{dataset_name}:{split}:{source_id}:"
        f"{stable_id(stem, *parts, length=12)}"
    )


def _convert_sciq(row: dict[str, Any]) -> tuple[str, tuple[str, ...], int]:
    """Match the original SciQ adapter's deterministic text sort.

    SciQ has no option labels.  The original CaTS handler combines three
    distractors and the correct answer, then sorts by option text before
    assigning A--D.  Sorting indexed pairs preserves the correct answer even
    if a mirror happens to contain duplicate option text.
    """

    required = ("question", "distractor1", "distractor2", "distractor3", "correct_answer")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"SciQ row missing fields: {missing}")
    raw_options = [row[key] for key in required[1:]]
    indexed = sorted(enumerate(raw_options), key=lambda item: str(item[1]))
    options = tuple(str(value).strip() for _, value in indexed)
    correct_positions = [position for position, (original, _) in enumerate(indexed) if original == 3]
    if len(correct_positions) != 1:
        raise ValueError("SciQ correct_answer is missing or duplicated unexpectedly")
    stem = f"Question: {str(row['question']).strip()}"
    return stem, options, correct_positions[0]


def _convert_winogrande(row: dict[str, Any]) -> tuple[str, tuple[str, ...], int]:
    """Convert WinoGrande's 1/2 answer convention to A/B positions."""

    required = ("sentence", "option1", "option2", "answer")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"WinoGrande row missing fields: {missing}")
    answer = row["answer"]
    token = str(answer).strip().upper()
    if token in {"1", "A"}:
        correct_index = 0
    elif token in {"2", "B"}:
        correct_index = 1
    else:
        raise ValueError(f"WinoGrande answer must be 1/2 (or A/B), got {answer!r}")
    options = (str(row["option1"]).strip(), str(row["option2"]).strip())
    stem = f"Question: {str(row['sentence']).strip()}"
    return stem, options, correct_index


def _convert_gsm8k(
    split: str, index: int, row: dict[str, Any]
) -> NumericExample:
    question = row.get("question", row.get("Question"))
    if question is None:
        raise KeyError("GSM8K row missing question")
    if "answer" not in row and "Answer" not in row:
        raise KeyError("GSM8K row missing answer")
    answer = row.get("answer", row.get("Answer"))
    stem = f"Question: {str(question).strip()}"
    return NumericExample(
        dataset_name="gsm8k",
        split=split,
        source_index=index,
        question_id=_question_id("gsm8k", split, _source_id(row, index), stem),
        stem=stem,
        correct_answer=_extract_gsm8k_gold(answer),
    )


def _convert_svamp(
    split: str, index: int, row: dict[str, Any]
) -> NumericExample:
    body = row.get("Body", row.get("body"))
    question = row.get("Question", row.get("question"))
    if body is None or question is None:
        raise KeyError("SVAMP row missing Body/Question")
    if "Answer" not in row and "answer" not in row:
        raise KeyError("SVAMP row missing Answer")
    # Keep the original CaTS concatenation (Body + Question) to preserve the
    # prompt distribution used by its released teacher data.  Do not silently
    # insert punctuation or a space here.
    stem = f"Question: {str(body)}{str(question)}".strip()
    answer = row.get("Answer", row.get("answer"))
    return NumericExample(
        dataset_name="svamp",
        split=split,
        source_index=index,
        question_id=_question_id("svamp", split, _source_id(row, index), stem),
        stem=stem,
        correct_answer=_normalise_numeric_value(answer, field_name="SVAMP answer"),
    )


def _convert_entry(
    dataset_name: str, split: str, index: int, row: dict[str, Any]
) -> DatasetExample:
    if dataset_name == "gsm8k":
        return _convert_gsm8k(split, index, row)
    if dataset_name == "svamp":
        return _convert_svamp(split, index, row)
    if dataset_name in {"arc_easy", "arc_challenge"}:
        labels = list(row["choices"]["label"])
        options = list(row["choices"]["text"])
        correct_index = _position_from_labels(labels, row["answerKey"])
        stem = f"Question: {row['question']}"
    elif dataset_name == "commonsense_qa":
        labels = list(row["choices"]["label"])
        options = list(row["choices"]["text"])
        correct_index = _position_from_labels(labels, row["answerKey"])
        stem = f"Question: {row['question']}"
    elif dataset_name == "openbookqa":
        labels = list(row["choices"]["label"])
        options = list(row["choices"]["text"])
        correct_index = _position_from_labels(labels, row["answerKey"])
        stem = f"Question: {row['question_stem']}"
    elif dataset_name == "reclor":
        options = list(row["answers"])
        correct_index = int(row["label"])
        stem = f"Passage:\n{row['context']}\n\nQuestion: {row['question']}"
    elif dataset_name == "logiqa":
        options = list(row["options"])
        raw_index = row["correct_option"]
        if isinstance(raw_index, str) and raw_index.strip().upper() in option_labels(len(options)):
            correct_index = option_labels(len(options)).index(raw_index.strip().upper())
        else:
            correct_index = int(raw_index)
        stem = f"Article:\n{row['context']}\n\nQuestion: {row['query']}"
    elif dataset_name == "math_qa":
        labels, options = _parse_mathqa_options(str(row["options"]))
        correct_index = _position_from_labels(labels, row["correct"])
        stem = f"Problem: {row['Problem']}"
    elif dataset_name == "sciq":
        stem, options, correct_index = _convert_sciq(row)
    elif dataset_name == "winogrande":
        stem, options, correct_index = _convert_winogrande(row)
    else:
        raise ValueError(f"Unsupported relational dataset: {dataset_name}")

    options_tuple = tuple(str(option).strip() for option in options)
    qid = _question_id(dataset_name, split, _source_id(row, index), stem, *options_tuple)
    return MCQExample(
        dataset_name=dataset_name,
        split=split,
        source_index=index,
        question_id=qid,
        stem=stem,
        options=options_tuple,
        correct_index=correct_index,
    )


def load_mcq_examples(
    dataset_name: str,
    split: str = "train",
    max_questions: int | None = None,
) -> list[MCQExample]:
    if dataset_name not in MCQ_DATASETS:
        raise ValueError(
            f"{dataset_name!r} is not an option-MCQ dataset; choose from {MCQ_DATASETS}"
        )
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be positive")
    handler = get_dataset(dataset_name)
    split_map, answer_type = handler.load_data()
    if answer_type != "option letter" and dataset_name != "math_qa":
        raise ValueError(f"{dataset_name} is not a multiple-choice dataset")
    if split not in split_map:
        raise KeyError(
            f"Split {split!r} unavailable for {dataset_name}; available={list(split_map)}"
        )
    source = split_map[split]
    limit = len(source) if max_questions is None else min(len(source), max_questions)
    examples: list[MCQExample] = []
    skipped: list[tuple[int, str]] = []
    for index in range(limit):
        try:
            examples.append(_convert_entry(dataset_name, split, index, dict(source[index])))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            skipped.append((index, str(exc)))
    if skipped:
        preview = "; ".join(f"{idx}: {reason}" for idx, reason in skipped[:3])
        raise ValueError(
            f"Failed to convert {len(skipped)}/{limit} {dataset_name} questions. "
            f"First failures: {preview}"
        )
    if not examples:
        raise ValueError(f"No examples loaded for {dataset_name}/{split}")
    return examples


def load_numeric_examples(
    dataset_name: str,
    split: str = "train",
    max_questions: int | None = None,
) -> list[NumericExample]:
    """Load GSM8K/SVAMP as scalar-answer identity-only examples."""

    if dataset_name not in NUMERIC_DATASETS:
        raise ValueError(
            f"{dataset_name!r} is not a numeric fallback dataset; "
            f"choose from {NUMERIC_DATASETS}"
        )
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be positive")
    handler = get_dataset(dataset_name)
    split_map, answer_type = handler.load_data()
    if answer_type != "number":
        raise ValueError(
            f"{dataset_name} handler reports {answer_type!r}, expected 'number'"
        )
    if split not in split_map:
        raise KeyError(
            f"Split {split!r} unavailable for {dataset_name}; available={list(split_map)}"
        )
    source = split_map[split]
    limit = len(source) if max_questions is None else min(len(source), max_questions)
    examples: list[NumericExample] = []
    skipped: list[tuple[int, str]] = []
    for index in range(limit):
        try:
            converted = _convert_entry(dataset_name, split, index, dict(source[index]))
            if not isinstance(converted, NumericExample):
                raise TypeError("numeric adapter returned a non-numeric example")
            examples.append(converted)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            skipped.append((index, str(exc)))
    if skipped:
        preview = "; ".join(f"{idx}: {reason}" for idx, reason in skipped[:3])
        raise ValueError(
            f"Failed to convert {len(skipped)}/{limit} {dataset_name} questions. "
            f"First failures: {preview}"
        )
    if not examples:
        raise ValueError(f"No examples loaded for {dataset_name}/{split}")
    return examples


def load_dataset_examples(
    dataset_name: str,
    split: str = "train",
    max_questions: int | None = None,
) -> list[DatasetExample]:
    """Dispatch to the option-MCQ or numeric adapter for one dataset.

    This is the preferred entry point for generation code.  It avoids relying
    on a fragile ``answer_type`` check in the original handlers (MathQA, for
    example, is labelled ``number`` there even though it is an MCQ dataset).
    """

    if dataset_name in NUMERIC_DATASETS:
        return load_numeric_examples(dataset_name, split, max_questions)
    if dataset_name in MCQ_DATASETS:
        return load_mcq_examples(dataset_name, split, max_questions)
    raise ValueError(
        f"Unsupported dataset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )


# A concise alias for callers that use the generic term ``load_examples``.
load_examples = load_dataset_examples


def generation_policy(dataset_name: str) -> dict[str, Any]:
    """Return the v1 relation/budget policy for a dataset.

    The values are metadata, not hidden global overrides.  A generator should
    use them as defaults and persist them in ``generation_metadata.json``.
    """

    if dataset_name in NUMERIC_DATASETS:
        return {
            "answer_type": "number",
            "relation_mode": "identity_only",
            "num_views": 1,
            "samples_per_view": 32,
            "total_budget": 32,
        }
    if dataset_name in MCQ_DATASETS:
        is_winogrande = dataset_name.strip().lower() == "winogrande"
        return {
            "answer_type": "option letter",
            "relation_mode": "option_permutation",
            "num_views": 2 if is_winogrande else 4,
            "samples_per_view": 16 if is_winogrande else 8,
            "total_budget": 32,
            "allow_repeated_views": False,
        }
    raise ValueError(
        f"Unsupported dataset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )
