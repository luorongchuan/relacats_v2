from __future__ import annotations

import unittest

from relacats_v2.evaluation.answer_parsing import (
    STRICT_EXPLICIT_ANSWER_PARSER_VERSION,
    extract_dataset_answer,
    extract_explicit_answer,
    extract_gold_answer,
    extract_mathqa_option_answer,
    parser_version,
)


class _RecordingHandler:
    """Small stand-in for a trusted upstream dataset handler."""

    def __init__(self, value=None):
        self.value = value
        self.calls = 0

    def extract_answer(self, text):
        del text
        self.calls += 1
        return self.value


class EvaluationAnswerParsingTests(unittest.TestCase):
    def test_option_spellings_required_by_protocol(self):
        cases = {
            "Answer: A": "A",
            "Answer: (a)": "A",
            "**Answer:** B": "B",
            r"Answer: \boxed{C}": "C",
            "Answer: 0.036 (d)": "D",
            "Explanation only says A": None,
            "I think the answer is probably A": None,
            "Answer: [A]": None,
            "Answer: 6": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_mathqa_option_answer(text), expected)

    def test_numeric_spellings_required_by_protocol(self):
        cases = {
            "Answer: 12": 12,
            r"Answer: \boxed{12}": 12,
            "**Answer:** three": 3,
            "Final Answer: twenty-one": 21,
            "There are three objects.": None,
            "Reasoning: 12\nAnswer:": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    extract_dataset_answer("object_counting", text), expected
                )

    def test_numeric_dataset_preserves_scalar_string_contract(self):
        self.assertEqual(extract_dataset_answer("gsm8k", "Answer: 12"), "12")
        self.assertEqual(extract_dataset_answer("gsm8k", "Answer: 12.00"), "12")
        self.assertEqual(extract_dataset_answer("svamp", "Answer: -3.5"), "-3.5")

    def test_last_valid_explicit_answer_wins(self):
        text = "Answer: A\nRevision: no\nAnswer: (C)"
        self.assertEqual(extract_dataset_answer("arc_challenge", text), "C")

        self.assertEqual(
            extract_dataset_answer("arc_challenge", "Answer: A; Answer: D"), "D"
        )

        # The final field is malformed, so the last *valid* explicit field is B.
        text = "Answer: A\nAnswer: B\nAnswer: unknown"
        self.assertEqual(extract_dataset_answer("arc_challenge", text), "B")

    def test_answer_field_does_not_spill_to_later_lines(self):
        text = "Answer:\nFurther reasoning mentions 99.\nNo final field follows."
        self.assertIsNone(extract_dataset_answer("gsm8k", text))
        self.assertIsNone(extract_dataset_answer("gsm8k", "notanswer: 99"))

    def test_arc_explicit_numeric_label_is_normalized(self):
        self.assertEqual(extract_dataset_answer("arc_easy", "Answer: (2)"), "B")

    def test_response_parser_never_calls_handler(self):
        handler = _RecordingHandler(value="A")
        self.assertIsNone(
            extract_dataset_answer(
                "object_counting", "The final number in this reasoning is 7", handler
            )
        )
        self.assertEqual(handler.calls, 0)

    def test_trusted_gold_has_separate_handler_path(self):
        handler = _RecordingHandler(value="42")
        self.assertEqual(extract_gold_answer("gsm8k", "work #### 42", handler), "42")
        self.assertEqual(handler.calls, 1)

    def test_mathqa_gold_uses_explicit_parser(self):
        handler = _RecordingHandler(value="Z")
        self.assertEqual(extract_gold_answer("math_qa", "Answer: (d)", handler), "D")
        self.assertEqual(handler.calls, 0)

    def test_answer_type_can_disambiguate_unknown_dataset(self):
        self.assertEqual(
            extract_explicit_answer("custom", "Answer: 3", answer_type="number"),
            "3",
        )
        self.assertEqual(
            extract_explicit_answer(
                "custom", "Answer: 0.036 (A)", answer_type="option letter"
            ),
            "A",
        )

    def test_all_datasets_share_one_parser_version(self):
        for dataset in ("math_qa", "object_counting", "arc_challenge", "gsm8k"):
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    parser_version(dataset), STRICT_EXPLICIT_ANSWER_PARSER_VERSION
                )


if __name__ == "__main__":
    unittest.main()
