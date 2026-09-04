from __future__ import annotations

import unittest
from unittest.mock import patch

from relacats_v2.data_creation import dataset_adapter as adapter


class _FakeHandler:
    def __init__(self, rows, answer_type):
        self.rows = rows
        self.answer_type = answer_type

    def load_data(self):
        return {"train": self.rows}, self.answer_type


class DatasetAdapterTests(unittest.TestCase):
    def test_sciq_matches_original_sorted_option_order(self):
        row = {
            "question": "Which part is exposed?",
            "distractor1": "the base",
            "distractor2": "the interior",
            "distractor3": "the top",
            "correct_answer": "the surface",
            "id": "sciq-1",
        }
        with patch.object(
            adapter,
            "get_dataset",
            return_value=_FakeHandler([row], "option letter"),
        ):
            examples = adapter.load_dataset_examples("sciq", max_questions=1)
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertIsInstance(example, adapter.MCQExample)
        self.assertEqual(
            example.options,
            ("the base", "the interior", "the surface", "the top"),
        )
        self.assertEqual(example.correct_answer, "C")
        self.assertEqual(example.answer_type, "option letter")
        self.assertEqual(example.relation_mode, "option_permutation")

    def test_winogrande_maps_one_based_answer_to_a_or_b(self):
        row = {
            "sentence": "Alex chose _ because it was newer.",
            "option1": "the blue one",
            "option2": "the red one",
            "answer": "2",
            "id": "wino-1",
        }
        with patch.object(
            adapter,
            "get_dataset",
            return_value=_FakeHandler([row], "option letter"),
        ):
            example = adapter.load_dataset_examples("winogrande")[0]
        self.assertEqual(example.options, ("the blue one", "the red one"))
        self.assertEqual(example.correct_answer, "B")
        self.assertEqual(example.labels, ("A", "B"))
        self.assertEqual(
            (example.num_views, example.samples_per_view, example.total_budget),
            (2, 16, 32),
        )
        self.assertFalse(example.allow_repeated_views)
        policy = adapter.generation_policy("winogrande")
        self.assertEqual(
            (policy["num_views"], policy["samples_per_view"], policy["total_budget"]),
            (2, 16, 32),
        )
        self.assertFalse(policy["allow_repeated_views"])

    def test_gsm8k_is_identity_only_and_extracts_hash_answer(self):
        row = {
            "question": "If there are 3 boxes of 4 apples, how many apples?",
            "answer": "There are 3 * 4 = 12 apples.\n#### 12",
            "id": "gsm-1",
        }
        with patch.object(
            adapter,
            "get_dataset",
            return_value=_FakeHandler([row], "number"),
        ):
            example = adapter.load_dataset_examples("gsm8k")[0]
        self.assertIsInstance(example, adapter.NumericExample)
        self.assertEqual(example.correct_answer, "12")
        self.assertEqual(example.answer_type, "number")
        self.assertEqual(example.relation_mode, "identity_only")
        self.assertEqual((example.num_views, example.samples_per_view, example.total_budget), (1, 32, 32))
        self.assertEqual(example.options, ())
        self.assertEqual(example.render(), "Question: If there are 3 boxes of 4 apples, how many apples?\n")

    def test_gsm8k_fallback_without_hash_uses_final_number(self):
        self.assertEqual(
            adapter._extract_gsm8k_gold("work gives 1,000.0 then final 2,000"),
            "2000",
        )

    def test_svamp_normalises_numeric_answer_and_preserves_prompt_concat(self):
        row = {
            "Body": "Mia has 10 books. ",
            "Question": "She buys 2 more. How many now?",
            "Answer": 12.0,
            "id": "svamp-1",
        }
        with patch.object(
            adapter,
            "get_dataset",
            return_value=_FakeHandler([row], "number"),
        ):
            example = adapter.load_dataset_examples("svamp")[0]
        self.assertEqual(example.correct_answer, "12")
        self.assertEqual(example.stem, "Question: Mia has 10 books. She buys 2 more. How many now?")
        self.assertEqual(adapter.generation_policy("svamp")["num_views"], 1)

    def test_mcq_loader_rejects_numeric_tasks_but_generic_loader_dispatches(self):
        with self.assertRaises(ValueError):
            adapter.load_mcq_examples("gsm8k")
        self.assertEqual(adapter.generation_policy("sciq")["samples_per_view"], 8)
        self.assertEqual(adapter.generation_policy("gsm8k")["samples_per_view"], 32)
        self.assertEqual(set(adapter.TRAIN_DATASETS), {
            "arc_easy", "commonsense_qa", "gsm8k", "logiqa", "openbookqa",
            "reclor", "sciq", "svamp", "winogrande",
        })


if __name__ == "__main__":
    unittest.main()
