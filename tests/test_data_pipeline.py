from __future__ import annotations

import math
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from relacats_v2.common import atomic_write_json, atomic_write_jsonl, read_jsonl
from relacats_v2.core import compute_relssc
from relacats_v2.data_creation.build_relssc_dataset import build_dataset
from relacats_v2.data_creation.build_relssc_dataset import flatten_question
from relacats_v2.data_creation.build_relssc_dataset import validate_question_payload
from relacats_v2.data_creation.dataset_adapter import MCQExample
from relacats_v2.data_creation.generate_relational_data import (
    extract_option_answer,
    extract_numeric_answer,
    generate_question_batch,
    load_candidate_examples,
)


class FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        del kwargs
        return "\n".join(f"<{row['role']}>{row['content']}" for row in chat)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def generate(self, prompts, params, use_tqdm=False):
        del use_tqdm
        if getattr(params, "n", 1) > 1:
            outputs = []
            for prompt in prompts:
                # Return the transformed-space label currently attached to the
                # semantic option "correct". Canonicalization must recover C.
                answer = None
                for line in prompt.splitlines():
                    if "correct" in line and len(line) >= 2 and line[1] == ".":
                        answer = line[0]
                if answer is None:
                    raise AssertionError(f"test option not found in prompt: {prompt}")
                candidates = [
                    SimpleNamespace(
                        text=f"Explanation: test\nAnswer: {answer}",
                        token_ids=[1, 2],
                        finish_reason="stop",
                    )
                    for _ in range(params.n)
                ]
                outputs.append(SimpleNamespace(outputs=candidates))
            return outputs
        yes = SimpleNamespace(decoded_token="Yes", logprob=math.log(0.8))
        no = SimpleNamespace(decoded_token="No", logprob=math.log(0.2))
        candidate = SimpleNamespace(logprobs=[{1: yes, 2: no}], text="Yes")
        return [SimpleNamespace(outputs=[candidate]) for _ in prompts]


class FakeNumericLLM:
    def generate(self, prompts, params, use_tqdm=False):
        del use_tqdm
        if getattr(params, "n", 1) > 1:
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="Explanation: arithmetic\nAnswer: 42",
                            token_ids=[1, 2],
                            finish_reason="stop",
                        )
                        for _ in range(params.n)
                    ]
                )
                for _ in prompts
            ]
        yes = SimpleNamespace(decoded_token="Yes", logprob=math.log(0.7))
        no = SimpleNamespace(decoded_token="No", logprob=math.log(0.3))
        candidate = SimpleNamespace(logprobs=[{1: yes, 2: no}], text="Yes")
        return [SimpleNamespace(outputs=[candidate]) for _ in prompts]


class MixedFakeLLM:
    """Fake backend that makes profile separation observable in one batch."""

    def __init__(self):
        self.generation_n_values = []

    def generate(self, prompts, params, use_tqdm=False):
        del use_tqdm
        n = getattr(params, "n", 1)
        if n > 1:
            self.generation_n_values.append(n)
            outputs = []
            for prompt in prompts:
                if "number (e.g., 1)" in prompt:
                    text = "Explanation: arithmetic\nAnswer: 42"
                else:
                    answer = None
                    for line in prompt.splitlines():
                        if "correct" in line and len(line) >= 2 and line[1] == ".":
                            answer = line[0]
                    if answer is None:
                        raise AssertionError(f"test option not found in prompt: {prompt}")
                    text = f"Explanation: test\nAnswer: {answer}"
                outputs.append(
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text=text,
                                token_ids=[1, 2],
                                finish_reason="stop",
                            )
                            for _ in range(n)
                        ]
                    )
                )
            return outputs
        yes = SimpleNamespace(decoded_token="Yes", logprob=math.log(0.8))
        no = SimpleNamespace(decoded_token="No", logprob=math.log(0.2))
        candidate = SimpleNamespace(logprobs=[{1: yes, 2: no}], text="Yes")
        return [SimpleNamespace(outputs=[candidate]) for _ in prompts]


def generation_args() -> Namespace:
    return Namespace(
        seed=42,
        num_views=4,
        samples_per_view=2,
        total_budget=8,
        max_new_tokens=64,
        temperature=0.8,
        confidence_temperature=0.0,
        relation_mode="auto",
        allow_nonstandard_budget=False,
    )


class RelationalDataPipelineTests(unittest.TestCase):
    def test_answer_extraction_is_explicit_and_uses_last_marker(self):
        self.assertEqual(extract_option_answer("Answer: A\nFinal Answer: (D)", 4), "D")
        self.assertEqual(extract_option_answer("Answer: 2", 4), "B")
        self.assertIsNone(extract_option_answer("I think the letter is A", 4))

    def test_numeric_extraction_prefers_final_marker(self):
        self.assertEqual(
            extract_numeric_answer("Step 1: 2+2\nFinal Answer: $4.00"), "4.00"
        )
        self.assertIsNone(extract_numeric_answer("The result is 42"))
        self.assertEqual(extract_numeric_answer(r"Answer: \boxed{42}"), "42")
        self.assertEqual(extract_numeric_answer("Answer: forty-two"), "42")
        self.assertIsNone(extract_numeric_answer("no scalar answer"))

    def test_numeric_identity_generation_uses_one_view_and_32_samples(self):
        example = SimpleNamespace(
            dataset_name="gsm8k",
            split="train",
            source_index=0,
            question_id="gsm8k:train:unit",
            stem="Question: What is 6*7?",
            options=(),
            correct_answer="42",
            answer_type="number",
        )
        payload = generate_question_batch(
            examples=[example],
            llm=FakeNumericLLM(),
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )[0]
        self.assertEqual(payload["num_views"], 1)
        self.assertEqual(payload["samples_per_view"], 32)
        self.assertEqual(payload["attempted_budget"], 32)
        self.assertEqual(payload["answer_type"], "number")
        self.assertEqual(payload["relation_mode"], "identity_only")
        self.assertEqual(len(payload["samples"]), 32)
        self.assertTrue(all(row["canonicalized_answer"] == "42" for row in payload["samples"]))
        self.assertTrue(all(row["permutation"] is None for row in payload["samples"]))

    def test_winogrande_generation_uses_two_unique_views_and_16_each(self):
        example = MCQExample(
            dataset_name="winogrande",
            split="train",
            source_index=0,
            question_id="winogrande:train:unit",
            stem="Question: Which option is marked correct?",
            options=("correct", "not this one"),
            correct_index=0,
        )
        llm = MixedFakeLLM()
        payload = generate_question_batch(
            examples=[example],
            llm=llm,
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )[0]
        # The test args intentionally use an 8-response smoke budget for
        # ordinary MCQ examples; WinoGrande's formal profile is still fixed
        # at 2x16=32 when allow_nonstandard_budget is false.
        self.assertEqual(llm.generation_n_values, [16])
        self.assertEqual(payload["num_views"], 2)
        self.assertEqual(payload["samples_per_view"], 16)
        self.assertEqual(payload["attempted_budget"], 32)
        self.assertEqual(
            {row["relation_id"] for row in payload["samples"]}, {"g0", "g1"}
        )
        self.assertEqual(
            Counter(row["relation_id"] for row in payload["samples"]),
            Counter({"g0": 16, "g1": 16}),
        )
        self.assertFalse(any(row["is_duplicate_view"] for row in payload["samples"]))
        self.assertEqual(
            {tuple(row["permutation"][label] for label in ("A", "B")) for row in payload["samples"]},
            {("A", "B"), ("B", "A")},
        )
        validate_question_payload(payload, allow_nonstandard_budget=False)

        duplicate = dict(payload)
        duplicate["samples"] = [
            {**sample, "is_duplicate_view": True}
            if sample["relation_id"] == "g1"
            else dict(sample)
            for sample in payload["samples"]
        ]
        with self.assertRaisesRegex(ValueError, "duplicate view"):
            validate_question_payload(duplicate, allow_nonstandard_budget=False)

    def test_wino_and_ordinary_mcq_sampling_profiles_are_not_mixed(self):
        wino = MCQExample(
            dataset_name="winogrande",
            split="train",
            source_index=0,
            question_id="winogrande:train:group",
            stem="Question: Which option is marked correct?",
            options=("correct", "not this one"),
            correct_index=0,
        )
        mcq = MCQExample(
            dataset_name="arc_easy",
            split="train",
            source_index=1,
            question_id="arc_easy:train:group",
            stem="Question: Which option is marked correct?",
            options=("correct", "bravo", "charlie", "delta"),
            correct_index=0,
        )
        llm = MixedFakeLLM()
        payloads = generate_question_batch(
            examples=[wino, mcq],
            llm=llm,
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )
        self.assertEqual([payload["question_id"] for payload in payloads], [wino.question_id, mcq.question_id])
        self.assertEqual(sorted(llm.generation_n_values), [2, 16])

    def test_mixed_batch_groups_numeric_1x32_and_mcq_4x2_sampling_calls(self):
        mcq = MCQExample(
            dataset_name="arc_easy",
            split="train",
            source_index=0,
            question_id="arc_easy:train:mixed",
            stem="Question: Which option is marked correct?",
            options=("alpha", "correct", "charlie", "delta"),
            correct_index=1,
        )
        numeric = SimpleNamespace(
            dataset_name="gsm8k",
            split="train",
            source_index=1,
            question_id="gsm8k:train:mixed",
            stem="Question: What is 6*7?",
            options=(),
            correct_answer="42",
            answer_type="number",
        )
        llm = MixedFakeLLM()
        payloads = generate_question_batch(
            examples=[mcq, numeric],
            llm=llm,
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )
        self.assertEqual([payload["question_id"] for payload in payloads], [
            mcq.question_id,
            numeric.question_id,
        ])
        self.assertEqual(llm.generation_n_values, [2, 32])
        self.assertEqual(payloads[0]["samples_per_view"], 2)
        self.assertEqual(payloads[1]["samples_per_view"], 32)

    def test_generation_canonicalization_relssc_and_builder(self):
        example = MCQExample(
            dataset_name="arc_easy",
            split="train",
            source_index=0,
            question_id="arc_easy:train:unit",
            stem="Question: Which option is marked correct?",
            options=("alpha", "bravo", "correct", "delta"),
            correct_index=2,
        )
        payload = generate_question_batch(
            examples=[example],
            llm=FakeLLM(),
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )[0]
        self.assertEqual(payload["attempted_budget"], 8)
        self.assertEqual(payload["valid_response_count"], 8)
        self.assertEqual({row["relation_id"] for row in payload["samples"]}, {"g0", "g1", "g2", "g3"})
        self.assertTrue(all(row["canonicalized_answer"] == "C" for row in payload["samples"]))
        # The persisted index and both mapping directions must agree for every
        # view.  This catches the common mistake of treating transformed-order
        # indices as the forward (original -> transformed) map.
        for sample in payload["samples"]:
            relation_id = sample["relation_id"]
            self.assertEqual(sample["view_index"], int(relation_id[1:]))
            forward = sample["permutation"]
            inverse = sample["inverse_permutation"]
            self.assertEqual(
                {inverse[transformed] for transformed in inverse},
                set(forward),
            )
            for original, transformed in forward.items():
                self.assertEqual(inverse[transformed], original)
            self.assertEqual(
                tuple(sample["transformed_options"]),
                tuple(
                    sample["original_options"][
                        "ABCDE".index(inverse[label])
                    ]
                    for label in sample["option_labels"]
                ),
            )
        result = compute_relssc(payload["samples"])
        self.assertAlmostEqual(result.scores["C"], 1.0)

        # RelSSC targets are gold-free: changing the diagnostic gold field must
        # not change either the question scores or per-response targets.
        altered_gold = dict(payload)
        altered_gold["gold_original_answer"] = "A"
        altered_gold["samples"] = [
            {**sample, "gold_original_answer": "A"}
            for sample in payload["samples"]
        ]
        original_rows, original_summary = flatten_question(payload)
        altered_rows, altered_summary = flatten_question(altered_gold)
        self.assertEqual(
            [row["relational_consistency"] for row in original_rows],
            [row["relational_consistency"] for row in altered_rows],
        )
        self.assertEqual(
            [row["ssc"] for row in original_rows],
            [row["ssc"] for row in altered_rows],
        )
        self.assertEqual(
            [row["relation_valid_ratio"] for row in original_rows],
            [row["relation_valid_ratio"] for row in altered_rows],
        )
        self.assertEqual(original_summary["scores"], altered_summary["scores"])
        self.assertFalse(original_summary.get("gold_used_in_target", False))

        # Create a second question so the group split has one train and one test
        # question. No response from a question may leak between the splits.
        second = dict(payload)
        second["question_id"] = "arc_easy:train:unit-2"
        second["source_index"] = 1
        second["samples"] = [
            {**sample, "question_id": second["question_id"], "sample_id": f"two-{i}"}
            for i, sample in enumerate(payload["samples"])
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qdir = root / "raw/arc_easy/questions"
            atomic_write_json(qdir / "000.json", payload)
            atomic_write_json(qdir / "001.json", second)
            stats = build_dataset(
                dataset_name="arc_easy",
                files=sorted(qdir.glob("*.json")),
                output_root=root / "built",
                test_ratio=0.5,
                seed=42,
                allow_nonstandard_budget=True,
            )
            train = list(read_jsonl(root / "built/arc_easy/train.jsonl"))
            test = list(read_jsonl(root / "built/arc_easy/test.jsonl"))
            self.assertEqual(stats["train_questions"], 1)
            self.assertEqual(stats["test_questions"], 1)
            self.assertFalse(
                {row["question_id"] for row in train}
                & {row["question_id"] for row in test}
            )
            self.assertTrue(
                all(row["target_provenance"] == "relssc_without_gold" for row in train + test)
            )
            self.assertTrue(
                all("gold_original_answer" in row for row in train + test),
                "gold is retained only as an audit field, not consumed by RelSSC",
            )

    def test_invalid_and_zero_weight_records_follow_raw_vs_training_policy(self):
        example = MCQExample(
            dataset_name="arc_easy",
            split="train",
            source_index=0,
            question_id="arc_easy:train:invalid-unit",
            stem="Question: Which option is marked correct?",
            options=("alpha", "bravo", "correct", "delta"),
            correct_index=2,
        )
        payload = generate_question_batch(
            examples=[example],
            llm=FakeLLM(),
            tokenizer=FakeTokenizer(),
            sampling_params_cls=FakeSamplingParams,
            model_name="Qwen2.5-test",
            args=generation_args(),
        )[0]
        payload["samples"][0]["canonicalized_answer"] = None
        payload["samples"][0]["extracted_answer"] = None
        payload["samples"][0]["is_valid_answer"] = False
        rows, summary = flatten_question(payload)
        self.assertEqual(summary["valid_response_count"], 7)
        self.assertEqual(summary["invalid_response_count"], 1)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["canonicalized_answer"] is not None for row in rows))

        zero_payload = dict(payload)
        zero_payload["samples"] = [
            {**sample, "confidence": 0.0}
            for sample in payload["samples"]
        ]
        zero_rows, zero_summary = flatten_question(zero_payload)
        self.assertEqual(zero_rows, [])
        self.assertFalse(zero_summary["defined"])
        self.assertIn("skip", zero_summary["reason"])

    def test_candidate_mode_strips_rendered_options_and_keeps_gold_out_of_target_path(self):
        # Diagnosis emits an original question plus options and a gold label.
        # Candidate loading should use gold only to identify the original
        # answer position; generation still receives the option contents.
        with tempfile.TemporaryDirectory() as temp:
            candidate_file = Path(temp) / "candidates.jsonl"
            atomic_write_jsonl(
                candidate_file,
                [
                    {
                        "question_id": "diag-1",
                        "dataset_name": "arc_easy",
                        "source_index": 4,
                        "original_question": (
                            "Question: Which one?\nOptions:\n"
                            "A. alpha\nB. bravo\nC. charlie\nD. delta"
                        ),
                        "options": ["alpha", "bravo", "charlie", "delta"],
                        "gold_original_answer": "C",
                    }
                ],
            )
            examples = load_candidate_examples(candidate_file, "train")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].stem, "Question: Which one?")
        self.assertEqual(examples[0].options[2], "charlie")
        self.assertEqual(examples[0].correct_answer, "C")


if __name__ == "__main__":
    unittest.main()
