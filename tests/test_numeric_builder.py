from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from relacats_v2.common import atomic_write_json, read_jsonl
from relacats_v2.data_creation.build_relssc_dataset import (
    build_dataset,
    flatten_question,
    validate_question_payload,
)


def _numeric_payload(question_id: str, source_index: int = 0) -> dict:
    samples = []
    for i in range(32):
        answer = "2" if i < 16 else "3"
        samples.append(
            {
                "sample_id": f"{question_id}-s{i}",
                "question_id": question_id,
                "source_index": source_index,
                "dataset_name": "gsm8k",
                "split": "train",
                "relation_type": "identity",
                "relation_mode": "identity_only",
                "answer_type": "number",
                "relation_id": "g0",
                "view_index": 0,
                "sample_index_in_view": i,
                "original_question": "Question: Add two numbers.",
                "transformed_question": "Question: Add two numbers.",
                "original_options": [],
                "transformed_options": [],
                "option_labels": [],
                "permutation": None,
                "inverse_permutation": None,
                "response": f"Explanation: ... Answer: {answer}",
                "extracted_answer": answer,
                "canonicalized_answer": answer,
                "canonicalization_status": "valid",
                "is_valid_answer": True,
                "confidence": 0.5,
                "relation_weight": 1.0,
                "dependency_weight": 1.0,
            }
        )
    return {
        "schema_version": "relacats-v1.raw-question.1",
        "question_id": question_id,
        "source_index": source_index,
        "dataset_name": "gsm8k",
        "split": "train",
        "original_question": "Question: Add two numbers.",
        "original_options": [],
        "gold_original_answer": "2",
        "answer_type": "number",
        "relation_mode": "identity_only",
        "num_views": 1,
        "samples_per_view": 32,
        "attempted_budget": 32,
        "valid_response_count": 32,
        "invalid_response_count": 0,
        "samples": samples,
    }


class NumericBuilderTests(unittest.TestCase):
    def test_validates_and_flattens_identity_numeric_payload(self):
        payload = _numeric_payload("gsm-0")
        validated = validate_question_payload(payload, allow_nonstandard_budget=False)
        self.assertEqual(len(validated), 32)
        rows, summary = flatten_question(payload)
        self.assertEqual(len(rows), 32)
        self.assertTrue(summary["defined"])
        self.assertEqual(summary["scores"], {"2": 0.5, "3": 0.5})
        self.assertTrue(all(row["answer"] in {"2", "3"} for row in rows))
        self.assertTrue(all(row["ssc"] == 0.5 for row in rows))
        self.assertTrue(all(row["relation_valid_ratio"] == 0.0 for row in rows))
        self.assertEqual(summary["relation_valid_ratio"], 0.0)
        self.assertTrue(all(row["target_provenance"] == "relssc_without_gold" for row in rows))

    def test_builder_writes_numeric_train_test_without_option_permutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw/gsm8k/questions"
            first = _numeric_payload("gsm-0", 0)
            second = _numeric_payload("gsm-1", 1)
            atomic_write_json(raw / "000.json", first)
            atomic_write_json(raw / "001.json", second)
            stats = build_dataset(
                dataset_name="gsm8k",
                files=[raw / "000.json", raw / "001.json"],
                output_root=root / "built",
                test_ratio=0.5,
                seed=42,
                allow_nonstandard_budget=False,
            )
            self.assertEqual(stats["raw_response_records"], 64)
            self.assertEqual(stats["valid_training_records"], 64)
            train = list(read_jsonl(root / "built/gsm8k/train.jsonl"))
            test = list(read_jsonl(root / "built/gsm8k/test.jsonl"))
            self.assertEqual(len(train), 32)
            self.assertEqual(len(test), 32)
            self.assertTrue(all(row["answer_type"] == "number" for row in train + test))

    def test_numeric_payload_rejects_option_relation_and_wrong_formal_budget(self):
        payload = _numeric_payload("gsm-bad")
        payload["relation_mode"] = "option_permutation"
        with self.assertRaises(ValueError):
            validate_question_payload(payload, allow_nonstandard_budget=False)
        payload = _numeric_payload("gsm-bad-budget")
        payload["samples_per_view"] = 8
        payload["attempted_budget"] = 8
        payload["samples"] = payload["samples"][:8]
        with self.assertRaises(ValueError):
            validate_question_payload(payload, allow_nonstandard_budget=False)
        # A reduced identity budget is explicitly available for smoke tests.
        self.assertEqual(
            len(validate_question_payload(payload, allow_nonstandard_budget=True)),
            8,
        )


if __name__ == "__main__":
    unittest.main()
