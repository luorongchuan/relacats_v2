from __future__ import annotations

import unittest

from relacats_v2.data_creation.build_full_relacats_strict import (
    _flatten_full,
    _strict_reparse_payload,
)
from relacats_v2.evaluation.answer_parsing import extract_explicit_answer
from relacats_v2.model_training.train_full_relacats_masked import fragility_suffix


class StrictFullMigrationTests(unittest.TestCase):
    def test_malformed_thousands_grouping_is_rejected_not_truncated(self):
        self.assertIsNone(
            extract_explicit_answer(
                "svamp", "Answer: 1,201565", answer_type="number"
            )
        )
        self.assertEqual(
            extract_explicit_answer(
                "svamp", "Answer: 1,201,565", answer_type="number"
            ),
            "1201565",
        )

    def test_strict_reparse_overwrites_old_last_number_fallback(self):
        payload = {
            "question_id": "svamp:test:1",
            "dataset_name": "svamp",
            "answer_type": "number",
            "samples": [
                {
                    "question_id": "svamp:test:1",
                    "dataset_name": "svamp",
                    "answer_type": "number",
                    "relation_id": "g0",
                    "view_index": 0,
                    "relation_type": "identity",
                    "response": "Work used 270.\nAnswer: 54 grades.",
                    "extracted_answer": "270",
                    "canonicalized_answer": "270",
                    "canonicalization_status": "valid",
                    "is_valid_answer": True,
                }
            ],
        }
        reparsed, counters = _strict_reparse_payload(payload)
        sample = reparsed["samples"][0]
        self.assertEqual(sample["extracted_answer"], "54")
        self.assertEqual(sample["canonicalized_answer"], "54")
        self.assertTrue(sample["is_valid_answer"])
        self.assertEqual(counters["canonical_changed"], 1)

    def test_identity_only_numeric_keeps_q_label_but_masks_fragility(self):
        payload = {
            "question_id": "gsm8k:test:1",
            "dataset_name": "gsm8k",
            "answer_type": "number",
            "attempted_budget": 2,
            "samples": [
                {
                    "question_id": "gsm8k:test:1",
                    "dataset_name": "gsm8k",
                    "answer_type": "number",
                    "relation_id": "g0",
                    "view_index": 0,
                    "relation_type": "identity",
                    "relation_weight": 1.0,
                    "dependency_weight": 1.0,
                    "confidence": 0.8,
                    "response": "Explanation: x\nAnswer: 12",
                    "transformed_prompt": "Question: x\n",
                    "is_valid_answer": True,
                    "canonicalized_answer": "12",
                    "canonicalization_status": "valid",
                },
                {
                    "question_id": "gsm8k:test:1",
                    "dataset_name": "gsm8k",
                    "answer_type": "number",
                    "relation_id": "g0",
                    "view_index": 0,
                    "relation_type": "identity",
                    "relation_weight": 1.0,
                    "dependency_weight": 1.0,
                    "confidence": 0.2,
                    "response": "Explanation: y\nAnswer: 13",
                    "transformed_prompt": "Question: x\n",
                    "is_valid_answer": True,
                    "canonicalized_answer": "13",
                    "canonicalization_status": "valid",
                },
            ],
        }
        rows, summary = _flatten_full(
            payload, beta=0.5, similarity_threshold=0.86, lambda_v=0.5
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(summary["fragility_available"])
        self.assertTrue(all(row["relssc"] is not None for row in rows))
        self.assertTrue(all(row["fragility_target"] is None for row in rows))
        self.assertTrue(all(not row["fragility_available"] for row in rows))

    def test_fragility_query_uses_model_family_chat_boundary(self):
        self.assertIn("<|im_start|>user", fragility_suffix("Qwen2.5-7B-Instruct"))
        self.assertIn("<|start_header_id|>user", fragility_suffix("Llama-3.1-8B-Instruct"))
        self.assertIn("<｜User｜>", fragility_suffix("DeepSeek-R1-Distill-Qwen-1.5B"))


if __name__ == "__main__":
    unittest.main()
