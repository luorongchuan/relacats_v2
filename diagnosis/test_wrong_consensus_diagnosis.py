"""CPU-only synthetic checks for the wrong-consensus diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from relacats_v2.core import generate_option_permutation_views
from relacats_v2.diagnosis.wrong_consensus_diagnosis import (
    compare_candidates_with_relational_data,
    identify_high_ssc_wrong_candidates,
    parse_question_and_options,
    weighted_consensus,
)


class WrongConsensusDiagnosisTest(unittest.TestCase):
    def test_weighted_consensus_excludes_invalid_answer(self) -> None:
        result = weighted_consensus(["A", "A", None, "B"], [0.9, 0.8, 1.0, 0.1])
        self.assertEqual(result.answer, "A")
        self.assertAlmostEqual(result.score or 0.0, 1.7 / 1.8)
        self.assertEqual(result.valid_count, 3)
        self.assertEqual(result.invalid_count, 1)

    def test_prompt_parser_ignores_chat_wrapper(self) -> None:
        record = {
            "prompt": (
                "<|im_start|>system\nExample A. is not an option<|im_end|>\n"
                "<|im_start|>user\nQuestion: Question: Which is correct?\n"
                "Options:\nA. alpha\nB. beta\nC. gamma\nD. delta\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
        }
        question, options = parse_question_and_options(record)
        self.assertEqual(question, "Which is correct?")
        self.assertEqual(options, ["alpha", "beta", "gamma", "delta"])

    def test_identify_then_compare_breaks_wrong_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_path = root / "original.json"
            prompt = (
                "<|im_start|>user\nQuestion: Question: Pick beta.\nOptions:\n"
                "A. alpha\nB. beta\nC. gamma\nD. delta\n<|im_end|>"
            )
            original_path.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "index": 7,
                                "prompt": prompt,
                                "most_common_response_c": "A",
                                "consistency_score_c": 0.95,
                                "correct_answer": "B",
                                "responses": [
                                    "Explanation: x\nAnswer: A",
                                    "Explanation: y\nAnswer: A",
                                ],
                                "confidence": [0.95, 0.95],
                            },
                            {
                                "index": 8,
                                "prompt": prompt,
                                "most_common_response_c": "B",
                                "consistency_score_c": 0.99,
                                "correct_answer": "B",
                                "responses": ["Answer: B"],
                                "confidence": [0.99],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidates, identify_summary = identify_high_ssc_wrong_candidates(
                original_path, dataset_name="arc_challenge", threshold=0.9
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(identify_summary["raw_high_ssc_wrong_count"], 1)
            candidate = candidates[0]

            views = generate_option_permutation_views(
                candidate["original_question"],
                candidate["options"],
                num_views=4,
                samples_per_view=8,
                total_budget=32,
                seed=9,
            )
            samples = []
            for view_index, view in enumerate(views):
                canonical_answer = "A" if view_index == 0 else "B"
                transformed = view.option_permutation.forward_answer(canonical_answer)
                for sample_index in range(2):
                    samples.append(
                        {
                            "sample_id": f"g{view_index}-{sample_index}",
                            "question_id": candidate["question_id"],
                            "dataset_name": "arc_challenge",
                            "relation_id": view.relation_id,
                            "relation_type": view.relation_type,
                            "permutation": view.permutation,
                            "inverse_permutation": view.inverse_permutation,
                            "option_labels": list("ABCD"),
                            "extracted_answer": transformed,
                            # Deliberately omit canonicalized_answer: compare()
                            # must apply phi_g^{-1} itself.
                            "confidence": 1.0,
                        }
                    )
            question_dir = root / "relational" / "arc_challenge" / "questions"
            question_dir.mkdir(parents=True)
            (question_dir / "question.json").write_text(
                json.dumps(
                    {
                        "question_id": candidate["question_id"],
                        "source_index": 7,
                        "dataset_name": "arc_challenge",
                        "original_question": candidate["original_question"],
                        "samples": samples,
                    }
                ),
                encoding="utf-8",
            )

            cases, summary = compare_candidates_with_relational_data(
                candidates, root / "relational"
            )
            self.assertEqual(summary["evaluated_count"], 1)
            self.assertEqual(summary["wrong_consensus_broken_count"], 1)
            self.assertTrue(cases[0]["wrong_consensus_broken"])
            self.assertEqual(cases[0]["relssc_top_answer"], "B")
            self.assertAlmostEqual(cases[0]["relssc_wrong_answer"], 0.25)
            self.assertAlmostEqual(cases[0]["absolute_ssc_drop"], 0.70)
            self.assertFalse(cases[0]["relssc_inputs_use_gold"])


if __name__ == "__main__":
    unittest.main()
