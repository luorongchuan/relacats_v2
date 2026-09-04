import copy
import unittest

from relacats_v2.core import (
    InvalidConfidenceError,
    UnsupportedV1WeightError,
    ZeroTotalWeightError,
    attach_relssc_targets,
    compute_relssc,
)


class RelSSCTests(unittest.TestCase):
    def test_confidence_weighting_uses_all_samples_across_all_views(self):
        records = [
            {"relation_id": "g0", "canonicalized_answer": "A", "confidence": 0.9},
            {"relation_id": "g0", "canonicalized_answer": "B", "confidence": 0.1},
            {"relation_id": "g1", "canonicalized_answer": "B", "confidence": 0.8},
            {"relation_id": "g2", "canonicalized_answer": "A", "confidence": 0.2},
        ]
        result = compute_relssc(records)
        self.assertTrue(result.defined)
        self.assertAlmostEqual(result.total_weight, 2.0)
        self.assertAlmostEqual(result.scores["A"], 0.55)
        self.assertAlmostEqual(result.scores["B"], 0.45)
        self.assertEqual(result.targets, (0.55, 0.45, 0.45, 0.55))
        self.assertEqual(result.top_answer, "A")

    def test_global_denominator_is_not_an_unweighted_mean_of_view_scores(self):
        # Unequal view sizes make the distinction observable.  g0 contributes
        # one high-confidence A; g1 contributes three low-confidence B votes.
        # The answer score must be 0.4/(0.4+0.3)=4/7, not the unweighted mean
        # of per-view scores (0.5).
        result = compute_relssc(
            [
                {"relation_id": "g0", "canonicalized_answer": "A", "confidence": 0.4},
                {"relation_id": "g1", "canonicalized_answer": "B", "confidence": 0.1},
                {"relation_id": "g1", "canonicalized_answer": "B", "confidence": 0.1},
                {"relation_id": "g1", "canonicalized_answer": "B", "confidence": 0.1},
            ]
        )
        self.assertAlmostEqual(result.scores["A"], 4.0 / 7.0)
        self.assertAlmostEqual(result.scores["B"], 3.0 / 7.0)

    def test_invalid_answers_are_excluded_from_numerator_and_denominator(self):
        records = [
            {"canonicalized_answer": "A", "confidence": 0.25},
            {
                "canonicalized_answer": None,
                "confidence": 1.0,
                "canonicalization_status": "missing_answer",
                "is_valid_answer": False,
            },
            {"canonicalized_answer": "B", "confidence": 0.75},
        ]
        result = compute_relssc(records)
        self.assertEqual(result.valid_sample_count, 2)
        self.assertEqual(result.invalid_sample_count, 1)
        self.assertAlmostEqual(result.total_weight, 1.0)
        self.assertEqual(result.scores, {"A": 0.25, "B": 0.75})
        self.assertEqual(result.targets, (0.25, None, 0.75))

    def test_serialized_false_valid_flag_is_not_treated_as_truthy(self):
        result = compute_relssc(
            [
                {"canonicalized_answer": "A", "confidence": 1.0},
                {
                    "canonicalized_answer": "B",
                    "confidence": 1.0,
                    "is_valid_answer": "false",
                },
            ]
        )
        self.assertEqual(result.scores, {"A": 1.0})
        self.assertEqual(result.targets, (1.0, None))

    def test_zero_weight_has_explicit_raise_or_skip_policy(self):
        records = [
            {"canonicalized_answer": "A", "confidence": 0.0},
            {"canonicalized_answer": None, "confidence": None, "valid": False},
        ]
        with self.assertRaises(ZeroTotalWeightError):
            compute_relssc(records)
        skipped = compute_relssc(records, zero_weight_policy="skip")
        self.assertFalse(skipped.defined)
        self.assertEqual(skipped.targets, (None, None))
        self.assertIn("skip", skipped.reason)

    def test_v1_rejects_non_unit_relation_or_dependency_weights(self):
        with self.assertRaisesRegex(UnsupportedV1WeightError, "r_g=1"):
            compute_relssc(
                [
                    {
                        "canonicalized_answer": "A",
                        "confidence": 0.5,
                        "relation_weight": 0.8,
                    }
                ]
            )
        with self.assertRaisesRegex(UnsupportedV1WeightError, "d_gi=1"):
            compute_relssc(
                [
                    {
                        "canonicalized_answer": "A",
                        "confidence": 0.5,
                        "dependency_weight": 0.8,
                    }
                ]
            )

    def test_invalid_confidence_on_valid_answer_is_not_silently_skipped(self):
        with self.assertRaises(InvalidConfidenceError):
            compute_relssc(
                [{"canonicalized_answer": "A", "confidence": 1.2}]
            )

    def test_attach_targets_is_pure_and_writes_both_training_field_names(self):
        records = [
            {"canonicalized_answer": "A", "confidence": 0.4},
            {"canonicalized_answer": "B", "confidence": 0.6},
        ]
        before = copy.deepcopy(records)
        attached = attach_relssc_targets(records)
        self.assertEqual(records, before)
        self.assertAlmostEqual(attached[0]["relssc"], 0.4)
        self.assertAlmostEqual(attached[0]["relational_consistency"], 0.4)
        self.assertAlmostEqual(attached[1]["relssc"], 0.6)


if __name__ == "__main__":
    unittest.main()
