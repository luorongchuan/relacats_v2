import unittest

from relacats_v2.core import (
    ControllerAction,
    annotate_dependency_weights,
    attach_full_targets,
    compute_fragility,
    compute_relssc_full,
    controller_state,
)


class FullRelaCaTSTests(unittest.TestCase):
    def test_dependency_cluster_total_matches_m_one_minus_beta(self):
        records = [
            {"canonicalized_answer": "A", "confidence": 1.0, "response": "same strategy Answer: A"},
            {"canonicalized_answer": "A", "confidence": 1.0, "response": "same strategy Answer: A"},
            {"canonicalized_answer": "A", "confidence": 1.0, "response": "same strategy Answer: A"},
            {"canonicalized_answer": "B", "confidence": 1.0, "response": "different Answer: B"},
        ]
        weighted, summary = annotate_dependency_weights(
            records, beta=1.0, similarity_threshold=0.9
        )
        a_weights = [row["dependency_weight"] for row in weighted[:3]]
        self.assertEqual(summary.cluster_count, 2)
        self.assertAlmostEqual(sum(a_weights), 1.0)

    def test_weighted_relssc_uses_dependency_weights(self):
        records = [
            {
                "canonicalized_answer": "A",
                "confidence": 1.0,
                "dependency_weight": 0.5,
                "relation_weight": 1.0,
            },
            {
                "canonicalized_answer": "A",
                "confidence": 1.0,
                "dependency_weight": 0.5,
                "relation_weight": 1.0,
            },
            {
                "canonicalized_answer": "B",
                "confidence": 1.0,
                "dependency_weight": 1.0,
                "relation_weight": 1.0,
            },
        ]
        result = compute_relssc_full(records)
        self.assertAlmostEqual(result.scores["A"], 0.5)
        self.assertAlmostEqual(result.scores["B"], 0.5)

    def test_fragility_detects_identity_support_that_collapses_across_views(self):
        records = [
            {
                "relation_id": "g0",
                "relation_type": "identity",
                "canonicalized_answer": "A",
                "confidence": 1.0,
                "dependency_weight": 1.0,
                "relation_weight": 1.0,
            },
            {
                "relation_id": "g1",
                "relation_type": "option_permutation",
                "canonicalized_answer": "B",
                "confidence": 1.0,
                "dependency_weight": 1.0,
                "relation_weight": 1.0,
            },
        ]
        fragility = compute_fragility(records, lambda_v=0.0)
        self.assertTrue(fragility.defined)
        self.assertAlmostEqual(fragility.identity_support["A"], 1.0)
        self.assertAlmostEqual(fragility.relational_support["A"], 0.5)
        self.assertAlmostEqual(fragility.scores["A"], 0.5)

    def test_full_target_builder_attaches_dependency_relssc_and_fragility(self):
        records = [
            {
                "relation_id": "g0",
                "view_index": 0,
                "relation_type": "identity",
                "canonicalized_answer": "A",
                "is_valid_answer": True,
                "confidence": 0.9,
                "response": "reason one Answer: A",
                "relation_weight": 1.0,
            },
            {
                "relation_id": "g1",
                "view_index": 1,
                "relation_type": "option_permutation",
                "canonicalized_answer": "A",
                "is_valid_answer": True,
                "confidence": 0.8,
                "response": "different reason Answer: B",
                "relation_weight": 1.0,
            },
        ]
        attached, relssc, fragility, dependency = attach_full_targets(records)
        self.assertTrue(relssc.defined)
        self.assertTrue(fragility.defined)
        self.assertGreaterEqual(dependency.cluster_count, 1)
        for row in attached:
            self.assertIn("dependency_weight", row)
            self.assertIn("relssc", row)
            self.assertIn("fragility_target", row)

    def test_controller_distinguishes_stop_sample_intervene(self):
        stop_records = [
            {"extracted_answer": "A", "confidence": 0.9, "fragility": 0.05, "response": "s1 Answer: A"},
            {"extracted_answer": "A", "confidence": 0.8, "fragility": 0.10, "response": "s2 Answer: A"},
        ]
        state = controller_state(
            stop_records, tau_support=0.7, tau_fragility=0.2, min_valid=2
        )
        self.assertEqual(state.action, ControllerAction.STOP)

        intervene_records = [
            {"extracted_answer": "A", "confidence": 0.9, "fragility": 0.8, "response": "s1 Answer: A"},
            {"extracted_answer": "A", "confidence": 0.8, "fragility": 0.7, "response": "s2 Answer: A"},
        ]
        state = controller_state(
            intervene_records, tau_support=0.7, tau_fragility=0.2, min_valid=2
        )
        self.assertEqual(state.action, ControllerAction.INTERVENE)

        sample_records = [
            {"extracted_answer": "A", "confidence": 0.6, "fragility": 0.1, "response": "s1 Answer: A"},
            {"extracted_answer": "B", "confidence": 0.6, "fragility": 0.1, "response": "s2 Answer: B"},
        ]
        state = controller_state(
            sample_records, tau_support=0.8, tau_fragility=0.2, min_valid=2
        )
        self.assertEqual(state.action, ControllerAction.SAMPLE)


if __name__ == "__main__":
    unittest.main()
