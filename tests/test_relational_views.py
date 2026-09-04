import unittest

from relacats_v2.core import (
    OptionPermutation,
    RelationalViewError,
    SamplingBudgetError,
    generate_option_permutation_views,
    generate_identity_views,
    validate_sampling_budget,
)


class SamplingBudgetTests(unittest.TestCase):
    def test_default_budget_is_exactly_original_cats_budget(self):
        self.assertEqual(validate_sampling_budget(4, 8, 32), 32)

    def test_budget_mismatch_is_rejected(self):
        with self.assertRaisesRegex(SamplingBudgetError, "4.*32.*128"):
            validate_sampling_budget(4, 32, 32)
        with self.assertRaises(SamplingBudgetError):
            validate_sampling_budget(True, 8, 32)

    def test_numeric_identity_profile_is_explicitly_supported(self):
        self.assertEqual(
            validate_sampling_budget(1, 32, 32, relation_mode="identity"), 32
        )
        with self.assertRaises(SamplingBudgetError):
            validate_sampling_budget(2, 16, 32, relation_mode="identity")


class OptionPermutationTests(unittest.TestCase):
    def test_forward_and_inverse_directions_are_exact(self):
        # Transformed display: A=original C, B=original A,
        # C=original D, D=original B.
        permutation = OptionPermutation.from_transformed_order([2, 0, 3, 1])
        self.assertEqual(
            permutation.forward_mapping,
            {"A": "B", "B": "D", "C": "A", "D": "C"},
        )
        self.assertEqual(
            permutation.inverse_mapping,
            {"A": "C", "B": "A", "C": "D", "D": "B"},
        )
        self.assertEqual(
            permutation.permute_options(("alpha", "beta", "gamma", "delta")),
            ("gamma", "alpha", "delta", "beta"),
        )
        for original in "ABCD":
            transformed = permutation.forward_answer(original)
            self.assertEqual(permutation.inverse_answer(transformed), original)

    def test_generated_views_are_identity_first_unique_and_deterministic(self):
        kwargs = dict(
            question_stem="Which option is correct?",
            options=["alpha", "beta", "gamma", "delta"],
            num_views=4,
            samples_per_view=8,
            total_budget=32,
            seed=17,
        )
        first = generate_option_permutation_views(**kwargs)
        second = generate_option_permutation_views(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0].relation_id, "g0")
        self.assertEqual(first[0].relation_type, "identity")
        self.assertTrue(first[0].option_permutation.is_identity)
        self.assertEqual(
            len({view.option_permutation.forward_indices for view in first}), 4
        )
        self.assertTrue(
            all(view.samples_per_view == 8 for view in first)
        )
        self.assertTrue(
            all(
                "permutation" in view.to_metadata()
                and "inverse_permutation" in view.to_metadata()
                for view in first
            )
        )

    def test_impossible_number_of_unique_views_is_rejected(self):
        # Two options only admit two distinct permutations.
        with self.assertRaises(RelationalViewError):
            generate_option_permutation_views(
                "Binary question?",
                ["yes", "no"],
                num_views=4,
                samples_per_view=8,
                total_budget=32,
            )

    def test_two_option_repeated_views_are_explicit_generic_opt_in(self):
        """Keep the legacy escape hatch separate from the Wino profile."""

        views = generate_option_permutation_views(
            "Binary question?",
            ["yes", "no"],
            num_views=4,
            samples_per_view=8,
            total_budget=32,
            allow_repeated_views=True,
        )
        self.assertEqual(len(views), 4)
        self.assertEqual(
            [view.option_permutation.forward_indices for view in views],
            [(0, 1), (1, 0), (0, 1), (1, 0)],
        )
        self.assertEqual(
            [view.is_duplicate_view for view in views], [False, False, True, True]
        )
        self.assertTrue(all(view.to_metadata()["is_duplicate_view"] == view.is_duplicate_view for view in views))

    def test_two_option_wino_profile_has_only_identity_and_swap(self):
        views = generate_option_permutation_views(
            "Binary question?",
            ["yes", "no"],
            num_views=2,
            samples_per_view=16,
            total_budget=32,
            seed=42,
        )
        self.assertEqual(len(views), 2)
        self.assertEqual(
            [view.option_permutation.forward_indices for view in views],
            [(0, 1), (1, 0)],
        )
        self.assertEqual([view.relation_id for view in views], ["g0", "g1"])
        self.assertEqual([view.samples_per_view for view in views], [16, 16])
        self.assertEqual([view.is_duplicate_view for view in views], [False, False])

    def test_five_choice_questions_are_supported(self):
        views = generate_option_permutation_views(
            "Five-way question?",
            {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"},
        )
        self.assertEqual(views[0].option_permutation.labels, tuple("ABCDE"))
        for view in views:
            self.assertEqual(
                view.option_permutation.inverse_answer(
                    view.option_permutation.forward_answer("E")
                ),
                "E",
            )

    def test_numeric_identity_view_has_no_fake_option_permutation(self):
        views = generate_identity_views(
            "Solve 2+2.", samples_per_view=32, total_budget=32, seed=9
        )
        self.assertEqual(len(views), 1)
        view = views[0]
        self.assertEqual(view.relation_id, "g0")
        self.assertEqual(view.relation_type, "identity")
        self.assertEqual(view.relation_mode, "identity_only")
        self.assertEqual(view.answer_type, "number")
        self.assertIsNone(view.option_permutation)
        self.assertEqual(view.original_question, view.transformed_question)
        self.assertEqual(view.samples_per_view, 32)
        metadata = view.to_metadata()
        self.assertNotIn("permutation", metadata)
        self.assertEqual(metadata["answer_type"], "number")


if __name__ == "__main__":
    unittest.main()
