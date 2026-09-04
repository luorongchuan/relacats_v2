import unittest

from relacats_v2.core import (
    CanonicalizationStatus,
    InvalidAnswerError,
    InvalidPermutationError,
    OptionPermutation,
    canonicalize_answer,
    normalize_numeric_answer,
    normalize_option_answer,
)


class AnswerNormalizationTests(unittest.TestCase):
    def test_letter_and_numeric_option_answers_normalize_to_a_through_e(self):
        labels = tuple("ABCDE")
        self.assertEqual(
            normalize_option_answer("Answer: (e)", labels=labels).normalized_answer,
            "E",
        )
        self.assertEqual(
            normalize_option_answer("5", labels=labels).normalized_answer,
            "E",
        )
        self.assertEqual(
            normalize_option_answer(0, labels=labels, numeric_base=0).normalized_answer,
            "A",
        )

    def test_scalar_numeric_normalization_is_stable(self):
        cases = {
            "Answer: $1,000.00": "1000",
            "-0.0": "0",
            ".5000": "0.5",
            12: "12",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                result = normalize_numeric_answer(raw)
                self.assertTrue(result.valid)
                self.assertEqual(result.normalized_answer, expected)

    def test_invalid_answer_is_an_explicit_non_voting_state(self):
        missing = normalize_option_answer(None, labels=tuple("ABCD"))
        self.assertFalse(missing.valid)
        self.assertEqual(missing.status, CanonicalizationStatus.MISSING_ANSWER)
        outside = normalize_option_answer("E", labels=tuple("ABCD"))
        self.assertFalse(outside.valid)
        self.assertEqual(outside.status, CanonicalizationStatus.OUT_OF_ANSWER_SPACE)
        free_text = normalize_option_answer(
            "I think the answer is probably A", labels=tuple("ABCD")
        )
        self.assertFalse(free_text.valid)
        self.assertEqual(free_text.status, CanonicalizationStatus.INVALID_FORMAT)


class CanonicalizationTests(unittest.TestCase):
    def setUp(self):
        self.permutation = OptionPermutation.from_transformed_order([2, 0, 3, 1])

    def test_canonicalization_applies_inverse_not_forward(self):
        # Transformed A displays original C, hence A canonicalizes to C.
        result = canonicalize_answer("Answer: A", self.permutation)
        self.assertTrue(result.valid)
        self.assertEqual(result.normalized_transformed_answer, "A")
        self.assertEqual(result.canonicalized_answer, "C")

        # Verify every original answer survives a forward + inverse round trip.
        for original in "ABCD":
            transformed = self.permutation.forward_answer(original)
            result = canonicalize_answer(transformed, self.permutation)
            self.assertEqual(result.canonicalized_answer, original)

    def test_serialized_metadata_round_trip_and_record_fields(self):
        metadata = self.permutation.to_metadata()
        metadata.update({"relation_type": "option_permutation", "relation_id": "g2"})
        result = canonicalize_answer(1, metadata)
        self.assertEqual(result.canonicalized_answer, "C")
        fields = result.to_record_fields()
        self.assertEqual(fields["extracted_answer"], "A")
        self.assertEqual(fields["canonicalized_answer"], "C")
        self.assertTrue(fields["is_valid_answer"])

    def test_inconsistent_mapping_directions_are_rejected(self):
        metadata = self.permutation.to_metadata()
        metadata["inverse_permutation"] = {
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
        }
        with self.assertRaisesRegex(InvalidPermutationError, "inconsistent"):
            canonicalize_answer("A", metadata)

    def test_invalid_answer_can_be_skipped_or_raised(self):
        result = canonicalize_answer("Z", self.permutation)
        self.assertFalse(result.valid)
        self.assertIsNone(result.canonicalized_answer)
        with self.assertRaises(InvalidAnswerError):
            canonicalize_answer("Z", self.permutation, strict=True)

    def test_number_identity_and_unsupported_relation(self):
        number = canonicalize_answer(
            "Answer: 42.00", {"relation_type": "identity"}, answer_type="number"
        )
        self.assertEqual(number.canonicalized_answer, "42")
        unsupported = canonicalize_answer(
            "42", {"relation_type": "add_k"}, answer_type="number"
        )
        self.assertFalse(unsupported.valid)
        self.assertEqual(
            unsupported.status, CanonicalizationStatus.UNSUPPORTED_RELATION
        )


if __name__ == "__main__":
    unittest.main()
