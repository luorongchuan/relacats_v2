"""Unit tests for RelaCaTS v2 confidence-target semantics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from relacats_v2.core import (
    attach_v2_target_inputs,
    compute_ssc_target_context,
    resolve_confidence_target,
)
from relacats_v2.model_training.train_relacats import prepare_mixed_examples


def _sample(
    answer: str | None,
    *,
    relation_type: str,
    valid: bool = True,
    duplicate: bool = False,
) -> dict:
    return {
        "canonicalized_answer": answer,
        "canonicalization_status": "valid" if valid else "invalid",
        "is_valid_answer": valid,
        "relation_type": relation_type,
        "is_duplicate_view": duplicate,
    }


class ResidualTargetTests(unittest.TestCase):
    def test_three_modes_and_residual_formula(self):
        arguments = {
            "ssc": 0.9,
            "relssc": 0.5,
            "relation_valid_ratio": 0.8,
            "lambda_rel": 0.5,
        }
        self.assertEqual(
            resolve_confidence_target(**arguments, target_mode="ssc"), 0.9
        )
        self.assertEqual(
            resolve_confidence_target(**arguments, target_mode="relssc_replace"),
            0.5,
        )
        # 0.9 - 0.5 * 0.8 * max(0.9 - 0.5, 0) = 0.74
        self.assertAlmostEqual(
            resolve_confidence_target(**arguments, target_mode="residual"), 0.74
        )

    def test_residual_only_penalizes_downward_relational_evidence_and_clips(self):
        self.assertEqual(
            resolve_confidence_target(
                ssc=0.6,
                relssc=0.8,
                relation_valid_ratio=1.0,
                target_mode="residual",
            ),
            0.6,
        )
        self.assertEqual(
            resolve_confidence_target(
                ssc=1.0,
                relssc=0.0,
                relation_valid_ratio=1.0,
                target_mode="residual",
                lambda_rel=10.0,
            ),
            0.01,
        )
        self.assertEqual(
            resolve_confidence_target(
                ssc=1.0,
                relssc=1.0,
                relation_valid_ratio=1.0,
                target_mode="residual",
            ),
            0.99,
        )

    def test_identity_only_residual_degenerates_to_ssc(self):
        records = [
            _sample("12", relation_type="identity") for _ in range(24)
        ] + [
            _sample("13", relation_type="identity") for _ in range(8)
        ]
        annotated, context = attach_v2_target_inputs(records)
        self.assertEqual(context.total_relation_samples, 0)
        self.assertEqual(context.relation_valid_ratio, 0.0)
        self.assertAlmostEqual(annotated[0]["ssc"], 0.75)
        self.assertAlmostEqual(
            resolve_confidence_target(
                ssc=annotated[0]["ssc"],
                relssc=0.2,
                relation_valid_ratio=annotated[0]["relation_valid_ratio"],
                target_mode="residual",
            ),
            0.75,
        )

    def test_relation_valid_ratio_excludes_identity_and_duplicate_views(self):
        records = [
            _sample("A", relation_type="identity"),
            _sample("A", relation_type="identity"),
            _sample("A", relation_type="option_permutation"),
            _sample(None, relation_type="option_permutation", valid=False),
            _sample("A", relation_type="option_permutation", duplicate=True),
        ]
        context = compute_ssc_target_context(records)
        self.assertEqual(context.scores, {"A": 1.0})
        self.assertEqual(context.total_relation_samples, 2)
        self.assertEqual(context.valid_relation_samples, 1)
        self.assertEqual(context.relation_valid_ratio, 0.5)


class TrainingSelectionTests(unittest.TestCase):
    @staticmethod
    def _write_row(root: Path, row: dict) -> None:
        dataset_dir = root / "toy"
        dataset_dir.mkdir(parents=True)
        encoded = json.dumps(row, ensure_ascii=False) + "\n"
        (dataset_dir / "train.jsonl").write_text(encoded, encoding="utf-8")
        (dataset_dir / "test.jsonl").write_text(encoded, encoding="utf-8")

    @staticmethod
    def _config(root: Path, *, causal_ratio: float, target_mode: str) -> dict:
        return {
            "dataset_root": str(root),
            "target_mode": target_mode,
            "lambda_rel": 0.5,
            "causal_lm_ratio": causal_ratio,
            "threshold": 0.75,
            "datasets": [{"name": "toy", "weight": 1.0}],
        }

    def test_generation_filter_uses_ssc_not_relssc_or_residual_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_row(
                root,
                {
                    "question_id": "q1",
                    "transformed_prompt": "prompt ",
                    "response": "answer",
                    "ssc": 0.9,
                    "relssc": 0.1,
                    "relation_valid_ratio": 1.0,
                },
            )
            for mode in ("relssc_replace", "residual"):
                with self.subTest(target_mode=mode):
                    examples = prepare_mixed_examples(
                        self._config(root, causal_ratio=1.0, target_mode=mode),
                        "train",
                        requested_total=1,
                        seed=42,
                    )
                    self.assertEqual(len(examples), 1)
                    self.assertEqual(examples[0]["task"], "causal_lm")
                    self.assertEqual(examples[0]["selection_ssc"], 0.9)

    def test_calibration_loss_record_receives_residual_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_row(
                root,
                {
                    "question_id": "q1",
                    "transformed_prompt": "prompt ",
                    "response": "answer",
                    "ssc": 0.9,
                    "relssc": 0.5,
                    "relation_valid_ratio": 0.8,
                },
            )
            examples = prepare_mixed_examples(
                self._config(root, causal_ratio=0.0, target_mode="residual"),
                "train",
                requested_total=1,
                seed=42,
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["task"], "calibration")
            self.assertAlmostEqual(examples[0]["target"], 0.74)
            self.assertEqual(examples[0]["target_mode"], "residual")


if __name__ == "__main__":
    unittest.main()
