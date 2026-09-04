"""CPU smoke tests for the full RelaCaTS q/f/CLM objective."""

from types import SimpleNamespace
import unittest

import torch

from relacats_v2.model_training.train_full_relacats import (
    FullBatchEncoder,
    full_task_loss,
)


class TinyTokenizer:
    eos_token = ""

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [1]

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            length = max(2, min(8, len(str(text).split()) + 1))
            rows.append([2 + index % 8 for index in range(length)])
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in rows], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            ),
        }


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=20):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        self.forward_calls += 1
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size) + self.bias
        if labels is None:
            loss = logits.sum() * 0.0
        else:
            valid = labels != -100
            loss = ((logits[..., 0][valid] - labels[valid].float()) ** 2).mean()
        return SimpleNamespace(logits=logits, loss=loss)


class FullTrainingObjectiveTests(unittest.TestCase):
    def test_joint_q_f_and_causal_loss_is_finite(self):
        model = TinyModel()
        encoder = FullBatchEncoder(TinyTokenizer(), "Qwen-test", 64, "cpu")
        batch = [
            {
                "task": "calibration",
                "question_id": "q1",
                "transformed_prompt": "prompt ",
                "response": "answer",
                "target": 0.7,
                "relssc": 0.8,
                "fragility_target": 0.2,
            },
            {
                "task": "causal_lm",
                "question_id": "q1",
                "transformed_prompt": "prompt ",
                "response": "answer",
                "target": -1.0,
            },
        ]
        loss, parts = full_task_loss(
            model,
            batch,
            encoder,
            smooth_l1_beta=0.25,
            causal_loss_scale=0.1,
            lambda_f=1.0,
            lambda_r=0.0,
        )
        self.assertEqual(model.forward_calls, 3)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("fragility_loss", parts)
        loss.backward()
        self.assertIsNotNone(model.bias.grad)

    def test_rank_loss_is_exposed_for_same_question_pair(self):
        model = TinyModel()
        encoder = FullBatchEncoder(TinyTokenizer(), "Qwen-test", 64, "cpu")
        batch = [
            {
                "task": "calibration",
                "question_id": "q1",
                "transformed_prompt": "p1 ",
                "response": "r1",
                "target": 0.8,
                "relssc": 0.9,
                "fragility_target": 0.1,
            },
            {
                "task": "calibration",
                "question_id": "q1",
                "transformed_prompt": "p2 ",
                "response": "r2",
                "target": 0.3,
                "relssc": 0.4,
                "fragility_target": 0.7,
            },
        ]
        loss, parts = full_task_loss(
            model,
            batch,
            encoder,
            smooth_l1_beta=0.25,
            causal_loss_scale=0.1,
            lambda_f=1.0,
            lambda_r=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(parts["ranking_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
