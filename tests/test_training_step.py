"""CPU micro-model smoke for the two RelaCaTS v2 training losses."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from relacats_v2.model_training.train_relacats import BatchEncoder, evaluate_loss, task_loss
from torch.utils.data import DataLoader


class TinyTokenizer:
    eos_token = ""

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [1]

    def __call__(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            texts = [texts]
        token_rows = []
        for text in texts:
            length = max(2, min(8, len(str(text).split()) + 1))
            token_rows.append([2 + index % 8 for index in range(length)])
        width = max(map(len, token_rows))
        input_ids = [row + [0] * (width - len(row)) for row in token_rows]
        attention = [[1] * len(row) + [0] * (width - len(row)) for row in token_rows]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


class TinyCausalModel(torch.nn.Module):
    def __init__(self, vocab_size=20):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.1))
        self.vocab_size = vocab_size
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.forward_calls += 1
        del attention_mask
        batch, length = input_ids.shape
        logits = torch.zeros(
            batch, length, self.vocab_size, dtype=torch.float32
        ) + self.bias
        if labels is None:
            loss = logits.sum() * 0.0
        else:
            valid = labels != -100
            loss = ((logits[..., 0][valid] - labels[valid].float()) ** 2).mean()
        return SimpleNamespace(logits=logits, loss=loss)


class TrainingStepSmokeTest(unittest.TestCase):
    def test_one_update_and_evaluation(self):
        tokenizer = TinyTokenizer()
        encoder = BatchEncoder(tokenizer, "Qwen2.5-test", 64, "cpu")
        model = TinyCausalModel()
        records = [
            {
                "task": "calibration",
                "transformed_prompt": "prompt ",
                "response": "answer",
                "target": 0.5,
            },
            {
                "task": "causal_lm",
                "transformed_prompt": "prompt ",
                "response": "answer",
                "target": -1.0,
            },
        ]
        loss, parts = task_loss(model, records, encoder, 0.25, 0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("calibration_loss", parts)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(model.bias).all())

        loader = DataLoader(records, batch_size=2, collate_fn=lambda batch: batch)
        metrics = evaluate_loss(
            model,
            loader,
            encoder,
            {"smooth_l1_beta": 0.25, "causal_loss_scale": 0.1},
            max_batches=1,
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval_loss"])))

    def test_mixed_batch_is_weighted_by_record_count(self):
        """Increasing micro-batch size must not change task mixing semantics.

        ``task_loss`` returns task components after applying the causal scale.
        For one calibration and two causal records, the mixed objective should
        therefore be ``(calibration + 2 * causal) / 3``.  This catches the
        former batch-level ``causal_mean + calibration_mean`` behavior, which
        over-weighted whichever task happened to have fewer records in a
        mixed batch.
        """
        tokenizer = TinyTokenizer()
        encoder = BatchEncoder(tokenizer, "Qwen2.5-test", 64, "cpu")
        model = TinyCausalModel()
        calibration = {
            "task": "calibration",
            "transformed_prompt": "prompt ",
            "response": "answer",
            "target": 0.5,
        }
        causal = {
            "task": "causal_lm",
            "transformed_prompt": "prompt ",
            "response": "answer",
            "target": -1.0,
        }

        calibration_loss, _ = task_loss(model, [calibration], encoder, 0.25, 0.1)
        causal_loss, _ = task_loss(model, [causal], encoder, 0.25, 0.1)
        mixed_loss, parts = task_loss(
            model, [calibration, causal, causal], encoder, 0.25, 0.1
        )
        expected = (calibration_loss + 2.0 * causal_loss) / 3.0
        self.assertTrue(torch.allclose(mixed_loss, expected, atol=1e-7, rtol=0.0))
        self.assertAlmostEqual(
            parts["causal_loss"], float(causal_loss.item() * 2.0 / 3.0), places=7
        )
        self.assertAlmostEqual(
            parts["calibration_loss"],
            float(calibration_loss.item() / 3.0),
            places=7,
        )

    def test_single_task_microbatch_keeps_two_forward_collective_order(self):
        """Each rank must execute both task forwards even for a single-task batch.

        DistributedSampler shards records independently, so a local batch can
        contain only one task.  The trainer uses a connected zero-weight dummy
        for the absent task; this regression test verifies that both forwards
        happen and that the resulting loss remains finite/backpropagatable.
        """
        tokenizer = TinyTokenizer()
        encoder = BatchEncoder(tokenizer, "Qwen2.5-test", 64, "cpu")
        model = TinyCausalModel()
        calibration = {
            "task": "calibration",
            "transformed_prompt": "prompt ",
            "response": "answer",
            "target": 0.5,
        }
        loss, _ = task_loss(model, [calibration], encoder, 0.25, 0.1)
        self.assertEqual(model.forward_calls, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.isfinite(model.bias.grad).all())

        model = TinyCausalModel()
        causal = dict(calibration, task="causal_lm", target=-1.0)
        loss, _ = task_loss(model, [causal], encoder, 0.25, 0.1)
        self.assertEqual(model.forward_calls, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.isfinite(model.bias.grad).all())


if __name__ == "__main__":
    unittest.main()
