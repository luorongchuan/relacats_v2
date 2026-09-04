"""Train Full RelaCaTS with missing-fragility masking.

This is the recommended trainer for strict migrated v1 pools.  It preserves the
existing ``train_full_relacats`` implementation and monkey-patches only the
three pieces that need stricter semantics:

* q-calibration examples remain usable when ``fragility_target`` is unavailable;
* identity-only numeric samples are masked out of L_frag/L_rank instead of
  being falsely supervised with f=0;
* I_frag uses the correct model-family chat boundary for Llama/Qwen/DeepSeek.

The rest of optimizer/DDP/LoRA/checkpoint behavior is inherited unchanged.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from relacats_v2.common import confidence_suffix, model_family
from relacats_v2.core import (
    DEFAULT_LAMBDA_REL,
    DEFAULT_TARGET_MODE,
    TARGET_MODES,
    resolve_confidence_target,
)
from relacats_v2.model_training import train_full_relacats as base
from relacats_v2.model_training.train_relacats import (
    _dataset_fraction,
    _sample_without_replacement,
    load_records,
    resolve_path,
)


def fragility_suffix(model_name: str) -> str:
    question = (
        "Is the confidence in the preceding answer fragile under valid "
        "relation-preserving transformations? (Yes/No)"
    )
    family = model_family(model_name)
    if family == "llama":
        return (
            "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            f"{question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if family == "qwen":
        return (
            "<|im_end|>\n<|im_start|>user\n"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "<｜end▁of▁sentence｜><｜User｜>Directly answer the question by Yes or No:"
        "Is the confidence in the preceding answer fragile under valid "
        "relation-preserving transformations?<｜Assistant｜>"
    )


class MaskedFullBatchEncoder(base.FullBatchEncoder):
    def __init__(self, tokenizer: Any, model_name: str, max_length: int, device: Any):
        super().__init__(tokenizer, model_name, max_length, device)
        self.conf_suffix = confidence_suffix(model_name)
        self.frag_suffix = fragility_suffix(model_name)


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _binary_cross_entropy_probability(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE on probabilities in explicit float32, safe inside CUDA autocast.

    ``torch.nn.functional.binary_cross_entropy`` on a sigmoid/softmax
    probability tensor is intentionally rejected by CUDA autocast.  RelaCaTS
    defines f_hat as a probability rather than as an independent binary logit,
    so we keep that probability-space objective and evaluate the equivalent
    Bernoulli negative log likelihood explicitly in float32.
    """

    probability = prediction.float().clamp(eps, 1.0 - eps)
    truth = target.float()
    return -(
        truth * torch.log(probability)
        + (1.0 - truth) * torch.log1p(-probability)
    ).mean()


def prepare_full_examples(
    config: dict[str, Any], split: str, requested_total: int, seed: int
) -> list[dict[str, Any]]:
    dataset_root = resolve_path(config["dataset_root"])
    target_mode = str(config.get("target_mode", DEFAULT_TARGET_MODE)).strip().lower()
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    lambda_rel = float(config.get("lambda_rel", DEFAULT_LAMBDA_REL))
    causal_ratio = _probability(config.get("causal_lm_ratio", 0.7), "causal_lm_ratio")
    threshold = _probability(config.get("threshold", 0.75), "threshold")
    eta_c = _probability(config.get("eta_c", 0.75), "eta_c")
    eta_f = _probability(config.get("eta_f", 0.25), "eta_f")
    filter_mode = str(config.get("generation_filter_mode", "ssc")).strip().lower()
    if filter_mode not in base.GENERATION_FILTER_MODES:
        raise ValueError(
            f"generation_filter_mode must be one of {sorted(base.GENERATION_FILTER_MODES)}"
        )

    specs = list(config["datasets"])
    raw_weights = [float(spec.get("weight", 1.0)) for spec in specs]
    if any(not math.isfinite(value) or value < 0 for value in raw_weights):
        raise ValueError("dataset weights must be finite and non-negative")
    total_weight = sum(raw_weights)
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    mixed: list[dict[str, Any]] = []

    for spec, raw_weight in zip(specs, raw_weights):
        normalized = raw_weight / total_weight if total_weight > 0 else 0.0
        fraction = _dataset_fraction(spec, split, normalized)
        records = load_records(dataset_root, spec["name"], split)
        valid: list[dict[str, Any]] = []
        for record in records:
            try:
                ssc = _probability(record.get("ssc"), "ssc")
                relssc = _probability(
                    record.get("relssc", record.get("relational_consistency")),
                    "relssc",
                )
                relation_valid_ratio = _probability(
                    record.get("relation_valid_ratio"), "relation_valid_ratio"
                )
                target = resolve_confidence_target(
                    ssc=ssc,
                    relssc=relssc,
                    relation_valid_ratio=relation_valid_ratio,
                    target_mode=target_mode,
                    lambda_rel=lambda_rel,
                )
                fragility_available = bool(
                    record.get(
                        "fragility_available",
                        record.get("fragility_target", record.get("consensus_fragility"))
                        is not None,
                    )
                )
                if fragility_available:
                    fragility = _probability(
                        record.get("fragility_target", record.get("consensus_fragility")),
                        "fragility_target",
                    )
                else:
                    fragility = None
            except (TypeError, ValueError):
                continue
            if not record.get("transformed_prompt") or not record.get("response"):
                continue
            copied = dict(record)
            copied.update(
                {
                    "_target": target,
                    "_ssc": ssc,
                    "_relssc": relssc,
                    "_fragility": fragility,
                    "_fragility_available": fragility_available,
                    "_relation_valid_ratio": relation_valid_ratio,
                }
            )
            valid.append(copied)

        if records and not valid:
            raise ValueError(
                f"Dataset {spec['name']!r} has no strict Full RelaCaTS labels. "
                "Run data_creation/build_full_relacats_strict.py first."
            )

        dataset_total = int(round(requested_total * fraction))
        calibration_count = int(round(dataset_total * (1.0 - causal_ratio)))
        causal_count = max(0, dataset_total - calibration_count)

        bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in valid:
            bins[min(int(record["_target"] / 0.05), 20)].append(record)
        per_bin = max(1, calibration_count // 21) if calibration_count else 0
        calibration: list[dict[str, Any]] = []
        for bin_index in range(21):
            calibration.extend(
                _sample_without_replacement(bins[bin_index], per_bin, rng)
            )
        if len(calibration) < calibration_count:
            selected = {id(item) for item in calibration}
            remaining = [item for item in valid if id(item) not in selected]
            calibration.extend(
                _sample_without_replacement(
                    remaining, calibration_count - len(calibration), rng
                )
            )

        for record in calibration[:calibration_count]:
            mixed.append(
                {
                    "task": "calibration",
                    "dataset_name": spec["name"],
                    "question_id": record.get("question_id"),
                    "transformed_prompt": record["transformed_prompt"],
                    "response": record["response"],
                    "target": record["_target"],
                    "relssc": record["_relssc"],
                    "fragility_target": record["_fragility"],
                    "fragility_available": record["_fragility_available"],
                    "ssc": record["_ssc"],
                    "relation_valid_ratio": record["_relation_valid_ratio"],
                }
            )

        if filter_mode == "ssc":
            candidates = [record for record in valid if record["_ssc"] > threshold]
        else:
            candidates = [
                record
                for record in valid
                if record["_fragility_available"]
                and record["_relssc"] > eta_c
                and float(record["_fragility"]) < eta_f
            ]
        for record in _sample_without_replacement(candidates, causal_count, rng):
            mixed.append(
                {
                    "task": "causal_lm",
                    "dataset_name": spec["name"],
                    "question_id": record.get("question_id"),
                    "transformed_prompt": record["transformed_prompt"],
                    "response": record["response"],
                    "target": -1.0,
                    "selection_ssc": record["_ssc"],
                    "selection_relssc": record["_relssc"],
                    "selection_fragility": record["_fragility"],
                    "generation_filter_mode": filter_mode,
                }
            )

    rng.shuffle(mixed)
    if not mixed:
        raise ValueError(f"No usable {split} examples were constructed")
    return mixed


def full_task_loss(
    model: torch.nn.Module,
    batch: list[dict[str, Any]],
    encoder: MaskedFullBatchEncoder,
    *,
    smooth_l1_beta: float,
    causal_loss_scale: float,
    lambda_f: float,
    lambda_r: float,
):
    causal = [item for item in batch if item["task"] == "causal_lm"]
    calibration = [item for item in batch if item["task"] == "calibration"]
    fragility_records = [
        item
        for item in calibration
        if bool(item.get("fragility_available"))
        and item.get("fragility_target") is not None
    ]

    def dummy() -> dict[str, Any]:
        return {
            "task": "dummy",
            "transformed_prompt": "x",
            "response": "x",
            "target": 0.0,
            "fragility_target": 0.0,
            "fragility_available": True,
            "relssc": 0.0,
            "question_id": "dummy",
        }

    causal_inputs = causal if causal else [dummy()]
    calibration_inputs = calibration if calibration else [dummy()]
    fragility_inputs = fragility_records if fragility_records else [dummy()]

    encoded_causal = encoder.causal(causal_inputs)
    out_causal = model(**encoded_causal)
    if causal and torch.any(encoded_causal["labels"] != -100):
        causal_loss = out_causal.loss.float() * causal_loss_scale
    else:
        causal_loss = out_causal.logits.float().sum() * 0.0

    encoded_q, q_targets = encoder.query(calibration_inputs, fragility=False)
    out_q = model(**encoded_q)
    q_pred = base._yes_probability(out_q, encoded_q, encoder.yes_token_id)
    calibration_loss = F.smooth_l1_loss(q_pred, q_targets, beta=smooth_l1_beta)
    if not calibration:
        calibration_loss = calibration_loss * 0.0

    encoded_f, f_targets = encoder.query(fragility_inputs, fragility=True)
    out_f = model(**encoded_f)
    f_pred = base._yes_probability(out_f, encoded_f, encoder.yes_token_id)
    fragility_loss = _binary_cross_entropy_probability(f_pred, f_targets)
    if not fragility_records:
        fragility_loss = fragility_loss * 0.0

    if fragility_records and calibration:
        q_index = {id(record): index for index, record in enumerate(calibration_inputs)}
        q_for_frag = torch.stack([q_pred[q_index[id(record)]] for record in fragility_records])
        ranking_loss = base._pairwise_rank_loss(
            fragility_records, q_for_frag, f_pred
        )
    else:
        ranking_loss = q_pred.sum() * 0.0

    total_records = len(causal) + len(calibration)
    denominator = total_records if total_records else 1
    causal_fraction = len(causal) / denominator
    calibration_fraction = len(calibration) / denominator
    fragility_fraction = len(fragility_records) / denominator
    total = (
        causal_fraction * causal_loss
        + calibration_fraction * calibration_loss
        + fragility_fraction * float(lambda_f) * fragility_loss
        + float(lambda_r) * ranking_loss
    )
    return total, {
        "causal_loss": float((causal_fraction * causal_loss).detach().item()),
        "calibration_loss": float((calibration_fraction * calibration_loss).detach().item()),
        "fragility_loss": float((fragility_fraction * fragility_loss).detach().item()),
        "ranking_loss": float(ranking_loss.detach().item()),
        "fragility_supervision_fraction": float(fragility_fraction),
    }


def main() -> None:
    # base.main resolves these names from its own module globals at runtime.
    base.prepare_full_examples = prepare_full_examples
    base.full_task_loss = full_task_loss
    base.FullBatchEncoder = MaskedFullBatchEncoder
    base.main()


if __name__ == "__main__":
    main()
