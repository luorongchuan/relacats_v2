"""Train the full RelaCaTS objective from Eqs. (26)--(33).

The existing ``train_relacats.py`` remains the conservative residual-target
trainer.  This file adds the missing theory path without removing that control:

* q-head-by-instruction: SmoothL1 calibration loss;
* f-head-by-instruction: BCE fragility loss;
* optional same-question pairwise ranking loss on q(1-f);
* optional Eq. (32) generation filtering by high RelSSC + low fragility;
* the three confidence target modes from v2 are preserved.

The q/f "heads" are two instructions on the same causal LM, exactly as in the
theory document; no architectural classifier head is added.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from relacats_v2.common import confidence_suffix
from relacats_v2.core import (
    DEFAULT_LAMBDA_REL,
    DEFAULT_TARGET_MODE,
    TARGET_MODES,
    fragility_suffix,
    resolve_confidence_target,
)
from relacats_v2.model_training.train_relacats import (
    MixedTextDataset,
    _dataset_fraction,
    _sample_without_replacement,
    distributed_context,
    load_records,
    raw_collate,
    resolve_path,
    save_adapter,
    seed_everything,
)


GENERATION_FILTER_MODES = frozenset({"ssc", "relssc_fragility"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--save-path")
    parser.add_argument("--resume-from")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--target-mode", choices=sorted(TARGET_MODES))
    parser.add_argument("--lambda-rel", type=float)
    return parser.parse_args()


def _finite_probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def prepare_full_examples(
    config: dict[str, Any], split: str, requested_total: int, seed: int
) -> list[dict[str, Any]]:
    """Build causal + joint q/f calibration examples.

    ``generation_filter_mode=ssc`` preserves the conservative v2 control.
    ``generation_filter_mode=relssc_fragility`` implements Eq. (32).
    """

    dataset_root = resolve_path(config["dataset_root"])
    target_mode = str(config.get("target_mode", DEFAULT_TARGET_MODE)).strip().lower()
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    lambda_rel = float(config.get("lambda_rel", DEFAULT_LAMBDA_REL))
    causal_ratio = _finite_probability(config.get("causal_lm_ratio", 0.8), "causal_lm_ratio")
    threshold = _finite_probability(config.get("threshold", 0.75), "threshold")
    eta_c = _finite_probability(config.get("eta_c", 0.75), "eta_c")
    eta_f = _finite_probability(config.get("eta_f", 0.25), "eta_f")
    filter_mode = str(config.get("generation_filter_mode", "ssc")).strip().lower()
    if filter_mode not in GENERATION_FILTER_MODES:
        raise ValueError(f"generation_filter_mode must be one of {sorted(GENERATION_FILTER_MODES)}")

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
                ssc = _finite_probability(record.get("ssc"), "ssc")
                relssc = _finite_probability(
                    record.get("relssc", record.get("relational_consistency")),
                    "relssc",
                )
                relation_valid_ratio = _finite_probability(
                    record.get("relation_valid_ratio"), "relation_valid_ratio"
                )
                fragility = _finite_probability(
                    record.get("fragility_target", record.get("consensus_fragility")),
                    "fragility_target",
                )
                target = resolve_confidence_target(
                    ssc=ssc,
                    relssc=relssc,
                    relation_valid_ratio=relation_valid_ratio,
                    target_mode=target_mode,
                    lambda_rel=lambda_rel,
                )
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
                    "_relation_valid_ratio": relation_valid_ratio,
                }
            )
            valid.append(copied)

        if records and not valid:
            raise ValueError(
                f"Dataset {spec['name']!r} has no full RelaCaTS labels. "
                "Run data_creation/build_full_relacats_dataset.py first."
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
                if record["_relssc"] > eta_c and record["_fragility"] < eta_f
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


class FullBatchEncoder:
    def __init__(self, tokenizer: Any, model_name: str, max_length: int, device: Any):
        self.tokenizer = tokenizer
        self.conf_suffix = confidence_suffix(model_name)
        self.frag_suffix = fragility_suffix(model_name)
        self.max_length = max_length
        self.device = device
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        if not yes_ids:
            raise ValueError("Tokenizer cannot encode Yes")
        self.yes_token_id = int(yes_ids[0])

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    def causal(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [record["transformed_prompt"] for record in records]
        eos = self.tokenizer.eos_token or ""
        full = [prompt + record["response"] + eos for prompt, record in zip(prompts, records)]
        encoded = self._tokenize(full)
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for row, prompt in enumerate(prompts):
            prompt_length = len(
                self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
            )
            labels[row, :prompt_length] = -100
        encoded["labels"] = labels
        return encoded

    def query(
        self, records: list[dict[str, Any]], *, fragility: bool
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        suffix = self.frag_suffix if fragility else self.conf_suffix
        target_key = "fragility_target" if fragility else "target"
        texts = [
            record["transformed_prompt"] + record["response"] + " " + suffix
            for record in records
        ]
        encoded = self._tokenize(texts)
        targets = torch.tensor(
            [float(record[target_key]) for record in records],
            dtype=torch.float32,
            device=self.device,
        )
        return encoded, targets


def _yes_probability(output: Any, encoded: dict[str, torch.Tensor], token_id: int) -> torch.Tensor:
    lengths = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(lengths.shape[0], device=lengths.device)
    logits = output.logits[rows, lengths, :].float()
    return F.softmax(logits, dim=-1)[:, token_id]


def _pairwise_rank_loss(
    records: list[dict[str, Any]], q: torch.Tensor, f: torch.Tensor
) -> torch.Tensor:
    """Eq. (31), restricted to comparable records in the current microbatch."""

    effective = q * (1.0 - f)
    terms: list[torch.Tensor] = []
    for i in range(len(records)):
        for j in range(len(records)):
            if i == j:
                continue
            if records[i].get("question_id") != records[j].get("question_id"):
                continue
            if float(records[i].get("relssc", 0.0)) <= float(records[j].get("relssc", 0.0)):
                continue
            terms.append(F.softplus(-(effective[i] - effective[j])))
    if not terms:
        return effective.sum() * 0.0
    return torch.stack(terms).mean()


def full_task_loss(
    model: torch.nn.Module,
    batch: list[dict[str, Any]],
    encoder: FullBatchEncoder,
    *,
    smooth_l1_beta: float,
    causal_loss_scale: float,
    lambda_f: float,
    lambda_r: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Three fixed-order forwards: CLM -> confidence q -> fragility f."""

    causal = [item for item in batch if item["task"] == "causal_lm"]
    calibration = [item for item in batch if item["task"] == "calibration"]
    device = encoder.device

    def dummy() -> dict[str, Any]:
        return {
            "task": "dummy",
            "transformed_prompt": "x",
            "response": "x",
            "target": 0.0,
            "fragility_target": 0.0,
            "relssc": 0.0,
            "question_id": "dummy",
        }

    causal_inputs = causal if causal else [dummy()]
    calibration_inputs = calibration if calibration else [dummy()]

    encoded_causal = encoder.causal(causal_inputs)
    out_causal = model(**encoded_causal)
    if causal and torch.any(encoded_causal["labels"] != -100):
        causal_loss = out_causal.loss.float() * causal_loss_scale
    else:
        causal_loss = out_causal.logits.float().sum() * 0.0

    encoded_q, q_targets = encoder.query(calibration_inputs, fragility=False)
    out_q = model(**encoded_q)
    q_pred = _yes_probability(out_q, encoded_q, encoder.yes_token_id)
    calibration_loss = F.smooth_l1_loss(q_pred, q_targets, beta=smooth_l1_beta)
    if not calibration:
        calibration_loss = calibration_loss * 0.0

    encoded_f, f_targets = encoder.query(calibration_inputs, fragility=True)
    out_f = model(**encoded_f)
    f_pred = _yes_probability(out_f, encoded_f, encoder.yes_token_id)
    fragility_loss = F.binary_cross_entropy(
        f_pred.clamp(1e-6, 1.0 - 1e-6), f_targets
    )
    if not calibration:
        fragility_loss = fragility_loss * 0.0
    ranking_loss = _pairwise_rank_loss(calibration_inputs, q_pred, f_pred)
    if not calibration:
        ranking_loss = ranking_loss * 0.0

    total_records = len(causal) + len(calibration)
    causal_fraction = len(causal) / total_records if total_records else 0.0
    calibration_fraction = len(calibration) / total_records if total_records else 0.0
    total = (
        causal_fraction * causal_loss
        + calibration_fraction * calibration_loss
        + calibration_fraction * float(lambda_f) * fragility_loss
        + float(lambda_r) * ranking_loss
    )
    return total, {
        "causal_loss": float((causal_fraction * causal_loss).detach().item()),
        "calibration_loss": float((calibration_fraction * calibration_loss).detach().item()),
        "fragility_loss": float((calibration_fraction * fragility_loss).detach().item()),
        "ranking_loss": float(ranking_loss.detach().item()),
    }


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    encoder: FullBatchEncoder,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    count = 0
    for batch_index, batch in enumerate(loader):
        if config.get("max_eval_batches") is not None and batch_index >= int(config["max_eval_batches"]):
            break
        loss, parts = full_task_loss(
            model,
            batch,
            encoder,
            smooth_l1_beta=float(config.get("smooth_l1_beta", 0.25)),
            causal_loss_scale=float(config.get("causal_loss_scale", 0.1)),
            lambda_f=float(config.get("lambda_f", 1.0)),
            lambda_r=float(config.get("lambda_r", 0.0)),
        )
        n = len(batch)
        count += n
        totals["eval_loss"] += n * float(loss.item())
        for key, value in parts.items():
            totals[f"eval_{key}"] += n * value
    vector = torch.tensor(
        [count] + [totals[key] for key in sorted(totals)],
        dtype=torch.float64,
        device=encoder.device,
    )
    if dist.is_initialized():
        dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    keys = sorted(totals)
    denominator = max(1.0, float(vector[0].item()))
    result = {key: float(vector[index + 1].item() / denominator) for index, key in enumerate(keys)}
    model.train()
    return result


def main() -> None:
    args = parse_args()
    with open(args.config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.target_mode is not None:
        config["target_mode"] = args.target_mode
    if args.lambda_rel is not None:
        config["lambda_rel"] = args.lambda_rel

    target_mode = str(config.get("target_mode", DEFAULT_TARGET_MODE)).strip().lower()
    if target_mode not in TARGET_MODES:
        raise ValueError(f"Unsupported target_mode={target_mode!r}")
    config["target_mode"] = target_mode
    config["lambda_rel"] = float(config.get("lambda_rel", DEFAULT_LAMBDA_REL))
    config["full_relacats"] = True

    local_rank, rank, world_size = distributed_context(bool(config.get("distributed", True)))
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    batch_size = args.batch_size or int(config.get("batch_size", 1))
    grad_accum = args.gradient_accumulation_steps or int(config["gradient_accumulation_steps"])
    train_total = args.max_train_samples or int(config["total_train_samples"])
    eval_total = args.max_eval_samples or int(config["total_eval_samples"])
    train_records = prepare_full_examples(config, "train", train_total, seed)
    eval_records = prepare_full_examples(config, "test", eval_total, seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], use_fast=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if str(config.get("dtype", "bfloat16")) == "bfloat16" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, local_files_only=True
    )
    base_model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    if args.resume_from:
        model = PeftModel.from_pretrained(base_model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=int(config["lora_r"]),
                lora_alpha=int(config["lora_alpha"]),
                target_modules=list(config.get("lora_target_modules", ["q_proj", "v_proj"])),
                lora_dropout=float(config["lora_dropout"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    device = torch.device("cuda", local_rank)
    model.to(device)
    if bool(config.get("distributed", True)):
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_dataset = MixedTextDataset(train_records)
    eval_dataset = MixedTextDataset(eval_records)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        if world_size > 1 else None
    )
    eval_sampler = (
        DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        collate_fn=raw_collate,
        drop_last=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=raw_collate,
        drop_last=False,
        num_workers=0,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]))
    updates_per_epoch = max(1, len(train_loader) // grad_accum)
    max_updates = int(config["num_epochs"]) * updates_per_epoch
    if args.max_optimizer_steps is not None:
        max_updates = min(max_updates, args.max_optimizer_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_updates))
    encoder = FullBatchEncoder(
        tokenizer, config["model_name"], int(config.get("max_length", 1024)), device
    )
    output_dir = resolve_path(args.save_path or config["output_dir"])

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    microstep = 0
    stop = False
    model.train()
    for epoch in range(int(config["num_epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            microstep += 1
            should_update = microstep % grad_accum == 0
            sync = contextlib.nullcontext() if should_update or not hasattr(model, "no_sync") else model.no_sync()
            with sync:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss, parts = full_task_loss(
                        model,
                        batch,
                        encoder,
                        smooth_l1_beta=float(config.get("smooth_l1_beta", 0.25)),
                        causal_loss_scale=float(config.get("causal_loss_scale", 0.1)),
                        lambda_f=float(config.get("lambda_f", 1.0)),
                        lambda_r=float(config.get("lambda_r", 0.0)),
                    )
                    scaled = loss / grad_accum
                scaled.backward()
            if should_update:
                if config.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if rank == 0:
                    print(
                        f"step={optimizer_step}/{max_updates} loss={loss.item():.6f} "
                        f"q={parts['calibration_loss']:.6f} f={parts['fragility_loss']:.6f} "
                        f"rank={parts['ranking_loss']:.6f} clm={parts['causal_loss']:.6f}",
                        flush=True,
                    )
                if optimizer_step >= max_updates:
                    stop = True
                    break
        if stop:
            break

    metrics = evaluate_loss(model, eval_loader, encoder, config)
    if rank == 0:
        print(f"Evaluation: {metrics}", flush=True)
        save_adapter(
            model,
            tokenizer,
            output_dir,
            config,
            {
                "optimizer_step": optimizer_step,
                "microstep": microstep,
                "eval_metrics": metrics,
                "world_size": world_size,
                "effective_update_batch": batch_size * world_size * grad_accum,
            },
        )
        print(f"Saved full RelaCaTS adapter to {output_dir}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
