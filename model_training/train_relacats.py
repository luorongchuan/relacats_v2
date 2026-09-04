"""Two-task trainer with the three RelaCaTS v2 confidence-target modes.

This is intentionally independent of ``model_training/train.py``.  It keeps
the original CaTS objective and published repository hyperparameters while
fixing generic implementation errors (padding labels, attention masks,
deterministic DDP sampling, and scheduler update counts).
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
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from relacats_v2.common import confidence_suffix, read_jsonl
from relacats_v2.core import (
    DEFAULT_LAMBDA_REL,
    DEFAULT_TARGET_MODE,
    TARGET_MODES,
    resolve_confidence_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--save-path")
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override config batch_size (per rank micro-batch size).",
    )
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable activation/gradient checkpointing.",
    )
    checkpointing.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation/gradient checkpointing (uses more VRAM, usually faster).",
    )
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument(
        "--target-mode",
        choices=sorted(TARGET_MODES),
        help="Override config target_mode for the confidence loss.",
    )
    parser.add_argument(
        "--lambda-rel",
        type=float,
        help="Override config lambda_rel (used only by residual mode).",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def distributed_context(enabled: bool) -> tuple[int, int, int]:
    if enabled:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("distributed=true requires torchrun (LOCAL_RANK is absent)")
        local_rank = int(os.environ["LOCAL_RANK"])
        # Select the rank-local CUDA device before NCCL is initialised.  This
        # avoids older NCCL/PyTorch combinations briefly initialising every
        # rank on device 0 when the process-group backend is brought up.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, dist.get_rank(), dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("RelaCaTS training requires CUDA")
    return 0, 0, 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(dataset_root: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    path = dataset_root / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RelSSC dataset split: {path}. Run 02_build_relssc_dataset.sh first."
        )
    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"RelSSC dataset split is empty: {path}")
    return records


def _sample_without_replacement(
    values: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count <= 0 or not values:
        return []
    return rng.sample(values, min(count, len(values)))


def _dataset_fraction(
    spec: dict[str, Any], split: str, normalized_weight: float
) -> float:
    """Return the CaTS allocation fraction for one split.

    Original CaTS configuration files express dataset allocation as
    ``train_percentage``/``eval_percentage``.  RelaCaTS v2 configs use a
    simpler normalized ``weight`` by default because the option-MCQ subset is
    intentionally different from the original numeric mix.  Supporting both
    forms here keeps the trainer compatible with either configuration without
    silently ignoring the paper's percentage fields.
    """

    percentage_key = (
        "train_percentage"
        if split == "train"
        else "eval_percentage"
        if split in {"test", "eval", "validation"}
        else f"{split}_percentage"
    )
    if percentage_key not in spec:
        return normalized_weight
    try:
        fraction = float(spec[percentage_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{percentage_key} for dataset {spec.get('name')!r} must be numeric"
        ) from exc
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"{percentage_key} for dataset {spec.get('name')!r} must be in [0,1]"
        )
    return fraction


def prepare_mixed_examples(
    config: dict[str, Any], split: str, requested_total: int, seed: int
) -> list[dict[str, Any]]:
    """Build calibration and causal examples with independent target semantics.

    The configured v2 target is used only for calibration labels and target
    bins.  Causal/generation examples are always admitted by their original
    count-based SSC, regardless of ``target_mode``.
    """

    dataset_root = resolve_path(config["dataset_root"])
    target_mode = str(config.get("target_mode", DEFAULT_TARGET_MODE)).strip().lower()
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}; got {target_mode!r}"
        )
    lambda_rel = float(config.get("lambda_rel", DEFAULT_LAMBDA_REL))
    causal_ratio = float(config["causal_lm_ratio"])
    threshold = float(config["threshold"])
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    mixed: list[dict[str, Any]] = []
    dataset_specs = config["datasets"]
    raw_weights: list[float] = []
    for spec in dataset_specs:
        try:
            weight = float(spec.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for dataset {spec.get('name')!r}") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Dataset weights must be finite and non-negative: {spec!r}")
        raw_weights.append(weight)
    total_weight = sum(raw_weights)
    percentage_key = (
        "train_percentage"
        if split == "train"
        else "eval_percentage"
        if split in {"test", "eval", "validation"}
        else f"{split}_percentage"
    )
    if total_weight <= 0 and not any(
        percentage_key in spec for spec in dataset_specs
    ):
        raise ValueError("Dataset weights must sum to a positive value")

    for spec, raw_weight in zip(dataset_specs, raw_weights):
        name = spec["name"]
        normalized_weight = raw_weight / total_weight if total_weight > 0 else 0.0
        weight = _dataset_fraction(spec, split, normalized_weight)
        records = load_records(dataset_root, name, split)
        valid: list[dict[str, Any]] = []
        for record in records:
            ssc = record.get("ssc")
            relssc = record.get(
                "relssc",
                record.get("relational_consistency", record.get("weighted_consistency")),
            )
            relation_valid_ratio = record.get("relation_valid_ratio")
            if ssc is None or relssc is None or relation_valid_ratio is None:
                continue
            try:
                ssc_float = float(ssc)
                relssc_float = float(relssc)
                relation_valid_ratio_float = float(relation_valid_ratio)
                target_float = resolve_confidence_target(
                    ssc=ssc_float,
                    relssc=relssc_float,
                    relation_valid_ratio=relation_valid_ratio_float,
                    target_mode=target_mode,
                    lambda_rel=lambda_rel,
                )
            except (TypeError, ValueError):
                continue
            if not record.get("transformed_prompt") or not record.get("response"):
                continue
            copied = dict(record)
            copied["_target"] = target_float
            copied["_ssc"] = ssc_float
            copied["_relssc"] = relssc_float
            copied["_relation_valid_ratio"] = relation_valid_ratio_float
            valid.append(copied)

        if records and not valid:
            raise ValueError(
                f"Dataset {name!r} has no usable RelaCaTS v2 target inputs. "
                "Rebuild it with relacats_v2/data_creation/build_relssc_dataset.py "
                "so every row contains ssc, relssc, and relation_valid_ratio."
            )

        dataset_total = int(round(requested_total * weight))
        calibration_count = int(round(dataset_total * (1.0 - causal_ratio)))
        causal_count = max(0, dataset_total - calibration_count)

        # Original CaTS uses 0.05-wide target buckets.  Split the requested
        # calibration allocation evenly across the 21 inclusive bins.
        bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in valid:
            bins[min(int(record["_target"] / 0.05), 20)].append(record)
        per_bin = max(1, calibration_count // 21) if calibration_count else 0
        calibration: list[dict[str, Any]] = []
        for bin_index in range(21):
            calibration.extend(_sample_without_replacement(bins[bin_index], per_bin, rng))
        if len(calibration) < calibration_count:
            selected_ids = {id(item) for item in calibration}
            remaining = [item for item in valid if id(item) not in selected_ids]
            calibration.extend(
                _sample_without_replacement(
                    remaining, calibration_count - len(calibration), rng
                )
            )
        for record in calibration[:calibration_count]:
            mixed.append(
                {
                    "task": "calibration",
                    "dataset_name": name,
                    "transformed_prompt": record["transformed_prompt"],
                    "response": record["response"],
                    "target": record["_target"],
                    "target_mode": target_mode,
                    "ssc": record["_ssc"],
                    "relssc": record["_relssc"],
                    "relation_valid_ratio": record["_relation_valid_ratio"],
                }
            )

        # This must never use ``_target`` or ``_relssc``.  In residual and
        # RelSSC-replacement modes those values may intentionally penalize a
        # high-SSC response, while the generation objective retains CaTS's
        # original high-SSC filter.
        high_ssc = [record for record in valid if record["_ssc"] > threshold]
        for record in _sample_without_replacement(high_ssc, causal_count, rng):
            mixed.append(
                {
                    "task": "causal_lm",
                    "dataset_name": name,
                    "transformed_prompt": record["transformed_prompt"],
                    "response": record["response"],
                    "target": -1.0,
                    "selection_ssc": record["_ssc"],
                    "target_mode": target_mode,
                }
            )

    rng.shuffle(mixed)
    if not mixed:
        raise ValueError(f"No usable {split} examples were constructed")
    return mixed


class MixedTextDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def raw_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class BatchEncoder:
    def __init__(
        self,
        tokenizer: Any,
        model_name: str,
        max_length: int,
        device: torch.device | str | int,
    ):
        self.tokenizer = tokenizer
        self.suffix = confidence_suffix(model_name)
        self.max_length = max_length
        self.device = device
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        if not yes_ids:
            raise ValueError("Tokenizer cannot encode the confidence token 'Yes'")
        self.yes_token_id = int(yes_ids[0])

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def causal(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [record["transformed_prompt"] for record in records]
        eos = self.tokenizer.eos_token or ""
        full_texts = [
            prompt + record["response"] + eos for prompt, record in zip(prompts, records)
        ]
        encoded = self._tokenize(full_texts)
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

    def calibration(
        self, records: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        texts = [
            record["transformed_prompt"] + record["response"] + " " + self.suffix
            for record in records
        ]
        encoded = self._tokenize(texts)
        targets = torch.tensor(
            [float(record["target"]) for record in records],
            dtype=torch.float32,
            device=self.device,
        )
        return encoded, targets


def task_loss(
    model: torch.nn.Module,
    batch: list[dict[str, Any]],
    encoder: BatchEncoder,
    smooth_l1_beta: float,
    causal_loss_scale: float,
    normalize_by_records: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the mixed causal/calibration objective for one micro-batch.

    The historical implementation added the *mean* causal and calibration
    losses once each.  That is equivalent when ``batch_size == 1`` (the
    original configuration), but silently changes the objective for a mixed
    batch: a batch containing one calibration record and seven causal records
    gave the calibration record the same weight as all seven causal records.
    ``normalize_by_records`` fixes that by weighting each task component by
    its number of records before dividing by the total records (the causal
    component remains the model's usual token-mean over its causal records).
    The default therefore preserves the per-record task mixture while
    remaining exactly identical for one-record batches.  Passing ``False`` is
    retained as a compatibility escape hatch for reproducing the old
    batch-level behavior.
    """
    causal_records = [item for item in batch if item["task"] == "causal_lm"]
    calibration_records = [item for item in batch if item["task"] == "calibration"]
    device = encoder.device
    causal_loss = torch.zeros((), device=device, dtype=torch.float32)
    calibration_loss = torch.zeros((), device=device, dtype=torch.float32)

    # DDP synchronization invariant
    # -----------------------------
    # A micro-batch is sharded independently on each rank.  Consequently one
    # rank may receive only causal records while another receives only
    # calibration records.  The old conditional forwards then made the ranks
    # execute a different number of DDP forwards.  With the default
    # ``broadcast_buffers=True`` this mismatched the per-forward buffer
    # broadcasts (and could eventually pair a broadcast with an all-reduce),
    # producing the NCCL timeout seen in multi-GPU training.  Always execute
    # the two task forwards in the same causal -> calibration order.  If a
    # task is absent locally, use a harmless one-record dummy and multiply its
    # connected loss by zero; this keeps the autograd/DDP graph identical on
    # every rank without changing the objective.
    def dummy_record() -> dict[str, Any]:
        # Keep the placeholder deliberately tiny.  Copying a real long
        # prompt/response here would retain an unnecessary second activation
        # graph on a near-full GPU batch and could turn the synchronization
        # workaround into an avoidable OOM.  The loss is multiplied by zero,
        # so the text and target have no effect on the objective.
        return {
            "task": "dummy",
            "transformed_prompt": "x",
            "response": "x",
            "target": 0.0,
        }

    real_causal = bool(causal_records)
    real_calibration = bool(calibration_records)
    causal_inputs = causal_records if real_causal else [dummy_record()]
    calibration_inputs = (
        calibration_records if real_calibration else [dummy_record()]
    )

    # Exactly one causal forward on every rank.  Do not skip the forward when
    # truncation masks all response tokens: a connected zero keeps collective
    # ordering intact while avoiding NaN from a mean over an empty target.
    encoded_causal = encoder.causal(causal_inputs)
    causal_output = model(**encoded_causal)
    if real_causal and torch.any(encoded_causal["labels"] != -100):
        if causal_output.loss is None:
            raise RuntimeError("Causal model returned no loss for labeled input")
        causal_loss = causal_output.loss.float() * causal_loss_scale
    else:
        causal_loss = causal_output.logits.float().sum() * 0.0

    # Exactly one calibration forward on every rank, in a fixed order after
    # the causal forward.  The dummy branch is connected to logits so DDP sees
    # the same autograd graph even though its contribution is zero.
    encoded_calibration, targets = encoder.calibration(calibration_inputs)
    calibration_output = model(**encoded_calibration)
    lengths = encoded_calibration["attention_mask"].sum(dim=1) - 1
    row_ids = torch.arange(lengths.shape[0], device=device)
    next_logits = calibration_output.logits[row_ids, lengths, :].float()
    yes_prob = F.softmax(next_logits, dim=-1)[:, encoder.yes_token_id]
    calibration_value = F.smooth_l1_loss(
        yes_prob, targets, beta=smooth_l1_beta
    )
    calibration_loss = calibration_value if real_calibration else calibration_value * 0.0

    if normalize_by_records:
        record_count = len(causal_records) + len(calibration_records)
        if record_count:
            causal_fraction = len(causal_records) / record_count
            calibration_fraction = len(calibration_records) / record_count
            total = causal_loss * causal_fraction + calibration_loss * calibration_fraction
            logged_causal_loss = causal_loss * causal_fraction
            logged_calibration_loss = calibration_loss * calibration_fraction
        else:
            total = causal_loss + calibration_loss
            logged_causal_loss = causal_loss
            logged_calibration_loss = calibration_loss
    else:
        total = causal_loss + calibration_loss
        logged_causal_loss = causal_loss
        logged_calibration_loss = calibration_loss
    return total, {
        "causal_loss": float(logged_causal_loss.detach().item()),
        "calibration_loss": float(logged_calibration_loss.detach().item()),
    }


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    encoder: BatchEncoder,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=encoder.device)
    processed_records = 0
    normalize_by_records = bool(
        config.get("normalize_task_loss_by_records", True)
    )
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        loss, parts = task_loss(
            model,
            batch,
            encoder,
            float(config.get("smooth_l1_beta", 0.25)),
            float(config.get("causal_loss_scale", 0.1)),
            normalize_by_records,
        )
        batch_records = len(batch)
        processed_records += batch_records
        sums += batch_records * torch.tensor(
            [float(loss.item()), parts["causal_loss"], parts["calibration_loss"]],
            dtype=torch.float64,
            device=encoder.device,
        )
    count_tensor = torch.tensor(
        [processed_records], dtype=torch.float64, device=encoder.device
    )
    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    denominator = max(1.0, float(count_tensor.item()))
    model.train()
    return {
        "eval_loss": float(sums[0].item() / denominator),
        "eval_causal_loss": float(sums[1].item() / denominator),
        "eval_calibration_loss": float(sums[2].item() / denominator),
    }


def save_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Path,
    config: dict[str, Any],
    state: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "relacats_train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    torch.save(state, output_dir / "trainer_state.pt")


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
        raise ValueError(
            f"target_mode must be one of {sorted(TARGET_MODES)}; got {target_mode!r}"
        )
    lambda_rel = config.get("lambda_rel", DEFAULT_LAMBDA_REL)
    # Resolve once to validate lambda_rel before allocating GPUs or loading a
    # model.  The actual per-example values are resolved during data prep.
    resolve_confidence_target(
        ssc=0.5,
        relssc=0.5,
        relation_valid_ratio=1.0,
        target_mode=target_mode,
        lambda_rel=lambda_rel,
    )
    config["target_mode"] = target_mode
    config["lambda_rel"] = float(lambda_rel)
    config["generation_selection_target"] = "ssc"

    local_rank, rank, world_size = distributed_context(bool(config.get("distributed", True)))
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    batch_size = (
        args.batch_size if args.batch_size is not None else int(config["batch_size"])
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    gradient_checkpointing = (
        args.gradient_checkpointing
        if args.gradient_checkpointing is not None
        else bool(config.get("gradient_checkpointing", True))
    )
    # Record effective command-line overrides in the saved metadata and make
    # all subsequent code use one resolved configuration.
    config["batch_size"] = batch_size
    config["gradient_checkpointing"] = gradient_checkpointing

    gradient_accumulation = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else int(config["gradient_accumulation_steps"])
    )
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    train_total = args.max_train_samples or int(config["total_train_samples"])
    eval_total = args.max_eval_samples or int(config["total_eval_samples"])
    if train_total <= 0 or eval_total <= 0:
        raise ValueError("train/eval sample budgets must be positive")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive when provided")
    train_records = prepare_mixed_examples(config, "train", train_total, seed)
    eval_records = prepare_mixed_examples(config, "test", eval_total, seed)
    if rank == 0:
        print(
            f"Prepared train={len(train_records)} eval={len(eval_records)}; "
            f"world_size={world_size}; effective_update_batch="
            f"{int(config['batch_size']) * world_size * gradient_accumulation}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], use_fast=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_name = str(config.get("dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        local_files_only=True,
    )
    base_model.config.use_cache = False
    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    if args.resume_from:
        model = PeftModel.from_pretrained(
            base_model, args.resume_from, is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            target_modules=list(config.get("lora_target_modules", ["q_proj", "v_proj"])),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    # Use an explicit device object here.  Passing a bare integer happens to
    # work on some CUDA/PyTorch versions, but can be interpreted differently by
    # ``Module.to`` and makes the same code brittle across environments.
    device = torch.device("cuda", local_rank)
    model.to(device)
    if bool(config.get("distributed", True)):
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # The model buffers (for example rotary-embedding frequencies)
            # are initialized identically on every rank and never modified by
            # training.  Disabling per-forward buffer broadcasts prevents a
            # stray broadcast from becoming rank-misaligned when a task branch
            # is represented by a zero-weight dummy.  task_loss still enforces
            # a fixed two-forward order and therefore keeps gradient
            # collectives synchronized.
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_dataset = MixedTextDataset(train_records)
    eval_dataset = MixedTextDataset(eval_records)
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        if world_size > 1
        else None
    )
    eval_sampler = (
        DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
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
    if len(train_loader) < gradient_accumulation:
        raise ValueError(
            f"Only {len(train_loader)} microbatches but gradient accumulation is "
            f"{gradient_accumulation}; lower it for smoke training"
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["learning_rate"]))
    updates_per_epoch = len(train_loader) // gradient_accumulation
    max_updates = int(config["num_epochs"]) * updates_per_epoch
    if args.max_optimizer_steps is not None:
        max_updates = min(max_updates, args.max_optimizer_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_updates))
    encoder = BatchEncoder(
        tokenizer, config["model_name"], int(config.get("max_length", 1024)), device
    )
    normalize_by_records = bool(
        config.get("normalize_task_loss_by_records", True)
    )
    output_dir = resolve_path(args.save_path or config["output_dir"])

    optimizer_step = 0
    microstep = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(int(config["num_epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            microstep += 1
            should_update = microstep % gradient_accumulation == 0
            sync_context = (
                contextlib.nullcontext()
                if should_update or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    loss, parts = task_loss(
                        model,
                        batch,
                        encoder,
                        float(config.get("smooth_l1_beta", 0.25)),
                        float(config.get("causal_loss_scale", 0.1)),
                        normalize_by_records,
                    )
                    scaled_loss = loss / gradient_accumulation
                scaled_loss.backward()

            if should_update:
                if config.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(config["max_grad_norm"])
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if rank == 0:
                    print(
                        f"epoch={epoch + 1} optimizer_step={optimizer_step}/{max_updates} "
                        f"loss={float(loss.item()):.6f} causal={parts['causal_loss']:.6f} "
                        f"calibration={parts['calibration_loss']:.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if optimizer_step >= max_updates:
                    stop = True
                    break
        if stop:
            break

    eval_metrics = evaluate_loss(
        model,
        eval_loader,
        encoder,
        config,
        config.get("max_eval_batches"),
    )
    if rank == 0:
        print(f"Evaluation: {eval_metrics}", flush=True)
        save_adapter(
            model,
            tokenizer,
            output_dir,
            config,
            {
                "optimizer_step": optimizer_step,
                "microstep": microstep,
                "eval_metrics": eval_metrics,
                "world_size": world_size,
                "effective_update_batch": batch_size
                * world_size
                * gradient_accumulation,
            },
        )
        print(f"Saved RelaCaTS v2 LoRA adapter to {output_dir}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
