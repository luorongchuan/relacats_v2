"""Train RelaCaTS v2 with the selective hybrid confidence target.

This wrapper reuses the battle-tested LoRA/DDP/loss implementation from
``model_training.train_relacats`` and replaces only its training-data selection.

Training pool semantics:
- ``target_field`` (default ``hybrid_consistency``) supplies the calibration target;
- the same field is used for the > eta causal-LM selection unless
  ``causal_selection_field`` is explicitly configured otherwise;
- calibration examples are balanced over 0.05-wide confidence bins, following
  the CaTS training code;
- requested global train/eval and calibration/causal budgets are allocated
  exactly across datasets with deterministic largest-remainder rounding.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

from relacats_v2.model_training import train_relacats as base

HYBRID_MODE = "hybrid_direct"


def _exact_weighted_allocation(
    total: int,
    weights: Sequence[float],
) -> list[int]:
    """Allocate an integer total exactly according to non-negative weights.

    Independent ``round(total * weight / sum(weights))`` calls can lose or add
    a few records globally (for example 100000 -> 99998).  Largest-remainder
    allocation preserves the requested total exactly while remaining as close
    as possible to the desired proportions.  Ties are resolved by dataset
    order for deterministic reproduction.
    """

    if total < 0:
        raise ValueError("allocation total must be non-negative")
    if not weights:
        if total == 0:
            return []
        raise ValueError("cannot allocate a positive total over no weights")

    clean = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0 for weight in clean):
        raise ValueError("allocation weights must be finite and non-negative")
    weight_sum = sum(clean)
    if weight_sum <= 0:
        raise ValueError("allocation weights must sum to a positive value")

    quotas = [total * weight / weight_sum for weight in clean]
    allocation = [int(math.floor(quota)) for quota in quotas]
    remainder = total - sum(allocation)

    order = sorted(
        range(len(clean)),
        key=lambda index: (-(quotas[index] - allocation[index]), index),
    )
    for index in order[:remainder]:
        allocation[index] += 1

    if sum(allocation) != total:
        raise AssertionError("exact weighted allocation failed to preserve total")
    return allocation


def prepare_hybrid_examples(
    config: dict[str, Any], split: str, requested_total: int, seed: int
) -> list[dict[str, Any]]:
    dataset_root = base.resolve_path(config["dataset_root"])
    target_field = str(config.get("target_field", "hybrid_consistency"))
    causal_field = str(config.get("causal_selection_field", target_field))
    causal_ratio = float(config["causal_lm_ratio"])
    threshold = float(config["threshold"])
    if not 0.0 <= causal_ratio <= 1.0:
        raise ValueError("causal_lm_ratio must be in [0,1]")
    if requested_total <= 0:
        raise ValueError("requested_total must be positive")

    rng = random.Random(seed + (0 if split == "train" else 10_000))
    mixed: list[dict[str, Any]] = []
    specs = config["datasets"]

    raw_weights: list[float] = []
    for spec in specs:
        weight = float(spec.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid dataset weight: {spec!r}")
        raw_weights.append(weight)
    if sum(raw_weights) <= 0:
        raise ValueError("Dataset weights must sum to a positive value")

    # Preserve both global budgets exactly.  This avoids independent per-dataset
    # rounding changing 100000 into 99998 (or 1000 into 1001), and also keeps
    # the configured causal/calibration mixture exact at the global level.
    causal_total = int(round(requested_total * causal_ratio))
    calibration_total = requested_total - causal_total
    calibration_allocations = _exact_weighted_allocation(
        calibration_total, raw_weights
    )
    causal_allocations = _exact_weighted_allocation(causal_total, raw_weights)

    for spec, calibration_count, causal_count in zip(
        specs, calibration_allocations, causal_allocations
    ):
        name = spec["name"]
        records = base.load_records(dataset_root, name, split)

        valid: list[dict[str, Any]] = []
        for record in records:
            try:
                target = float(record[target_field])
                causal_score = float(record[causal_field])
            except (KeyError, TypeError, ValueError):
                continue
            if not (
                math.isfinite(target)
                and 0.0 <= target <= 1.0
                and math.isfinite(causal_score)
                and 0.0 <= causal_score <= 1.0
            ):
                continue
            if not record.get("transformed_prompt") or not record.get("response"):
                continue
            copied = dict(record)
            copied["_target"] = target
            copied["_causal_score"] = causal_score
            valid.append(copied)

        if records and not valid:
            raise ValueError(
                f"Dataset {name!r} has no rows with usable {target_field!r} and "
                f"{causal_field!r}. Rebuild the hybrid dataset first."
            )

        bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in valid:
            bins[min(int(record["_target"] / 0.05), 20)].append(record)

        per_bin = max(1, calibration_count // 21) if calibration_count else 0
        calibration: list[dict[str, Any]] = []
        for bin_index in range(21):
            calibration.extend(
                base._sample_without_replacement(bins[bin_index], per_bin, rng)
            )
        if len(calibration) < calibration_count:
            chosen = {id(item) for item in calibration}
            remaining = [item for item in valid if id(item) not in chosen]
            calibration.extend(
                base._sample_without_replacement(
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
                    "target_mode": HYBRID_MODE,
                    "target_method": record.get("target_method"),
                }
            )

        high_conf = [record for record in valid if record["_causal_score"] > threshold]
        for record in base._sample_without_replacement(high_conf, causal_count, rng):
            mixed.append(
                {
                    "task": "causal_lm",
                    "dataset_name": name,
                    "transformed_prompt": record["transformed_prompt"],
                    "response": record["response"],
                    "target": -1.0,
                    "selection_score": record["_causal_score"],
                    "target_mode": HYBRID_MODE,
                    "target_method": record.get("target_method"),
                }
            )

    rng.shuffle(mixed)
    if not mixed:
        raise ValueError(f"No usable {split} examples were constructed")
    return mixed


def _hybrid_resolver(
    *, ssc: Any, relssc: Any, relation_valid_ratio: Any,
    target_mode: str, lambda_rel: float
) -> float:
    if str(target_mode).strip().lower() == HYBRID_MODE:
        # main() calls the resolver once only to validate configuration before
        # data loading. Actual hybrid targets are read directly from JSONL.
        value = float(relssc)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("hybrid validation target must be in [0,1]")
        return value
    return _ORIGINAL_RESOLVER(
        ssc=ssc,
        relssc=relssc,
        relation_valid_ratio=relation_valid_ratio,
        target_mode=target_mode,
        lambda_rel=lambda_rel,
    )


_ORIGINAL_RESOLVER = base.resolve_confidence_target
base.TARGET_MODES = frozenset(set(base.TARGET_MODES) | {HYBRID_MODE})
base.resolve_confidence_target = _hybrid_resolver
base.prepare_mixed_examples = prepare_hybrid_examples


if __name__ == "__main__":
    base.main()
