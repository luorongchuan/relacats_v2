# RelaCaTS v2

This repository contains two compatible RelaCaTS paths:

1. **Conservative v2 control**: relation-aware SSC/RelSSC targets, the residual target, strict parser/ASC fixes, and unified CaTS evaluation.
2. **Full theory path**: dependency correction, weighted RelSSC, consensus fragility, joint q/f distillation, optional ranking loss, optional Eq. (32) generation filtering, and the STOP/SAMPLE/INTERVENE controller.

## Full theory components

The implementation follows the equations in `RelaCaTS_研究痛点与理论分析`:

- Eq. (19): `d_i = |B_i|^{-beta}` in `core/full_relacats.py`.
- Eqs. (20)-(21): relation/dependency-weighted RelSSC via `compute_relssc_full`.
- Eqs. (22)-(25): consensus fragility in `compute_fragility`.
- Eqs. (26)-(33): same-LM q/f instruction distillation in `model_training/train_full_relacats.py`.
- Eqs. (34)-(37): effective vote and STOP/SAMPLE/INTERVENE controller in `core/full_relacats.py` + `core/runtime_controller.py`.

The theory defines a *dependency block* but does not prescribe one universal clustering algorithm. The repository therefore uses the following deterministic contract:

- if an upstream `dependency_cluster_id` / `strategy_cluster_id` exists, it is used exactly;
- otherwise, a lexical-Jaccard strategy clustering fallback is used with configurable `beta` and similarity threshold.

A semantic/embedding clusterer can be plugged in upstream simply by writing cluster IDs before the full dataset builder.

## Full offline label construction

```bash
python -m relacats_v2.data_creation.build_full_relacats_dataset \
  --beta 0.5 \
  --strategy-similarity-threshold 0.86 \
  --lambda-v 0.5
```

This writes to:

```text
relacats_v2/outputs/full_relacats_dataset
```

without overwriting the conservative v2 dataset.

## Full training

Three full configs are provided:

```text
configs/qwen2_5_7b_full.json
configs/llama3_1_8b_full.json
configs/deepseek_1_5b_full.json
```

Example:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
  -m relacats_v2.model_training.train_full_relacats \
  --config-file relacats_v2/configs/qwen2_5_7b_full.json
```

The full trainer jointly optimizes:

```text
L_total = L_cal + lambda_f L_frag + lambda_r L_rank + omega L_CLM
```

where `L_rank` is active when comparable same-question calibration examples occur in the same microbatch.

### Safe control vs exact Eq. (32)

The full trainer supports both generation filters:

```text
generation_filter_mode = ssc
```

preserves the conservative v2 control and minimizes generation-distribution drift, while

```text
generation_filter_mode = relssc_fragility
```

implements the theory's high-RelSSC / low-fragility Eq. (32) filter using `eta_c` and `eta_f`.

The provided full configs intentionally default to `ssc` so the first full experiment isolates the q/f/dependency gains without repeating the v1 generation-collapse failure mode. Change only this field for the exact Eq. (32) ablation.

## Full test-time fragility and controller

After normal confidence calculation, predict fragility:

```bash
python -m relacats_v2.evaluation.calculate_fragility \
  --model /path/to/merged/model \
  --input /path/to/confidence.jsonl \
  --output /path/to/full_confidence.jsonl
```

Then evaluate the three-action controller:

```bash
python -m relacats_v2.evaluation.full_controller \
  --input /path/to/full_confidence.jsonl \
  --output /path/to/controller.jsonl \
  --tau-support 0.8 \
  --tau-fragility 0.25 \
  --max-budget 16
```

The CPU controller reports `INTERVENE` explicitly but does **not** fabricate an intervention response. A GPU relation-challenge stage should consume those states; this keeps CPU aggregation scientifically honest.

## Tests

The existing parser/ASC/residual tests remain unchanged. Full-theory pure-function tests are in:

```text
tests/test_full_relacats.py
```

No full training is launched automatically by repository code or scripts. `scripts/15_full_relacats_pipeline.sh` stops after full-label construction unless `RUN_TRAIN=1` is explicitly set.
