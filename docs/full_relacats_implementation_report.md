# Full RelaCaTS implementation report

## Scope

The repository now contains an explicit full-theory path corresponding to the uploaded RelaCaTS theory document, while keeping the previous conservative residual-target path available for controlled ablation.

## Theory-to-code mapping

| Theory component | Equation | Implementation |
|---|---:|---|
| Relation canonicalization | (13)-(17) | existing `core/relational_views.py`, `core/canonicalization.py` |
| Dependency blocks | (18) | `core/full_relacats.py::annotate_dependency_weights` |
| Dependency weight | (19) | `d_i = |B_i|^{-beta}` |
| Weighted RelSSC | (20)-(21) | `compute_relssc_full`, using `r_i d_i C_i` |
| View support | (22) | `compute_fragility` |
| Relation aggregate support | (23) | `compute_fragility` |
| Fragility pseudo-label | (24)-(25) | `compute_fragility` / `attach_full_targets` |
| q instruction | (26) | full trainer confidence query |
| f instruction | (27) | `fragility_suffix` + full trainer fragility query |
| Calibration loss | (28) | SmoothL1 in `train_full_relacats.py` |
| Fragility loss | (29) | BCE in `train_full_relacats.py` |
| Effective response weight | (30) | q(1-f), used by rank/controller |
| Pairwise rank loss | (31) | optional same-question microbatch rank loss |
| Filtered CLM | (32) | `generation_filter_mode=relssc_fragility` |
| Joint objective | (33) | q + lambda_f f + lambda_r rank + scaled CLM |
| Full vote | (34)-(36) | `effective_vote_state` |
| STOP/SAMPLE/INTERVENE | (37) | `controller_state` / `evaluation/full_controller.py` |

## Dependency-clustering implementation decision

The theory defines dependency blocks semantically but intentionally does not prescribe a single clustering algorithm. The implementation therefore supports two levels:

1. **Preferred:** upstream code writes `dependency_cluster_id` or `strategy_cluster_id`; these IDs are used directly.
2. **Fallback:** deterministic lexical-Jaccard clustering over reasoning text, with configurable similarity threshold. Final-answer lines are removed before similarity calculation and, by default, responses supporting different canonical answers are not merged into the same fallback block.

This makes the current implementation reproducible without adding an embedding-model dependency, while leaving a clean interface for a stronger semantic strategy clusterer in later experiments.

## Relation reliability

The existing deterministic option-permutation relations continue to use `r_g=1`, which is consistent with the theory document's treatment of programmatically exact transformations. `compute_relssc_full` already accepts non-unit `relation_weight` values if a later transformation generator supplies estimated relation reliability.

## Safe default retained after the v1 failure

The exact theory Eq. (32) uses high relation confidence and low fragility to select CLM examples. Because the previous RelaCaTS-v1 experiment showed a large generation-quality collapse on Llama Object Counting, the *provided full configs* deliberately default to:

```text
generation_filter_mode = ssc
```

This keeps the original SSC-based generation filter while enabling dependency-corrected RelSSC and the fragility objective. The exact theory filter is fully implemented and can be activated with:

```text
generation_filter_mode = relssc_fragility
eta_c = 0.75
eta_f = 0.25
```

This gives a clean ablation rather than confounding all changes at once.

## Test-time intervention boundary

The CPU controller computes and emits all three theory actions. For `INTERVENE`, it records that a relation challenge is required but does not invent a transformed response. A separate GPU orchestration stage should construct a valid relation view and obtain the challenged model response. This is intentional: a CPU aggregator must not claim that an intervention was executed when no model call occurred.

## New files

- `core/full_relacats.py`
- `core/runtime_controller.py`
- `data_creation/build_full_relacats_dataset.py`
- `model_training/train_full_relacats.py`
- `evaluation/calculate_fragility.py`
- `evaluation/full_controller.py`
- `tests/test_full_relacats.py`
- `configs/qwen2_5_7b_full.json`
- `configs/llama3_1_8b_full.json`
- `configs/deepseek_1_5b_full.json`
- `scripts/15_full_relacats_pipeline.sh`

## What has not been run

No full training was started. The GitHub connector used for these edits does not execute the user's two-A100 environment, so GPU training and end-to-end runtime validation still need to be run on the user's server. The first recommended runtime step is the pure unit-test suite, followed by full-label construction on a small question subset, then a one-step/tiny smoke training run before any long training.
