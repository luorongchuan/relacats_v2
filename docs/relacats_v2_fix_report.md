# RelaCaTS-v2 Fix Report

Date: 2026-09-03

## Scope and isolation

The project was copied from `relacats_v1` to `relacats_v2` before edits. All
source changes and generated audit outputs described here are under
`relacats_v2`; no training was started in this pass. The v2 package has its own
Git root at `/home/luorongchuan/workspace_135/RelaCaTS/relacats_v2`.
The verification command was `pwd && git rev-parse --show-toplevel` from the
v2 directory, and both paths resolved to that directory. The source tree at
`/home/luorongchuan/workspace_135/RelaCaTS/relacats_v1` was only read.

## Modified files

Functional v2 changes are in:

- `common.py`, `core/__init__.py`, `core/targets.py`
- `evaluation/__init__.py`, `evaluation/_artifacts.py`,
  `evaluation/aggregate.py`, `evaluation/answer_parsing.py`,
  `evaluation/calculate_confidence.py`, `evaluation/generate_responses.py`,
  `evaluation/reaggregate_existing.py`, `evaluation/synthetic_smoke.py`
- `model_training/merge_lora.py`, `model_training/train_relacats.py`
- `scripts/01_generate_all_models.sh`, `scripts/01_generate_relational_data.sh`,
  `scripts/02_build_relssc_dataset.sh`, `scripts/03_smoke_train.sh`,
  `scripts/04_train_relacats.sh`, `scripts/05_merge_model.sh`,
  `scripts/06_generate_eval.sh`, `scripts/07_calculate_confidence.sh`,
  `scripts/08_evaluate.sh`, `scripts/09_wrong_consensus_diagnosis.sh`,
  `scripts/10_final_audit_and_train.sh`, `scripts/11_fast_train_gpu67.sh`,
  `scripts/12_evaluate_serial_tp2_gpu67.sh`,
  `scripts/13_reaggregate_existing_v2.sh`, and `scripts/run_train_pipeline_gpu67.sh`
- v2 JSON configs and the parser, evaluator, DeepSeek-protocol, and target
  tests under `tests/`.

## Implemented changes

- Unified the CPU evaluator and canonicalized all report labels to the
  `RelaCaTS-*` namespace.
- Replaced answer guessing with a strict explicit-final-answer parser,
  including option, boxed, decimal-plus-option, numeric, and number-word forms.
- Added parser, ASC, target, protocol, data-pipeline, and script tests.
- Kept ordinary ASC count-based and made `RelaCaTS-ASC` use the same
  confidence-weighted `V_k` state for stopping and final voting.
- Calibrated ES/ASC-family thresholds on the calibration partition only;
  persisted model/dataset-specific thresholds and applied a target-budget hard
  cap in both calibration and test execution.
- Added `ssc`, `relssc_replace`, and `residual` target modes. The default
  residual coefficient is `lambda_rel=0.5`, while generation-example
  selection remains based on original SSC.
- Preserved DeepSeek tokenizer chat templates and aligned its generation
  protocol (`max_new_tokens=2048`, `max_model_len=8192`) without forcing an
  extra `<think>` prefix.
- Added CPU reaggregation in
  `evaluation/reaggregate_existing.py` and
  `scripts/13_reaggregate_existing_v2.sh`.

## Tests

Command:

```bash
cd /home/luorongchuan/workspace_135/RelaCaTS
/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python -m pytest -q relacats_v2/tests
```

Result: **94 passed, 2 warnings, 25 subtests passed**. The warnings are from
the environment's deprecated `pynvml` package and unavailable NVML; no test
failed.

## CPU reaggregation

The final output is in
`outputs/eval_outputs_v2`. It contains 3 models x 3 datasets, separate
validation/test reports, one threshold artifact per model/dataset, and 90
budget-16 method rows (10 methods per model/dataset). Every reported row has
`accuracy`, `actual_avg_samples`, `valid_samples`, and `invalid_rate`.

The existing confidence artifacts were labelled `split=test`, so the audit
used a deterministic SHA-256 question-id partition (`validation_fraction=0.2`,
`seed=42`) to select thresholds and held out the remaining 80% for reporting.
This is **not an official validation split**; the manifest records
`official_validation_split=false`. Test aggregation reloads persisted
validation thresholds and does not tune on test questions.

Raw response-pool invalid rates:

| Model | Questions | Samples | Invalid samples | Invalid rate |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 4,407 | 141,024 | 719 | 0.5098% |
| Llama-3.1-8B-Instruct | 4,407 | 141,024 | 24,985 | 17.7168% |
| DeepSeek-R1-Distill-Qwen-1.5B | 4,407 | 141,024 | 54,604 | 38.7197% |

All final dynamic test rows use `budget_cap=16` and have
`actual_avg_samples <= 16`; the final output has no non-finite metric values.
The previous first pass is retained at
`outputs/eval_outputs_v2_first_pass_20260903_0945`; it had 10 dynamic rows
above the target budget, which is why it is not the formal output.
The pre-path-fix cap output is also retained at
`outputs/eval_outputs_v2_cap_pre_pathfix_20260903_1010`; all persisted
`threshold_calibration_file` references in the formal output now resolve to
files under `outputs/eval_outputs_v2`.

Question-weighted averages over the three audited datasets are shown below;
the per-model/per-dataset/per-method values (including `valid_samples` and
`invalid_rate`) are in `summary.csv` and `model_method_summary.csv`.

| Model | Fixed methods (SC/CISC/Self-Certainty/Best-of-N/RelaCaTS-SC) | RelaCaTS-ES | ASC | RelaCaTS-ASC | ESC | RASC |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 16.000 | 15.992 | 3.804 | 3.807 | 16.000 | 15.997 |
| Llama-3.1-8B-Instruct | 16.000 | 16.000 | 6.552 | 6.548 | 16.000 | 16.000 |
| DeepSeek-R1-Distill-Qwen-1.5B | 16.000 | 15.983 | 6.757 | 6.751 | 16.000 | 15.992 |

The complete model-level metric rows are in
`outputs/eval_outputs_v2/model_method_summary.csv`. They include all 10
methods and the requested `accuracy`, `actual_avg_samples`, `valid_samples`,
and `invalid_rate` fields; the 90 unrounded model/dataset/method rows are in
`outputs/eval_outputs_v2/summary.csv`.

Because the legacy reports use the full test pool while v2 reports use the
80% deterministic held-out partition, `old_new_comparison.csv/json` records
the scope difference and sets `directly_comparable=false`. Differences there
must not be interpreted as a statistically controlled accuracy improvement.

Descriptive accuracy deltas (new minus legacy) from
`old_new_comparison.csv` are:

| Method | Comparisons | Mean delta | Min | Max |
|---|---:|---:|---:|---:|
| SC | 9 | +0.0043 | -0.0065 | +0.0192 |
| CISC | 3 | +0.0007 | -0.0095 | +0.0085 |
| Self-Certainty | 3 | +0.0003 | -0.0054 | +0.0076 |
| RelaCaTS-SC | 9 | +0.0026 | -0.0086 | +0.0088 |
| Best-of-N | 9 | +0.0028 | -0.0065 | +0.0221 |
| RelaCaTS-ES | 9 | +0.0132 | -0.0147 | +0.0898 |
| ASC | 9 | -0.0011 | -0.0210 | +0.0154 |
| RelaCaTS-ASC | 9 | -0.0031 | -0.0204 | +0.0148 |
| ESC | 3 | -0.0091 | -0.0215 | +0.0032 |
| RASC | 3 | +0.0043 | -0.0207 | +0.0498 |

The current confidence artifacts contain no native CISC, Self-Certainty, or
RASC score fields. Those three rows are therefore explicitly marked
`implementation_status=proxy` in each report; they are not claimed as strict
reproductions of the authors' native scoring implementations.

The copied response/confidence metadata still contains the original v1
generation provenance, including the earlier 1024-token setting. This pass
reused those artifacts as requested; the v2 DeepSeek defaults apply to future
generation runs and were not retroactively applied to existing responses.

No full training was launched. The next training run should use the v2 target
configuration only after review of this audit.

The fresh old-model retest entry point is
`scripts/14_retest_old_models_gpu67.sh`. It is deliberately separate from
the CPU reaggregation above: it loads one author CaTS checkpoint or one
already-merged RelaCaTS-v1 checkpoint at a time on physical GPUs 6 and 7,
writes only to `outputs/eval_outputs_v2_retest_old_models`, and never calls
training or merge code. GPU execution was not started during this audit
because those devices were occupied by an unrelated process.
