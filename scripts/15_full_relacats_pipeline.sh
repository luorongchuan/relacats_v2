#!/usr/bin/env bash
set -euo pipefail

# Full RelaCaTS theory pipeline.  This script intentionally stops before a
# full training run unless RUN_TRAIN=1 is supplied by the caller.
#
# Examples:
#   bash relacats_v2/scripts/15_full_relacats_pipeline.sh
#   RUN_TRAIN=1 MODEL=qwen bash relacats_v2/scripts/15_full_relacats_pipeline.sh

ROOT="${ROOT:-/home/luorongchuan/workspace_135/RelaCaTS}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

BETA="${BETA:-0.5}"
STRATEGY_THRESHOLD="${STRATEGY_THRESHOLD:-0.86}"
LAMBDA_V="${LAMBDA_V:-0.5}"

python -m relacats_v2.data_creation.build_full_relacats_dataset \
  --beta "$BETA" \
  --strategy-similarity-threshold "$STRATEGY_THRESHOLD" \
  --lambda-v "$LAMBDA_V"

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "Full labels built. Training was NOT started (set RUN_TRAIN=1 to continue)."
  exit 0
fi

MODEL="${MODEL:-qwen}"
case "$MODEL" in
  qwen)
    CONFIG="relacats_v2/configs/qwen2_5_7b_full.json"
    ;;
  llama)
    CONFIG="relacats_v2/configs/llama3_1_8b_full.json"
    ;;
  deepseek)
    CONFIG="relacats_v2/configs/deepseek_1_5b_full.json"
    ;;
  *)
    echo "MODEL must be qwen, llama, or deepseek" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}" \
  torchrun --nproc_per_node=2 -m relacats_v2.model_training.train_full_relacats \
  --config-file "$CONFIG"
