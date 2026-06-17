#!/usr/bin/env bash
# RunPod entrypoint: fetch data, train, and leave the final checkpoint where it can be
# downloaded before the pod is terminated. Keep it lean — no persistent volume needed.
#
# Override the config via env:  CONFIG=configs/runpod_a100.yaml bash docker/runpod_entrypoint.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/runpod_24gb.yaml}"

echo "==> Preparing COCO (downloads val2017 + captions if missing)"
python scripts/prepare_coco.py --root data/coco --download

echo "==> wandb"
# Set WANDB_API_KEY in the pod env, or this falls back to offline logging.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "==> Training with $CONFIG"
accelerate launch scripts/train.py --config "$CONFIG"

echo "==> Done. Download the final checkpoint from outputs/ before terminating the pod:"
find outputs -name "*.safetensors" -o -name "*.pt" | sort
