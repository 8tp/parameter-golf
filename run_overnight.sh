#!/bin/bash
# Overnight training runs on DGX Spark
# DISABLE_COMPILE=1 required — GB10 Triton SM resources too small
set -e

cd ~/parameter-golf
source .venv/bin/activate

echo "=== Overnight Training ==="
echo "Started: $(date)"
echo ""

# Run 1: Track A — Standard 9-layer
echo ">>> Track A: Standard 9-layer (dim=512, MLP 3x)"
source configs/track_a_standard.sh
DISABLE_COMPILE=1 \
GRAD_ACCUM=8 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_spark.py 2>&1 | tee logs/overnight_track_a.log

echo ""
echo "Track A finished: $(date)"
echo ""

# Run 2: Track B — Depth Recurrent
echo ">>> Track B: Depth Recurrent (3x5, dim=640, MLP 3x)"
source configs/track_b_recurrent.sh
DISABLE_COMPILE=1 \
GRAD_ACCUM=8 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_HIDDEN=1920 \
ITERATIONS=5000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 \
python3 train_spark.py 2>&1 | tee logs/overnight_track_b.log

echo ""
echo "All runs finished: $(date)"
