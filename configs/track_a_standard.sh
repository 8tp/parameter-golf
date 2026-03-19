#!/bin/bash
# Track A: Standard 9-layer (proven competitive approach)
# Target: ~1.16 BPB on 8xH100 in 10 minutes
# Uses all winning techniques: int6, QAT, MLP 3x, sliding window, Muon 0.99
#
# Run on 8xH100:
#   torchrun --standalone --nproc_per_node=8 train_spark.py
# Run on 1xH100 (testing):
#   GRAD_ACCUM=8 python3 train_spark.py
# Run on DGX Spark (dev):
#   DISABLE_COMPILE=1 GRAD_ACCUM=8 python3 train_spark.py

export NUM_PHYSICAL_LAYERS=9
export NUM_LOOPS=1
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_HIDDEN=1536          # 3x expansion (winning technique)
export USE_SWIGLU=1
export TIE_EMBEDDINGS=1
export LORA_RANK=0

# Optimizer (tuned to match top entries)
export MATRIX_LR=0.02
export MUON_MOMENTUM=0.99
export GRAD_CLIP_NORM=0.3
export WARMDOWN_ITERS=3000

# QAT + quantization
export QAT_ENABLED=1
export QAT_START_FRAC=0.5
export QUANT_BITS=6

# Eval
export VAL_STRIDE=64
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=200

# Training budget
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600  # 10 min cap for leaderboard
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
