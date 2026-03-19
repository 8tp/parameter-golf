#!/bin/bash
# Track B: Depth Recurrent Transformer (moonshot approach)
# Target: sub-1.15 BPB — unproven but potentially best-in-class
# 3 physical layers x 5 loops = 15 effective depth, wider model via int6 savings
#
# Run on 8xH100:
#   torchrun --standalone --nproc_per_node=8 train_spark.py
# Run on DGX Spark (dev):
#   DISABLE_COMPILE=1 GRAD_ACCUM=8 python3 train_spark.py

export NUM_PHYSICAL_LAYERS=3
export NUM_LOOPS=5
export MODEL_DIM=768
export NUM_HEADS=12
export NUM_KV_HEADS=6
export MLP_HIDDEN=2304          # 3x expansion
export USE_SWIGLU=1
export TIE_EMBEDDINGS=1
export LORA_RANK=0              # Set to 8-16 for per-loop LoRA experiments

# Optimizer
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
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
