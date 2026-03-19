#!/bin/bash
# Track B variant: Depth Recurrent + LoRA adapters
# Per-loop LoRA enables specialization across loop iterations
# Narrower dim to fit LoRA in the 16MB budget

export NUM_PHYSICAL_LAYERS=3
export NUM_LOOPS=5
export MODEL_DIM=640
export NUM_HEADS=10
export NUM_KV_HEADS=5
export MLP_HIDDEN=1920
export USE_SWIGLU=1
export TIE_EMBEDDINGS=1
export LORA_RANK=32

export MATRIX_LR=0.02
export LORA_LR=0.02
export MUON_MOMENTUM=0.99
export GRAD_CLIP_NORM=0.3
export WARMDOWN_ITERS=3000

export QAT_ENABLED=1
export QAT_START_FRAC=0.5
export QUANT_BITS=6

export VAL_STRIDE=64
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=200

export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
