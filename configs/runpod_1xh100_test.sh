#!/bin/bash
# RunPod 1xH100 test config
# Quick validation run to ensure everything works before 8xH100 submission

export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_HIDDEN=1536

# BigramHash
export BIGRAM_HASH_BUCKETS=10240
export BIGRAM_HASH_DIM=128

# Partial RoPE + LN Scale
export ROPE_FRAC=0.25
export LN_SCALE=1

# Shorter run for testing (1 GPU = fewer steps in 5 min)
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=300   # 5 min test (save credits)
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export WARMUP_STEPS=10
export WARMDOWN_ITERS=1500
export GRAD_ACCUM=8                # manual grad accum since single GPU

# Optimizer
export MATRIX_LR=0.02
export SCALAR_LR=0.04
export TIED_EMBED_LR=0.05
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=750
export MUON_WEIGHT_DECAY=0.04
export GRAD_CLIP_NORM=0.3

# QAT
export QAT_ENABLED=1
export QAT_START_FRAC=0.5

# Mixed quantization
export QUANT_MLP_BITS=5
export QUANT_ATTN_BITS=6

# EMA
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRAC=0.1

# Pruning
export PRUNE_FRACTION=0.03

# Evaluation (larger stride for faster eval on 1 GPU)
export VAL_STRIDE=256
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=50

# Misc
export TIE_EMBEDDINGS=1
export SEED=1337

echo "=== RunPod 1xH100 Test Run ==="
echo "Architecture: ${NUM_LAYERS}L dim=${MODEL_DIM} heads=${NUM_HEADS} kv=${NUM_KV_HEADS} mlp=${MLP_HIDDEN}"
echo "BigramHash: ${BIGRAM_HASH_BUCKETS} buckets x ${BIGRAM_HASH_DIM} dim"
echo "Quant: MLP=int${QUANT_MLP_BITS} attn=int${QUANT_ATTN_BITS}"
echo "EMA: decay=${EMA_DECAY} start_frac=${EMA_START_FRAC}"
echo "Partial RoPE: frac=${ROPE_FRAC} LN Scale: ${LN_SCALE}"
echo "Wallclock limit: ${MAX_WALLCLOCK_SECONDS}s (5 min test)"
echo "Single GPU — no torchrun needed"

python train_spark.py
