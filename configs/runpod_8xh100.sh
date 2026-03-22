#!/bin/bash
# RunPod 8xH100 SXM submission config
# Target: sub-1.13 BPB on FineWeb validation
# Budget: ~$3.59 per 10-minute run ($25 total credits)
#
# Architecture: 11-layer transformer + BigramHash + SmearGate
# Frontier techniques: Partial RoPE, LN Scale, EMA, mixed int5/int6

export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_HIDDEN=1536          # 3x expansion with relu^2

# BigramHash
export BIGRAM_HASH_BUCKETS=10240
export BIGRAM_HASH_DIM=128

# Partial RoPE: position encoding on 25% of head dims
export ROPE_FRAC=0.25

# LN Scale: damp deeper layers
export LN_SCALE=1

# Training schedule (tuned for 8xH100, ~12500 steps in 10 min)
export ITERATIONS=20000         # will be capped by wallclock
export MAX_WALLCLOCK_SECONDS=580  # stop 20s before 10 min limit (eval needs time)
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3000

# Optimizer (matching SOTA)
export MATRIX_LR=0.02
export SCALAR_LR=0.04
export TIED_EMBED_LR=0.05
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WEIGHT_DECAY=0.04
export GRAD_CLIP_NORM=0.3

# QAT
export QAT_ENABLED=1
export QAT_START_FRAC=0.5

# Mixed quantization
export QUANT_MLP_BITS=5
export QUANT_ATTN_BITS=6

# EMA (replaces SWA — better for short runs)
export EMA_ENABLED=1
export EMA_DECAY=0.997
export EMA_START_FRAC=0.1
export EMA_EVERY=10

# Pruning
export PRUNE_FRACTION=0.03

# Evaluation
export VAL_STRIDE=64
export VAL_LOSS_EVERY=1000
export TRAIN_LOG_EVERY=100

# Misc
export TIE_EMBEDDINGS=1
export LOGIT_SOFTCAP=30.0
export SEED=1337

echo "=== RunPod 8xH100 Submission ==="
echo "Architecture: ${NUM_LAYERS}L dim=${MODEL_DIM} heads=${NUM_HEADS} kv=${NUM_KV_HEADS} mlp=${MLP_HIDDEN}"
echo "BigramHash: ${BIGRAM_HASH_BUCKETS} buckets x ${BIGRAM_HASH_DIM} dim"
echo "Quant: MLP=int${QUANT_MLP_BITS} attn=int${QUANT_ATTN_BITS}"
echo "EMA: decay=${EMA_DECAY} start_frac=${EMA_START_FRAC}"
echo "Partial RoPE: frac=${ROPE_FRAC} LN Scale: ${LN_SCALE}"
echo "Wallclock limit: ${MAX_WALLCLOCK_SECONDS}s"

# Launch distributed training on 8 GPUs
torchrun --nproc_per_node=8 train_spark.py
