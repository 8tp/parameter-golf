# Parameter Golf Strategy

## Two-Track Approach

### Track A: Standard 9-Layer (Quick Win)
Proven competitive architecture with all winning techniques stacked.

- 9 unique transformer layers, dim=512, MLP 3x (1536)
- Int6 quantization + STE QAT + fp16 embeddings
- Sliding window eval (stride=64)
- Muon momentum=0.99, LR=0.02, warmdown=3000
- Gradient clipping 0.3
- **Target: ~1.16 BPB** (competitive with current leaderboard)

```bash
source configs/track_a_standard.sh
torchrun --standalone --nproc_per_node=8 train_spark.py
```

### Track B: Depth Recurrent Transformer (Moonshot)
Novel architecture — no depth recurrence submission has won yet, but our parameter
efficiency advantage (22M params via int6) could leapfrog the field.

- 3 physical layers x 5 loops = 15 effective depth, dim=768
- SwiGLU MLP 3x (2304), optional per-loop LoRA adapters
- Per-effective-layer control params (NOT shared across loops)
- U-net skip connections over effective layers
- **Target: sub-1.15 BPB** (unproven)

```bash
source configs/track_b_recurrent.sh
torchrun --standalone --nproc_per_node=8 train_spark.py
```

## Key Techniques Implemented

| Technique | BPB Impact | Status |
|-----------|-----------|--------|
| STE QAT | ~-0.049 (quant gap) | Implemented |
| Int6 quantization | Saves ~3.7MB (25%) | Implemented |
| SwiGLU MLP 3x | ~-0.019 | Implemented |
| Sliding window eval | ~-0.034 (free) | Implemented |
| fp16 tied embeddings | ~-0.007 | Implemented |
| Muon momentum=0.99 | ~-0.005 | Implemented |
| Gradient clipping 0.3 | Stabilizes training | Implemented |
| Warmdown 3000 iters | Better final quality | Implemented |
| zstd-22 compression | Frees artifact bytes | Implemented (needs zstandard) |
| Depth recurrence | Novel architecture | Implemented |
| Per-loop LoRA | Per-iteration specialization | Implemented |

## Hardware

- **Development:** DGX Spark "larry" (GB10 Blackwell, 128GB, CUDA 13.0)
  - torch.compile does NOT work (Triton SM shared memory too small)
  - ~12.8s/step uncompiled — use for architecture search only
  - SSH: `larry@172.16.21.170`
- **Submission:** 8xH100 SXM via RunPod ($21.52/hr)
  - torch.compile works, ~48ms/step
  - Credits requested via modelcraft.runpod.io

## Parameter Budget (16MB limit)

| Config | Params | Compressed Est. |
|--------|--------|----------------|
| Track A: 9-layer int6 | ~17M | ~11.5MB |
| Track B: 3x5 dim=768 int6 | ~22M | ~14.8MB |
| Track B: 3x5 dim=640 LoRA-32 int6 | ~14.6M | ~13.9MB |
| Baseline (9-layer int8) | ~17M | ~15.8MB |

## Leaderboard Context (March 19, 2026)

Current SOTA: **1.1618 BPB** (PR #102)
Key winning techniques: MLP 3x + STE int6 QAT + sliding window + Muon 0.99

## Running Experiments

```bash
# DGX Spark (development, no compile)
source configs/track_a_standard.sh
DISABLE_COMPILE=1 GRAD_ACCUM=8 ITERATIONS=5000 VAL_LOSS_EVERY=500 python3 train_spark.py

# 1xH100 (quick validation)
source configs/track_a_standard.sh
GRAD_ACCUM=8 python3 train_spark.py

# 8xH100 (final submission)
source configs/track_a_standard.sh
torchrun --standalone --nproc_per_node=8 train_spark.py
```
