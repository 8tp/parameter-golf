# Parameter Golf — Findings & Plan

## Competition Overview
- **Goal**: Train best LM that fits in 16MB, trains in 10 min on 8xH100 SXM
- **Metric**: Bits-per-byte (BPB) on FineWeb validation
- **Prize**: $1M in compute credits, hiring pipeline for early-career researchers
- **Deadline**: April 30, 2026
- **Repo**: https://github.com/openai/parameter-golf

## Current Leaderboard (as of March 21, 2026)
| Rank | BPB | Key Techniques |
|------|-----|----------------|
| 1 (merged) | **1.1428** | 10L, mixed int5/int6, BigramHash, SWA, MuonWD=0.04 |
| 1 (unmerged) | **1.1246** | Tight SWA + Shared Value Embed + Partial RoPE + LN Scale + XSA4 |
| Baseline | **1.2244** | 9L, int8, zlib |

## Our Results

### Run 1: 1xH100 Validation (5 min)
- **600 steps**, 500ms/step
- val_bpb: **5.964** (step 601, pre-SWA)
- post_swa val_bpb: 7.195 (SWA hurt on short run)
- Model size: 6.5MB (well within budget)
- **Purpose**: Validated script works end-to-end

### Run 2: 8xH100 (10 min) — BPB Bug
- 11 layers, 60ms/step → 70ms after EMA (every step)
- val_bpb numbers were **4x inflated** due to BPB calculation bug
- EMA overhead: 15% throughput loss (copying state dict to CPU every step)
- **Killed early** to fix bugs

### Run 3: 8xH100 (10 min) — BPB Fixed, QAT Never Fired
- 11 layers, ~60ms/step, EMA every 10 steps (fixed overhead)
- 9,196 steps in 580s
- val_bpb: **1.2222** (pre-EMA), **1.2217** (post-EMA)
- **Roundtrip val_bpb: 3.20** — catastrophic quantization degradation
- **Root cause**: QAT_START_FRAC=0.5 → activate at step 10000, but wallclock stopped at 9196. QAT never activated.
- **Size: 18.87MB** — exceeded 16MB by 2.87MB (11 layers too big, zlib not zstd)

### Run 4: 8xH100 (10 min) — All Fixes Applied
- 10 layers, QAT_START_FRAC=0.3, 5% pruning, zstandard installed
- ~55ms/step (fastest run)
- val_bpb at step 2000: **1.3196**
- **Connection dropped mid-run** — results lost
- Lesson: Use `tmux` to protect against disconnects

## Bugs Found & Fixed

### 1. BPB Calculation Bug (FIXED)
- `val_token_count` counted all `seq_len` tokens per window
- `val_byte_count` only counted `stride` tokens
- Result: BPB inflated by `seq_len / stride` ratio (16x for stride=64)
- Fix: Separate counters for loss averaging vs BPB scoring

### 2. QAT Timing Bug (FIXED)
- `QAT_START_FRAC` computed against `args.iterations` (20000)
- But wallclock stops training at ~9200 steps
- 50% of 20000 = step 10000 → never reached
- Fix: Lowered to 0.3 (activates at step 6000)

### 3. EMA Overhead (FIXED)
- Copying full state dict to CPU every step: 15% throughput loss
- Fix: `EMA_EVERY=10` with effective decay `decay^N`

### 4. GQA Compatibility (FIXED)
- `enable_gqa` kwarg not available in PyTorch < 2.5
- Fix: Manual KV head expansion (moot on official template with PyTorch 2.9.1)

### 5. Size Budget (FIXED)
- 11 layers + zlib exceeded 16MB
- Fix: Back to 10 layers, zstd-22 compression, 5% pruning

## Our Architecture (v2)

```
10 layers, dim=512, heads=8, kv_heads=4, mlp=1536 (3x relu^2)
BigramHash: 10240 buckets x 128 dim
SmearGate: learned per-dim blending with previous token
U-net skip connections (5 encoder, 5 decoder)
Partial RoPE: 25% of head dims (16/64)
LN Scale: 1/sqrt(layer+1) damping
Tied embeddings, logit softcap=30.0
~25.5M params → mixed int5(MLP)/int6(attn) + zstd-22 → ~6-7MB
```

## Techniques We Have
- [x] Muon optimizer (WD=0.04, momentum warmup 0.92→0.99)
- [x] Mixed int5/int6 quantization
- [x] STE Quantization-Aware Training
- [x] EMA (decay 0.997, every 10 steps)
- [x] BigramHash + SmearGate
- [x] Orthogonal init + muP scaling
- [x] relu^2 MLP at 3x expansion
- [x] U-net skip connections
- [x] Partial RoPE (25% of head dims)
- [x] LN Scale (1/sqrt(layer+1))
- [x] Magnitude pruning (5%)
- [x] zstd-22 compression
- [x] Sliding window eval (stride=64)
- [x] Logit softcapping (30.0)

## Techniques to Explore (Not Yet Implemented)
- [ ] **XSA (Exclusive Self-Attention)** — used by top 3 unmerged entries
- [ ] **Test-Time Training (TTT)** — 3 epochs SGD on already-evaluated val tokens (~0.002 BPB)
- [ ] **FlashAttention 3** — Hopper-optimized, faster training = more steps
- [ ] **Shared Value Embeddings** — extra learned embeddings mixed into attention values
- [ ] **Gradient-Guided Quantization** — smarter bit allocation per layer
- [ ] **NTK-RoPE** — improved position encoding extrapolation
- [ ] **Larger vocab (4096/8192)** — pre-tokenized datasets available on HuggingFace

## Infrastructure Notes

### RunPod Setup (Official Template)
- Template: https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j
- PyTorch 2.9.1 + CUDA 12.8 pre-installed
- Must `pip install zstandard` (not in template)
- VPN required for SSH (home network blocks port 22)
- SCP/rsync don't work through RunPod SSH proxy — use git
- **Always use tmux** to protect against web terminal disconnects

### Quick Start
```bash
tmux new -s train
cd /workspace && git clone https://github.com/8tp/parameter-golf.git && cd parameter-golf
pip install zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024
bash configs/runpod_8xh100.sh
```

### Credits
- Started with $25 RunPod credits
- ~$3.59 per 10-min run on 8xH100
- Remaining: ~$5.61 (1 more run)
- Consider requesting more credits from OpenAI (competition sponsors $1M in compute)

## Next Session Plan

### Priority 1: Get a Clean Complete Run
1. Start pod with official template
2. Use `tmux` from the start
3. Install zstandard, clone repo, download data
4. Run `configs/runpod_8xh100.sh`
5. Verify: QAT activates, roundtrip BPB is close to pre-quant, size < 16MB

### Priority 2: Optimize for Better BPB
- If roundtrip works and size fits: try 11 layers + more aggressive compression
- Implement XSA for last 4 layers (biggest gap vs unmerged SOTA)
- Try TTT for free ~0.002 BPB improvement
- Experiment with higher vocab sizes if data available

### Priority 3: Submit
- Fork openai/parameter-golf
- Create record folder with README, train_gpt.py, submission.json, logs
- Need 3+ seed runs for statistical significance (p < 0.01)
- PR to main repo

### Request More Credits
- OpenAI is sponsoring $1M in compute for this competition
- Check Discord #parameter-golf-announcements for credit programs
- RunPod referral/promo codes may also help
