# RunPod 8xH100 Setup Guide

## Use the Official Template

Deploy from: https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j

This template includes:
- PyTorch 2.9.1 + CUDA 12.8 (all deps pre-installed)
- Python 3.12
- 50GB volume + 50GB container disk
- No pip installs needed

## Quick Start (copy-paste on pod)

```bash
cd /workspace
git clone https://github.com/8tp/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
bash configs/runpod_8xh100.sh
```

## Step-by-step

### 1. SSH into the pod
```bash
ssh <POD_ID>@ssh.runpod.io -i ~/.ssh/id_ed25519
```
Note: VPN required — home network blocks port 22.

### 2. Clone and setup
```bash
cd /workspace
git clone https://github.com/8tp/parameter-golf.git
cd parameter-golf
```

### 3. Download dataset (~324MB)
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 4. Run the submission
```bash
bash configs/runpod_8xh100.sh
```

## What to Expect

- ~60-70ms/step on 8xH100 with DDP
- ~8,000-12,500 steps in 10 minutes
- Auto-stops at 580s (20s buffer for final eval + EMA)
- Final output: val_bpb score, quantized model size

## Architecture (v2 — Frontier)

- **11 layers** (up from 10)
- **Partial RoPE**: position encoding on 25% of head dims
- **LN Scale**: 1/sqrt(layer+1) damping for deeper layers
- **EMA** (decay 0.997): replaces SWA for better short-run performance
- Mixed int5/int6 quantization + STE QAT
- BigramHash(10240) + SmearGate
- Muon WD=0.04, relu^2 MLP at 3x, U-net skips
- zstd-22 compression, sliding eval stride=64

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH times out | Connect VPN first, verify with `nc -zv ssh.runpod.io 22` |
| `No module 'sentencepiece'` | Use the official template (has all deps) |
| `No files found for pattern` | Run the dataset download script (Step 3) |
| NCCL timeout | Check all 8 GPUs visible: `nvidia-smi -L` |
| `enable_gqa` TypeError | Template has PyTorch 2.9.1, this is fixed |
| SCP/rsync fails | RunPod SSH proxy doesn't support file transfer; use git |
