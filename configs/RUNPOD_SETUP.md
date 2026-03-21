# RunPod 8xH100 Setup Guide

## Prerequisites
- RunPod account with 8xH100 SXM pod
- SSH key added to RunPod (id_ed25519)
- VPN connected (home network blocks port 22)

## Step 1: SSH into the pod

```bash
ssh <POD_ID>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Step 2: Clone the repo and install deps

```bash
git clone https://github.com/8tp/parameter-golf.git
cd /parameter-golf
pip install sentencepiece huggingface-hub datasets tqdm
```

## Step 3: Download the dataset

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This downloads:
- Tokenizer to `./data/tokenizers/fineweb_1024_bpe.{model,vocab}`
- Training shards (80 x 100M tokens) to `./data/datasets/fineweb10B_sp1024/`
- Validation shard to same directory

Note: tokenizer is also in the repo already, but the training/validation
data (~324MB) must be downloaded via this script.

## Step 4: Run the 8xH100 submission

```bash
bash configs/runpod_8xh100.sh
```

This runs `torchrun --nproc_per_node=8 train_spark.py` with all env vars set.

## Quick copy-paste (all steps after SSH):

```bash
git clone https://github.com/8tp/parameter-golf.git && cd /parameter-golf
pip install sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024
bash configs/runpod_8xh100.sh
```

## What to expect

- ~500ms/step on 1xH100, expect ~60-70ms/step on 8xH100 with DDP
- ~12,500 steps in 10 minutes
- Auto-stops at 580s (20s buffer for final eval + SWA)
- Final output: val_bpb score, quantized model size, submission size

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH times out | Connect VPN first, verify with `nc -zv ssh.runpod.io 22` |
| `No module 'sentencepiece'` | `pip install sentencepiece` |
| `No files found for pattern` | Run the dataset download script (Step 3) |
| `enable_gqa` TypeError | Already fixed — make sure you pulled latest |
| SCP/rsync fails | RunPod SSH proxy doesn't support file transfer; use git |
| NCCL timeout | Check all 8 GPUs visible: `nvidia-smi -L` |
