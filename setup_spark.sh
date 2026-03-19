#!/bin/bash
# =============================================================
# DGX Spark Setup Script for Parameter Golf
# Run this on the Spark once you have SSH/monitor access
# =============================================================
set -e

echo "=== DGX Spark Parameter Golf Setup ==="

# 1. Enable SSH for remote access
echo "[1/7] Enabling SSH..."
sudo systemctl enable ssh
sudo systemctl start ssh
echo "SSH enabled. IP address:"
hostname -I

# 2. System info
echo "[2/7] System info..."
nvidia-smi || echo "nvidia-smi not found"
cat /proc/cpuinfo | grep "model name" | head -1
free -h
python3 --version

# 3. Python environment
echo "[3/7] Setting up Python environment..."
cd ~
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
echo "[4/7] Installing dependencies..."
pip install --upgrade pip
pip install numpy tqdm sentencepiece huggingface-hub datasets
pip install torch==2.10 --index-url https://download.pytorch.org/whl/cu128
pip install kernels typing-extensions==4.15.0

# 5. Download data (full 80 shards)
echo "[5/7] Downloading FineWeb dataset (80 shards)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# 6. Quick sanity check with baseline
echo "[6/7] Baseline sanity check (500 steps)..."
ITERATIONS=500 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=250 python3 train_gpt.py

# 7. Quick sanity check with our recurrent model
echo "[7/7] Recurrent model sanity check (500 steps)..."
ITERATIONS=500 VAL_LOSS_EVERY=250 python3 train_spark.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run experiments:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Default config (dim=768, 3x5=15 layers, SwiGLU, no LoRA)"
echo "  python3 train_spark.py"
echo ""
echo "  # With LoRA adapters (dim=640, rank=32)"
echo "  MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_HIDDEN=1408 LORA_RANK=32 python3 train_spark.py"
echo ""
echo "  # 4x4 config"
echo "  NUM_PHYSICAL_LAYERS=4 NUM_LOOPS=4 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_HIDDEN=1280 python3 train_spark.py"
echo ""
echo "SSH IP for remote access:"
hostname -I
