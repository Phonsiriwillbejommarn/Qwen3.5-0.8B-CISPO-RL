#!/bin/bash
# ─────────────────────────────────────────────────────────────
# CISPO RL Training Script
# Policy : Phonsiri/Qwen3.5-0.8B-Distillation-Phase2
# Judge  : Qwen/Qwen3.5-9B
# Dataset: nohurry/Opus-4.6-Reasoning-3000x-filtered
# GPU    : H100 (SDPA, BF16, TF32)
# ─────────────────────────────────────────────────────────────
set -e

# ─── Environment ────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# SDPA (ไม่ใช้ Flash Attention)
export TORCH_SDPA_ENABLE_FLASH_ATTENTION=0

# TF32 สำหรับ H100
export CUDA_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1

# HuggingFace cache
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"

echo "════════════════════════════════════════════════════════"
echo "  CISPO RL Training — Qwen3.5-0.8B (8k Context)"
echo "  $(date)"
echo "════════════════════════════════════════════════════════"

# ─── Pre-flight GPU check ────────────────────────────────────
echo "[GPU] Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
echo ""

# ─── Args (override ผ่าน env var) ───────────────────────────
CONFIG="${CONFIG:-configs/cispo_config.yaml}"
MAX_STEPS="${MAX_STEPS:-1000}"
RUN_NAME="${RUN_NAME:-cispo-qwen3.5-0.8b-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/cispo_run_002}"
LR="${LR:-5e-7}"

echo "[Config] $CONFIG"
echo "[Steps ] $MAX_STEPS"
echo "[Run   ] $RUN_NAME"
echo ""

# ─── Run ─────────────────────────────────────────────────────
python train_cispo.py \
    --config "$CONFIG" \
    --max_steps "$MAX_STEPS" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Output: $OUTPUT_DIR/final_model"
echo "  $(date)"
echo "════════════════════════════════════════════════════════"
