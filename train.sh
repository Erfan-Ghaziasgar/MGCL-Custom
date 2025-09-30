#!/bin/bash

# ===================================================================================
# Training Script for Enhanced MGCL with Semantic Contrastive Learning
#
# This script configures and runs the training process.
# To run a different experiment, simply change the variables in the
# "CONFIGURATION" section below.
# ===================================================================================

# --- CONFIGURATION ---
# Define the target and source domains for the recommendation task.
TARGET_DOMAIN="Books"
SOURCE_DOMAIN="Movies_and_TV"

# Define a directory name for this specific training run.
# This helps in organizing logs, metrics, and models.
TRAIN_DIR="semantic_contrastive_v1"

# Define the root directory where all model checkpoints will be saved.
# IMPORTANT: Ensure this path is correct for your environment (e.g., Google Drive).
CHECKPOINT_ROOT="/content/drive/MyDrive/mgcl_checkpoints"

# --- DYNAMIC PATHS ---
# Automatically create a specific directory for this run's checkpoints.
CKPT_DIR="${CHECKPOINT_ROOT}/${TARGET_DOMAIN}_${TRAIN_DIR}"

# --- EXECUTION ---
echo "==================================================="
echo "Starting Training..."
echo "  Target Domain: $TARGET_DOMAIN"
echo "  Source Domain: $SOURCE_DOMAIN"
echo "  Checkpoint Dir: $CKPT_DIR"
echo "==================================================="

python main.py \
  --target_domain="$TARGET_DOMAIN" \
  --source_domain="$SOURCE_DOMAIN" \
  --train_dir="$TRAIN_DIR" \
  --maxlen=100 \
  --hidden_units=50 \
  --dropout_rate=0.5 \
  --device=cuda \
  --num_epochs=500 \
  --batch_size=128 \
  --save_every=1 \
  --resume=true \
  --ckpt_dir="$CKPT_DIR" \
  --alpha=0.5 \
  --beta=0.5 \
  --gamma=0.5 \
  --delta=0.1

