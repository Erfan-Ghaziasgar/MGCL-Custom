#!/bin/bash
set -euo pipefail

# stream logs line-by-line (prevents buffering through pipes)
export PYTHONUNBUFFERED=1

# ===================================================================================
# Training Script for MGCL with Integrated Metadata Description Encoder
# ===================================================================================

TARGET_DOMAIN="Books"
SOURCE_DOMAIN="Movies_and_TV"
TRAIN_DIR="metadata_semantic_v2"

CHECKPOINT_ROOT="/content/drive/MyDrive/mgcl_checkpoints"
CKPT_DIR="${CHECKPOINT_ROOT}/${TARGET_DOMAIN}_${TRAIN_DIR}"

# --- HYPERPARAMETERS ---
MAXLEN=100
HIDDEN_UNITS=50
NUM_BLOCKS=2
NUM_HEADS=1
DROPOUT_RATE=0.5
NUM_EPOCHS=10
BATCH_SIZE=128
LR=0.001
L2_EMB=0.001
ALPHA=0.5
BETA=0.5
GAMMA=0.5
DELTA=0.2
SAVE_EVERY=5
NUM_WORKERS=2   # if you still see stalls, try 1 or 0

echo "==================================================="
echo "Training MGCL with Metadata Semantic Integration"
echo "==================================================="
echo "  Target Domain:    $TARGET_DOMAIN"
echo "  Source Domain:    $SOURCE_DOMAIN"
echo "  Checkpoint Dir:   $CKPT_DIR"
echo "  Semantic Weight:  $DELTA (temp=0.1)"
echo "==================================================="

mkdir -p "$CKPT_DIR"

# The magic: stdbuf + -u for unbuffered Python and line-buffered pipes
# - stdbuf -oL -eL : line-buffer stdout and stderr
# - python -u       : unbuffered Python runtime
stdbuf -oL -eL python -u main.py \
  --target_domain="$TARGET_DOMAIN" \
  --source_domain="$SOURCE_DOMAIN" \
  --train_dir="$TRAIN_DIR" \
  --maxlen=$MAXLEN \
  --hidden_units=$HIDDEN_UNITS \
  --num_blocks=$NUM_BLOCKS \
  --num_heads=$NUM_HEADS \
  --dropout_rate=$DROPOUT_RATE \
  --device=cuda \
  --num_epochs=$NUM_EPOCHS \
  --batch_size=$BATCH_SIZE \
  --lr=$LR \
  --l2_emb=$L2_EMB \
  --save_every=$SAVE_EVERY \
  --num_workers=$NUM_WORKERS \
  --resume=false \
  --ckpt_dir="$CKPT_DIR" \
  --alpha=$ALPHA \
  --beta=$BETA \
  --gamma=$GAMMA \
  --delta=$DELTA \
  --temp=0.1
