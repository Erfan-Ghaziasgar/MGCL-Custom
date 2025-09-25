#!/bin/bash
python main.py \
  --target_domain=Sports_and_Outdoors \
  --source_domain=Clothing_Shoes_and_Jewelry \
  --maxlen=20 \
  --hidden_units=50 \
  --dropout_rate=0.5 \
  --device=cuda \
  --num_epochs=500 \
  --batch_size=256 \
  --save_every=50 \
  --resume true \
  --ckpt_dir=/content/drive/MyDrive/mgcl_checkpoints/Sports_and_Outdoors_default
