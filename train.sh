#!/bin/bash
python3 -u train_lseg.py --dataset own --data_path ./datasets --batch_size 8 --exp_name own \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 250 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384
