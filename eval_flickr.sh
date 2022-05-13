#!/bin/bash
export OUTPUT_DIR=/scratch/rc4499/dls/project/text-od-robustness/output
export IMG_DIR=/scratch/sbp354/DL_Systems/final_project/data/flickr30k-images
export MDETR_GIT_DIR=/scratch/sbp354/DL_Systems/final_project/mdetr

python ./eval_flickr.py \
  --img_dir $IMG_DIR \
  --mdetr_git_dir $MDETR_GIT_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $1 \
  --pretrained_model $2 \
  --gpu_type $3
  