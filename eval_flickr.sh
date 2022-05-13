#!/bin/bash
export OUTPUT_DIR=/scratch/rc4499/dls/project/text-od-robustness/output
# export BATCH_SIZE=1
# export PRETRAINED_MODEL=mdetr_resnet101


python ./eval_flickr.py \
  --img_dir /scratch/sbp354/DL_Systems/final_project/data/flickr30k-images \
  --annotations_dir /scratch/rc4499/dls/project/text-od-robustness/data \
  --mdetr_git_dir /scratch/sbp354/DL_Systems/final_project/mdetr \
  --output_dir $OUTPUT_DIR \
  --batch_size $1 \
  --pretrained_model $2 \
  --gpu_type $3
  