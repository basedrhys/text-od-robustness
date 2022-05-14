#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=./slurm_logs/eval-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:v100:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"

conda activate dls
cd /scratch/rc4499/dls/project/text-od-robustness
./eval_flickr.sh 10 mdetr_resnet101 v100

