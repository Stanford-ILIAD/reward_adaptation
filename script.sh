#!/bin/bash
#SBATCH --partition=iliad --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --gres=gpu:1 
#SBATCH --job-name="spbr7"
#SBATCH --output=/iliad/u/minae/reward_adaptation/jobs/%x.o

##CUDA_VISIBLE_DEVICES=0 python train.py --env nav1 --expt_type finetune --bs 7
CUDA_VISIBLE_DEVICES=0 python train.py --env nav1_sparse --bs 7 --experiment_dir output/sparse --expt_type ours
###CUDA_VISIBLE_DEVICES=0 python baselines/PNN/train.py --bs 7
###CUDA_VISIBLE_DEVICES=0 python baselines/L2SP/train.py --bs 5
###CUDA_VISIBLE_DEVICES=0 python baselines/BSS/train.py --bs 7

echo "done"
