#!/bin/bash
#SBATCH --partition=iliad --qos=normal
#SBATCH --time=7-00
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --output=run1.out
python train.py 
echo "done"
