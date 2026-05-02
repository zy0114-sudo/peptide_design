#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=c_gpu_4090
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=train.log
#SBATCH --error=train.err
#SBATCH --mail-type=all
#SBATCH --mail-user=zy0114@sjtu.edu.cn
#SBATCH --exclude=dragon5,dragon6

#conda activate flow
#python train.py --config ./config/train_gvp.yaml --device cuda:1
CUDA_VISIBLE_DEVICES=0,1
python train.py --config ./config/train_gvp.yaml --device cuda:1
echo "train done!"