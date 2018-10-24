#!/bin/bash

#SBATCH --job-name=Bilstm
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:1

# EXPERIMENT NUMBER
exp_num=1

# learning rate
lr=0.001

# batch size
bs=50

# optimizer
opt=adam

#########################
#########################
module load anaconda/3 cuda/9.0 cudnn/7.0

source activate tf1.7_py3.5

python -u cnn_lstm_train.py $exp_num $lr $bs $opt
