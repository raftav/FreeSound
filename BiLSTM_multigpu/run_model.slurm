#!/bin/bash

#SBATCH --job-name=rnn
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:4

# EXPERIMENT NUMBER
exp_num=$1

# learning rate
lr=0.001

# batch size
bs=125

#num hidden layers
nl=5

#num hidden neuros
nn=150

#########################
#########################
module load anaconda/3 cuda/9.0 cudnn/7.0

source activate tf1.7_py3.5

python -u lstm_train.py $exp_num $lr $bs $nl $nn
