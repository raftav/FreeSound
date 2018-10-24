#!/bin/bash


#SBATCH --job-name=cnn-rnn
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:4

# EXPERIMENT NUMBER
exp_num=1

# learning rate
lr=0.0001

# batch size
bs=80

# num layers
nl=5

# neurons per layer
nn=150

# lambda
lb=0.0001

#########################
#########################
module load anaconda/3 cuda/9.0 cudnn/7.0

source activate tf1.7_py3.5

python -u cnn_skiplstm_train.py $exp_num $lr $bs $nl $nn $lb
