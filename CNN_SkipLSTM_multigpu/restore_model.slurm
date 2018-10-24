#!/bin/bash

#SBATCH --job-name=cnn_rnn_re
#SBATCH --error=training-%j.err
#SBATCH --output=training-%j.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=20000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:4

# EXPERIMENT NUMBER
exp_num=$1

#########################
#########################

module load anaconda/3 cuda/9.0 cudnn/7.0

source activate tf1.7_py3.5

python -u cnn_skiplstm_restore_train.py $exp_num
