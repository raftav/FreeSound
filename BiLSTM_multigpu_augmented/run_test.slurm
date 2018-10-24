#!/bin/bash

#SBATCH --job-name=test
#SBATCH --error=test-%j.err
#SBATCH --output=test-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000
#SBATCH --partition=longrun
#SBATCH --gres=gpu:0

# EXPERIMENT NUMBER
exp_num=1

# restore epoch
re=11

module load anaconda/3 cuda/9.0 cudnn/7.0

source activate tf1.7_py3.5

python -u lstm_test.py $exp_num $re
