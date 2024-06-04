#!/bin/bash

#SBATCH -n 8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=tr_gb_50
#SBATCH --output=__tr_gb_50.out
#SBATCH --error=__tr_gb_50.err

# 'SoftMax', 'Garbage', 'EOS', 'Objectosphere', 'MultiBinary'
# 'LeNet_plus_plus', 'ResNet_18', 'ResNet_50'
python ../openset-binary/training.py -ar ResNet_50 -a Garbage -dt SmallScale -rt '/cluster/scratch/khyeongkyun/UZH-MT/data'