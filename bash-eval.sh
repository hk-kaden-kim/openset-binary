#!/bin/bash

#SBATCH -n 4
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=ev_sm_gb
#SBATCH --output=__ev_sm_gb.out
#SBATCH --error=__ev_sm_gb.err

# 'SoftMax', 'Garbage', 'EOS', 'Objectosphere', 'OpenMax', 'MultipleBinary'
python openset-binary/evaluation.py -a SoftMax Garbage -d '/cluster/scratch/khyeongkyun/UZH-MT/data' -p 'Evaluation_sm_gb.pdf'