#!/bin/bash
#SBATCH --job-name=tokenizer-train
#SBATCH --output=%x_%j.out
#SBATCH --comment stability
#SBATCH --partition=cpu64
#SBATCH --nodes=1

# XXX Change this to your directory.
cd /fsx/home-polm/tokenizer-train
source env/bin/activate

srun --comment stability sp-train.py $1 $2 $3
