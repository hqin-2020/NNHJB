#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=plot_ae_0150_ah_0050
#SBATCH --output=./job-outs/plot_ae_0150_ah_0050.out
#SBATCH --error=./job-outs/plot_ae_0150_ah_0050.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_Plots.py --model ae_0150_ah_0050

