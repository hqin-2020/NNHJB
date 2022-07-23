#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=plot_gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0100
#SBATCH --output=./job-outs/plot_gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0100.out
#SBATCH --error=./job-outs/plot_gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0100.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_Plots.py --model gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0100
