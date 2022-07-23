#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=plot_psie_1000_psih_1000_chi_0200_gammah_8000_gammae_1000_ae_0140_ah_0135
#SBATCH --output=./job-outs/plot_psie_1000_psih_1000_chi_0200_gammah_8000_gammae_1000_ae_0140_ah_0135.out
#SBATCH --error=./job-outs/plot_psie_1000_psih_1000_chi_0200_gammah_8000_gammae_1000_ae_0140_ah_0135.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_Plots.py --model psie_1000_psih_1000_chi_0200_gammah_8000_gammae_1000_ae_0140_ah_0135

