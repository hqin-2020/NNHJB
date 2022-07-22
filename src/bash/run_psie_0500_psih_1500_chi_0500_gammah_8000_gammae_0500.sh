#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=run_psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500
#SBATCH --output=./job-outs/run_psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500.out
#SBATCH --error=./job-outs/run_psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500.err
#SBATCH --time=0-2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_BFGS.py --model psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500
srun python3 standard_Plots.py --model psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500

