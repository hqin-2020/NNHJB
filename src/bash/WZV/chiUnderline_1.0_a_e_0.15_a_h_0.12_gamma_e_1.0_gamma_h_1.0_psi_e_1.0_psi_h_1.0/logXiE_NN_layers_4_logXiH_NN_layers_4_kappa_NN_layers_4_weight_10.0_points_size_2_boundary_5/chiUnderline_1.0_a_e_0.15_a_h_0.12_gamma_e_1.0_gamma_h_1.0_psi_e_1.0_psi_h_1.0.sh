#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=chiUnderline_1.0_a_e_0.15_a_h_0.12_gamma_e_1.0_gamma_h_1.0_psi_e_1.0_psi_h_1.0
#SBATCH --output=./job-outs/WZV/chiUnderline_1.0_a_e_0.15_a_h_0.12_gamma_e_1.0_gamma_h_1.0_psi_e_1.0_psi_h_1.0/logXiE_NN_layers_4_logXiH_NN_layers_4_kappa_NN_layers_4_weight_10.0_points_size_2_boundary_5/chiUnderline_1.0_a_e_0.15_a_h_0.12_gamma_e_1.0_gamma_h_1.0_psi_e_1.0_psi_h_1.0.out
#SBATCH --error=./job-outs/WZV/chiUnderline_1.0_a_e_0.15_a_h_0.12_gamma_e_1.0_gamma_h_1.0_psi_e_1.0_psi_h_1.0/logXiE_NN_layers_4_logXiH_NN_layers_4_kappa_NN_layers_4_weight_10.0_points_size_2_boundary_5/chiUnderline_1.0_a_e_0.15_a_h_0.12_gamma_e_1.0_gamma_h_1.0_psi_e_1.0_psi_h_1.0.err
#SBATCH --time=0-3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G

module load cuda/11.4
module unload python
module load python/booth/3.10/3.10.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 NN_structure.py    --logXiE_NN_layers 4 --logXiH_NN_layers 4 --kappa_NN_layers 4
srun python3 standard_BFGS.py   --chiUnderline 1.0 --a_e 0.15 --a_h 0.12 --gamma_e 1.0 --gamma_h 1.0 --psi_e 1.0 --psi_h 1.0                                 --nV 30 --nVtilde 0 --V_bar 1.0 --Vtilde_bar 0.0 --sigma_V_norm 0.132 --sigma_Vtilde_norm 0.0                                 --logXiE_NN_layers 4 --logXiH_NN_layers 4 --kappa_NN_layers 4                                 --weight 10.0 --points_size 2 --boundary 5

