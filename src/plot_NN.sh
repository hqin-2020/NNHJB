#! /bin/bash

mkdir -p ./job-outs
mkdir -p ./bash

declare -a model_list=("psie_1000_psih_1000_chi_0500_gammah_8000_gammae_1000_ae_0140_ah_0135" "psie_1000_psih_1000_chi_0200_gammah_8000_gammae_1000_ae_0140_ah_0135" \
              "psie_1000_psih_1000_chi_0500_gammah_8000_gammae_0500_ae_0140_ah_0135" "psie_1000_psih_1000_chi_0200_gammah_8000_gammae_0500_ae_0140_ah_0135" \
              "psie_0500_psih_1500_chi_1000_gammah_5000_gammae_1000" "psie_1500_psih_0500_chi_1000_gammah_5000_gammae_1000" \
             "psie_1000_psih_1000_chi_1000_gammah_3000_gammae_1000_ae_0150_ah_0100" "psie_1000_psih_1000_chi_1000_gammah_3000_gammae_1000_ae_0150_ah_0050" \
             "psie_1000_psih_1000_chi_1000_gammah_3000_gammae_0500_ae_0150_ah_0100" "psie_1000_psih_1000_chi_1000_gammah_3000_gammae_0500_ae_0150_ah_0050" \
             "psie_1000_psih_1000_chi_0500_gammah_3000_gammae_1000_ae_0150_ah_0100" "psie_1000_psih_1000_chi_0500_gammah_3000_gammae_1000_ae_0150_ah_0050" \
             "psie_1000_psih_1000_chi_0500_gammah_3000_gammae_0500_ae_0150_ah_0100" "psie_1000_psih_1000_chi_0500_gammah_3000_gammae_0500_ae_0150_ah_0050" \
             "psie_1000_psih_1000_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0100" "psie_1000_psih_1000_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0050" \
             "psie_1000_psih_1000_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0100" "psie_1000_psih_1000_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0050" \
             "gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0100" "gammah_8000_gammae_0500_chi_0500_ae_0150_ah_0050" \
             "psie_0500_psih_0500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0050" "psie_0500_psih_1500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0050" \
             "psie_1500_psih_0500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0050" "psie_1500_psih_1500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0050" \
             "psie_0500_psih_0500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0100" "psie_0500_psih_1500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0100" \
             "psie_1500_psih_0500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0100" "psie_1500_psih_1500_chi_1000_gammah_8000_gammae_1000_ae_0150_ah_0100" \
             "psie_0500_psih_0500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0050" "psie_0500_psih_1500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0050" \
             "psie_1500_psih_0500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0050" "psie_1500_psih_1500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0050" \
             "psie_0500_psih_0500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0100" "psie_0500_psih_1500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0100" \
             "psie_1500_psih_0500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0100" "psie_1500_psih_1500_chi_1000_gammah_8000_gammae_0500_ae_0150_ah_0100")

for model in "${model_list[@]}"
do
    touch ./bash/plot_$model.sh
    tee ./bash/plot_$model.sh << EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=plot_$model
#SBATCH --output=./job-outs/plot_$model.out
#SBATCH --error=./job-outs/plot_$model.err
#SBATCH --time=0-1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_Plots.py --model $model

EOF
    sbatch ./bash/plot_$model.sh
done