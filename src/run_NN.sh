#! /bin/bash

mkdir -p ./job-outs
mkdir -p ./bash

declare -a model_list=("psie_1000_psih_1000_chi_0500_gammah_5000_gammae_1000_ae_0140_ah_0135" "psie_1000_psih_1000_chi_0200_gammah_5000_gammae_1000_ae_0140_ah_0135" \
             "psie_1000_psih_1000_chi_0500_gammah_5000_gammae_0500_ae_0140_ah_0135" "psie_1000_psih_1000_chi_0200_gammah_5000_gammae_0500_ae_0140_ah_0135")


for model in "${model_list[@]}"
do
    touch ./bash/run_$model.sh
    tee ./bash/run_$model.sh << EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=run_$model
#SBATCH --output=./job-outs/run_$model.out
#SBATCH --error=./job-outs/run_$model.err
#SBATCH --time=0-3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_BFGS.py --model $model
srun python3 standard_Plots.py --model $model

EOF
    sbatch ./bash/run_$model.sh
done