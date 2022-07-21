#! /bin/bash

mkdir -p ./job-outs
mkdir -p ./bash

declare -a model_list=("chi_0800" "chi_0500" "chi_0200" \
                        "gammah_8000_gammae_1000" "gammah_5000_gammae_1000" "gammah_3000_gammae_1000" "gammah_8000_gammae_0500" "gammah_1000_gammae_0500" \
                        "psie_0500_psih_0500" "psie_0500_psih_1500" "psie_1500_psih_0500" "psie_1500_psih_1500" \
                        "psie_0500_psih_0500_chi_1000_gammah_5000_gammae_1000" "psie_1500_psih_1500_chi_1000_gammah_5000_gammae_1000" \
                        "psie_0500_psih_0500_chi_1000_gammah_8000_gammae_0500" "psie_1500_psih_1500_chi_1000_gammah_8000_gammae_0500" \
                        "psie_0500_psih_0500_chi_0500_gammah_8000_gammae_0500" "psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500" \
                        "psie_1500_psih_0500_chi_0500_gammah_8000_gammae_0500" "psie_1500_psih_1500_chi_0500_gammah_8000_gammae_0500")

for model in "${model_list[@]}"
do
    touch ./bash/run_$model.sh
    tee ./bash/run_$model.sh << EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=run_$model
#SBATCH --output=./job-outs/run_$model.out
#SBATCH --error=./job-outs/run_$model.err
#SBATCH --time=0-2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5
module load cuda/11.4

source ~/venv/tensorflow-gpu/bin/activate

srun python3 standard_BFGS.py --model $model

EOF
    sbatch ./bash/run_$model.sh
done