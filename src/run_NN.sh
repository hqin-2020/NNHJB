#! /bin/bash

mkdir -p ./job-outs
mkdir -p ./bash

declare -a model_list=("psie_0500_psih_0500_chi_0500_gammah_8000_gammae_0500" "psie_1500_psih_0500_chi_0500_gammah_8000_gammae_0500" "psie_0500_psih_1500_chi_0500_gammah_8000_gammae_0500" "psie_1500_psih_1500_chi_0500_gammah_8000_gammae_0500")

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