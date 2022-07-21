#! /bin/bash

mkdir -p ./job-outs
mkdir -p ./bash

declare -a model_list=("ae_0150_ah_0100" "ae_0150_ah_0050")

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