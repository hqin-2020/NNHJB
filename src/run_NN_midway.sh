#! /bin/bash

nv=30
nVtilde=0
V_bar=1.0
Vtilde_bar=0.0
sigma_V_norm=0.132
sigma_Vtilde_norm=0.0

if (( $(echo "$sigma_Vtilde_norm == 0.0" |bc -l) )); then
    domain_folder='WZV'
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
elif (( $(echo "$sigma_V_norm == 0.0" |bc -l) )); then
    domain_folder='WZVtilde'
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
fi


for a_e in 0.15
do
    for a_h in 0.13 0.12 0.11 0.090 0.080 0.070 0.060
    do
        for psi_e in 1.0
        do
            for psi_h in 1.0
            do
                for gamma_e in 1.0
                do
                    for gamma_h in 1.0
                    do
                        for chiUnderline in 1.0
                        do  
                            model_folder=chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_psi_e_${psi_e}_psi_h_${psi_h}
                            mkdir -p ./job-outs/$domain_folder/$model_folder
                            mkdir -p ./bash/$domain_folder/$model_folder

                            for weight in 10.0
                            do
                                for points_size in 2
                                do
                                    for boundary in 5
                                    do
                                        for logXiE_NN_layers in 4
                                        do 
                                            for logXiH_NN_layers in 4
                                            do  
                                                for kappa_NN_layers in 4
                                                do

                                                    layer_folder=logXiE_NN_layers_${logXiE_NN_layers}_logXiH_NN_layers_${logXiH_NN_layers}_kappa_NN_layers_${kappa_NN_layers}_weight_${weight}_points_size_${points_size}_boundary_${boundary}
                                                    mkdir -p ./job-outs/$domain_folder/$model_folder/$layer_folder
                                                    mkdir -p ./bash/$domain_folder/$model_folder/$layer_folder

                                                    touch ./bash/$domain_folder/$model_folder/$layer_folder/$model_folder.sh
                                                    tee ./bash/$domain_folder/$model_folder/$layer_folder/$model_folder.sh << EOF
#!/bin/bash

#SBATCH --job-name=$model_folder
#SBATCH --output=./job-outs/$domain_folder/$model_folder/$layer_folder/$model_folder.out
#SBATCH --error=./job-outs/$domain_folder/$model_folder/$layer_folder/$model_folder.err
#SBATCH --time=0-3:00:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load python/anaconda-2020.02
module load cuda/10.1
module load tensorflow/2.1

srun python3 NN_structure.py    --logXiE_NN_layers ${logXiE_NN_layers} --logXiH_NN_layers ${logXiH_NN_layers} --kappa_NN_layers ${kappa_NN_layers}
srun python3 standard_BFGS.py   --chiUnderline ${chiUnderline} --a_e ${a_e} --a_h ${a_h} --gamma_e ${gamma_e} --gamma_h ${gamma_h} --psi_e ${psi_e} --psi_h ${psi_h} \
                                --nV $nV --nVtilde $nVtilde --V_bar $V_bar --Vtilde_bar $Vtilde_bar --sigma_V_norm $sigma_V_norm --sigma_Vtilde_norm $sigma_Vtilde_norm \
                                --logXiE_NN_layers $logXiE_NN_layers --logXiH_NN_layers $logXiH_NN_layers --kappa_NN_layers $kappa_NN_layers \
                                --weight $weight --points_size $points_size --boundary $boundary

EOF
                                                    sbatch ./bash/$domain_folder/$model_folder/$layer_folder/$model_folder.sh
                                                done        
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
