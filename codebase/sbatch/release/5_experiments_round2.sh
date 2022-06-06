#!/bin/bash
# Auto-generated from hparams
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --qos=legacy
#SBATCH --account=legacy
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/mi_test/rd_ood/oodd'
list=(
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune2_ncgan100_iwae_s01_rfiwae_k_1000000_qiter_1000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune2_ncgan100_iwae_s01_rfiwae_k_1_qiter_1000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}




