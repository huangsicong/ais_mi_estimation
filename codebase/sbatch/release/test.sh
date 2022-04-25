#!/bin/bash
# Auto-generated from hparams
#SBATCH --partition=p100,t4v1
#SBATCH --gres=gpu:1
#SBATCH --qos=legacy
#SBATCH --account=legacy
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-14%15
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/mutual_oodd/rd_ood/oodd'
list=(
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release_saved_0323/MAIS_1110_fcvae2_mnist_1110_a_1_prior.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release_saved_0323/mp_tune_k1kl30_fcvae100_mnist_1110_prior_rad_3_a_500_nc_1000_nc_30.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release_saved_0323/mp_tune_k1kl30_fcvae100_mnist_1110_prior_rad_9_a_500_nc_1000_nc_30.p"

)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
