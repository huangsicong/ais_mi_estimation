#!/bin/bash
# Auto-generated from hparams
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --qos=legacy
#SBATCH --account=legacy
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-8%9
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/mi_test/rd_ood/oodd'
list=(
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcvae2_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcvae10_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcvae100_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcgan2_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcgan10_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/fcgan100_mnist_1110_data.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/ncgan5_data_s01.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/ncgan10_data_s01.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/ncgan100_data_s01.p"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}


