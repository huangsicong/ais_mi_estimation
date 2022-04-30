#!/bin/bash
# Auto-generated from hparams
#SBATCH --partition=p100
#SBATCH --gres=gpu:1
#SBATCH --qos=legacy
#SBATCH --account=legacy
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --array=0-23%24
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
dir_path='/h/huang/git_code/mi_test/rd_ood/oodd'
list=(
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/CAIS_G100_symmetric_ncgan100_ais_s01_1104_a_100000_rad_8_q_SKL_Ct_a500_ncgan100_ais_s01_1104_a_500_vpsym_qiter_1000000_rad_8_lr_0_0005.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/CAIS_G10_symmetric_ncgan10_ais_s01_1104_a_100000_rad_9_q_SKL_Ct_a500_gan10_ncgan10_ais_s01_1104_a_500_vpsym_qiter_20000_rad_8_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/CAIS_G5_symmetric_ncgan5_ais_s01_1104_a_100000_rad_4_q_SKL_Ct_a500_gan5_ncgan5_ais_s01_1104_a_500_vpsym_qiter_1000000_rad_10_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G100_0209_legacy_fcgan100_mnist_1110_a_1_rad_4_q_MIWAE_tune3_fcgan100_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G100_0209_legacy_fcgan100_mnist_1110_a_30000_rad_8_q_MIWAE_tune3_fcgan100_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G100_0209_legacy_fcgan100_mnist_1110_a_500_rad_3_q_MIWAE_rftune4_fcgan100_mnist_rfiwae_k_1000_qiter_1000000_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G10_0209_fcgan10_mnist_1110_a_1_rad_8_q_MIWAE_tune3_fcgan10_mnist_1110_reverse_k_1000000_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G10_0209_fcgan10_mnist_1110_a_30000_rad_10_q_MIWAE_tune3_fcgan10_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G10_0209_fcgan10_mnist_1110_a_500_rad_3_q_MIWAE_rftune3_fcgan10_mnist_rfiwae_k_1_qiter_1000000_lr_1e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G2_0209_fcgan2_mnist_1110_a_1_rad_8_q_MIWAE_rftune4_fcgan2_mnist_rfiwae_k_1_qiter_1000000_lr_1e_06.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G2_0209_fcgan2_mnist_1110_a_30000_rad_15_q_MIWAE_tune3_fcgan2_mnist_1110_reverse_k_1000000_qiter_1000000_lr_0_001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_G2_0209_fcgan2_mnist_1110_a_500_rad_3_q_MIWAE_tune3_fcgan2_mnist_1110_reverse_k_1000000_qiter_1000000_lr_0_001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V100_0209_fcvae100_mnist_1110_a_1_rad_15_q_MIWAE_tune3_fcvae100_mnist_1110_reverse_k_1000_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V100_0209_fcvae100_mnist_1110_a_30000_rad_9_q_MIWAE_tune3_fcvae100_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V100_0209_fcvae100_mnist_1110_a_500_rad_1_q_MIWAE_tune3_fcvae100_mnist_1110_reverse_k_1000_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V10_0209_fcvae10_mnist_1110_a_1_rad_12_q_MIWAE_tune3_fcvae10_mnist_1110_reverse_k_1000000_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V10_0209_fcvae10_mnist_1110_a_30000_rad_3_q_MIWAE_tune3_fcvae10_mnist_1110_reverse_k_1000000_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V10_0209_fcvae10_mnist_1110_a_500_rad_2_q_MIWAE_rftune3_fcvae10_mnist_rfiwae_k_1000000_qiter_1000000_lr_5e_06.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V2_0210_fcvae2_mnist_1110_a_1_rad_4_q_MIWAE_tune3_fcvae2_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V2_0210_fcvae2_mnist_1110_a_30000_rad_8_q_MIWAE_tune3_fcvae2_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/MAIS_V2_0210_fcvae2_mnist_1110_a_500_rad_11_q_MIWAE_tune3_fcvae2_mnist_1110_reverse_k_1_qiter_1000000_lr_0_0001.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune2_ncgan100_iwae_s01_rfiwae_k_1000000_qiter_1000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune2_ncgan100_iwae_s01_rfiwae_k_1_qiter_1000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
"cd ${dir_path} ; python -m codebase.mi --hparam_from_pickle ./codebase/hparams/mi_release/loadq_gan5_0206a_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05_q_Ciwae_tune4_ncgan100_iwae_s01_rfiwae_k_1000_qiter_5000000_lr_5e_05.p"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}




