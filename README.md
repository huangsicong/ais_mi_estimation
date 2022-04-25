# Improving Mutual Information Estimation with Annealed and Energy-Based Bounds

Research code for the [ICLR2022 paper](https://openreview.net/forum?id=T0B9AoM_bFg): Improving Mutual Information Estimation with Annealed and Energy-Based Bounds 
 

## Environment Setup

```
conda create --name pt17test python=3.6.6
source activate pt17test
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install matplotlib scipy tqdm wandb
git clone git@github.com:BorealisAI/lite_tracer.git
cd lite_tracer
pip install .
```

This codebase uses wandb to log results. See https://docs.wandb.ai/quickstart for how to setup wandb. 


## Reproducing our results.

Following those steps to reproduce our results in Tables 1, 4 and 5.

- Get the checkpoints.

  The checkpoint for trained MNSIT GANs checkpoints can be found [here](https://drive.google.com/file/d/1sD2USkEVs1eIgTRXLcXgHHm43TAXBEZK/view?usp=sharing).
  Set the **FILEPATH** (it should end with .zip) and run this command to download the checkpoints zip:


  ```
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sD2USkEVs1eIgTRXLcXgHHm43TAXBEZK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sD2USkEVs1eIgTRXLcXgHHm43TAXBEZK" -O FILEPATH && rm -rf /tmp/cookies.txt
  ```
 

  And then unzip it into the **./runoutputs/checkpoints/**, where . should be the codebase folder in this repo.

  ```
  unzip FILEPATH -d ./runoutputs/checkpoints/
  ```

  The training code to train the rest of the models are included.  

- Run the scripts. 
  If you have access to a compute cluster that has slurm and sbatch installed, to reproduce our experimental results in tables 1, 4 and 5, you can run the sbatch files in [codebase/sbatch](codebase/sbatch) in order and only start the next one after the previous ones are finished. Remember to set the **dir_path** in the sbatch files to the absolute path of where you cloned this repo. You will also need to adjust the sbatch arguments for the specific compute cluster you are using. 
  ```
  sbatch FILENAME.sh
  ```
  It is important to run them in order since some experiments have dependencies with each other. If you don't have slurm, just execute all the python commends, following the same ordering.  
- Collet the results. 
  Resulsts should be automatically organized and uploaded to [here](https://wandb.ai/site). In the logged results, **u_MI** is the upper bound of the MI estimate, **u_MI_p** is the upper (plus) confidence interval of the upper bound estimation and **u_MI_m** is the lower (minus) confidence interval. Similarly, **l_MI**, **l_MI_p** and **l_MI_m** are for lower bounds of the MI estimate. If an experiment name ends with **_ub**, it means only the upper bound in this run was used, and similarly for ones ending with with **_lb**, it means only the lower bound in this run was used.  
 
 
## Citing this work
If you use our code, please consider citing the following:
``` 
@inproceedings{
brekelmans2022improving,
title={Improving Mutual Information Estimation with Annealed and Energy-Based Bounds},
author={Rob Brekelmans and Sicong Huang and Marzyeh Ghassemi and Greg Ver Steeg and Roger Baker Grosse and Alireza Makhzani},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=T0B9AoM_bFg}
}
```