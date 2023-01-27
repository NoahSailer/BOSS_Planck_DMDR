#!/bin/bash
# Job name:
#SBATCH --job-name=tmp
# Request one node
#SBATCH --nodes=1
# Partition:
#SBATCH --partition=lr3
# QoS:
#SBATCH --qos=lr_normal
# Account:
#SBATCH --account=pc_mldarkenergy
# Wall-clock time
#SBATCH -t 2:00:00 
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/tmp.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/tmp.err

cd /global/home/users/nsailer/BOSS_Planck_DMDR
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
conda activate cobaya
jupyter notebook --no-browser --port=8080
