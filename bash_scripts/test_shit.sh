#!/bin/bash
# Job name:
#SBATCH --job-name=test_shit
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
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=nsailer@berkeley.edu
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/testing.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/testing.err

cd /global/home/users/nsailer/BOSS_Planck_DMDR/
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
conda activate cobaya
module load orca/5.0.3-shared-openmpi411
echo Made it to installing mpi4py
conda install -c conda-forge 'mpi4py>=3.1.0' 
echo Installed it!
