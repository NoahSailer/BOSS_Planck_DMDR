#!/bin/bash
# Job name:
#SBATCH --job-name=accuracy
# Request one node
#SBATCH --nodes=1
# Partition:
#SBATCH --partition=lr3
# QoS:
#SBATCH --qos=lr_normal
# Account:
#SBATCH --account=pc_mldarkenergy
# Wall-clock time
#SBATCH -t 1:00:00 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=nsailer@berkeley.edu
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/accuracy.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/accuracy.err

cd /global/home/users/nsailer/BOSS_Planck_DMDR/
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
#conda activate cobaya

cd /global/home/users/nsailer
yes | rm -r .julia

#module load orca

#python3 -m pip install --no-cache-dir juliacall
#echo installed juliacall

#python -c "print('here we go'); import juliacall; print('worked!')"

#python notebooks/testing_accuracy.py
