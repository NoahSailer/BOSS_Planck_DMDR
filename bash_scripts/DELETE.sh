#!/bin/bash
# Job name:
#SBATCH --job-name=del
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
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/del.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/del.err

cd /global/home/users/nsailer
#source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
#conda activate cobaya
#module load orca

#python3 -m pip install --no-cache-dir juliacall
#echo installed juliacall

#python -c "import juliacall"

rm -r miniconda3

#python notebooks/testing_accuracy.py
