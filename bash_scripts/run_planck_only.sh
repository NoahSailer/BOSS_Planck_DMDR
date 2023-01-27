#!/bin/bash
# Job name:
#SBATCH --job-name=run_DMDR_chains_n4
# Request one node
#SBATCH --nodes=1
# Partition:
#SBATCH --partition=lr3
# QoS:
#SBATCH --qos=lr_normal
# Account:
#SBATCH --account=pc_mldarkenergy
# Wall-clock time
#SBATCH -t 48:00:00 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=nsailer@berkeley.edu
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/planck_run_DMDR_chains_n4.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/planck_run_DMDR_chains_n4.err

cd /global/home/users/nsailer/BOSS_Planck_DMDR/
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
conda activate cobaya
#module load gcc openmpi
#srun -N 1 -n 4 -c 2 
#rm chains/planck_only_n4/planck.input.yaml.locked
cobaya-run yamls/planck_only_n4.yaml
