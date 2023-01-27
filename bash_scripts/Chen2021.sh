#!/bin/bash
# Job name:
#SBATCH --job-name=Chen2021
# Request one node
#SBATCH --nodes=2
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
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/Chen2021.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/Chen2021.err

cd /global/home/users/nsailer/BOSS_Planck_DMDR/
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
conda activate cobaya
module load orca
rm chains/Chen2021/Chen2021.input.yaml.locked
srun -N 2 -n 8 -c 4 cobaya-run -r yamls/Chen2021.yaml
