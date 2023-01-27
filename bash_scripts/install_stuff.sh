#!/bin/bash
# Job name:
#SBATCH --job-name=install_mpi4py
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
#SBATCH -o /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/installing.out
#SBATCH -e /global/home/users/nsailer/BOSS_Planck_DMDR/chains/log/installing.err

#cd /global/home/users/nsailer/BOSS_Planck_DMDR
#source /global/home/users/nsailer/miniconda3/etc/profile.d/conda.sh
#conda activate /global/home/users/nsailer/miniconda3
source /global/home/users/nsailer/anaconda3/etc/profile.d/conda.sh
conda activate cobaya
#conda env export > environment.yml

#module load orca/5.0.3-shared-openmpi411
module load orca
#module load orca/5.0.3-shared-openmpi411
#`module load python/3.9.12
#pip install --user mpi4py

#conda remove --force -y mpi
#conda remove --force -y mpich


#pip uninstall -y mpi4py

#env MPICC=/global/software/sl-7.x86_64/modules/gcc/12.1.0/openmpi/4.1.4/bin/mpicc pip install --user mpi4py
#pip3 install --user mpi4py --no-cache-dir

#conda install -c conda-forge mpi4py
#conda install -c conda-forge 'mpi4py>=3.1.0' 
#echo Installed it!

echo -------------------------------------------------------------
echo testing installation
mpirun -n 6 python -c "from mpi4py import MPI, __version__; print(MPI.COMM_WORLD.Get_rank())"

#echo -------------------------------------------------------------
#echo uninstalling mpi4py
#pip uninstall -y mpi4py
#conda remove -y mpi4py
