#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --sockets-per-node=1
#SBATCH --time=00:02:00
##SBATCH --mem=0
##SBATCH -p build
#SBATCH -p debug
module load nvhpc
module load gcc
module list


#======START=====

echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_NNODES nodes"
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NPROCS tasks is used"
export OMP_NUM_THREADS=20

ulimit -s unlimited
nodes=1
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
#/srun --gpu-bind=closest --cpu-bind=rank ./xthi > out_affinity
srun -n 1 -c 20 nvprof /home/peyberne/Codes/Jacobi_opt/Jacobi_offload


