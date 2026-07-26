#!/bin/bash
#SBATCH --account=mlopt-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=hf_transfer
#SBATCH --output=hf_transfer_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Pass-through args:
#   sbatch transfer_to_ratlab.slurm --dry_run
#   sbatch transfer_to_ratlab.slurm
#   sbatch transfer_to_ratlab.slurm --only risk-ipo-K8-neutral
python transfer_to_ratlab.py "$@"
