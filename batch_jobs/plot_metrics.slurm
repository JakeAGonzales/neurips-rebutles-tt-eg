#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=plot_metrics
#SBATCH --output=plot_metrics_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

LOG_FILE=${1:-/gscratch/mlopt/jakegonz/llm_tune/group_risk_egpo_34815099.out}
OUTDIR=${2:-/gscratch/mlopt/jakegonz/llm_tune/plots}
PREFIX=${3:-group_risk}

python plot_training_metrics.py "$LOG_FILE" --outdir "$OUTDIR" --prefix "$PREFIX"
