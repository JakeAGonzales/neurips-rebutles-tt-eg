#!/bin/bash
#SBATCH --account=ctrl
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=pref_judge_sanity
#SBATCH --output=pref_judge_sanity_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

python tests/test_pref_judge.py
