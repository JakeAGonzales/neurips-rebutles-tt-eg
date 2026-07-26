#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --job-name=tteg_k8_ab
#SBATCH --output=output_files/tteg_k8_ab_%A_%a.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune
#SBATCH --array=0-1

# A/B at the headline K=8 config (matches all other experiments; none use K=2).
# entropic c=10, gamma=0.1. Task 0 = base bias point, task 1 = extrapolated.
# This run also populates the NEW in-process diagnostics: window-averaged vector
# bias (resid_ratio/cos/scale for W=20,50,100) and the empirical MC plug-in
# sanity (rho_plugin, bias_payoff_delta, bias_rel).

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -e

BIAS_POINTS=(base extrapolated)
BP=${BIAS_POINTS[$SLURM_ARRAY_TASK_ID]}
# SLURM_JOB_ID is unique per array task, avoiding port clashes on shared nodes.
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

echo "[tteg_k8_ab] gamma=0.1 bias_point=$BP c=10 K=8"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point $BP
