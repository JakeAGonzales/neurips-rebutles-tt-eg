#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=group_cvar025_sev1248
#SBATCH --output=group_cvar025_sev1248_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

# Severity-weighted prior + CVaR-over-groups (Option D + Option B).
# p(e) ∝ 2^e = [1,2,4,8]; CVaR alpha=0.25 picks the worst-loss group(s)
# under that prior. With G=4 groups this is Sagawa-style hard worst-group DRO
# but applied on top of the severity-tilted baseline rather than the empirical one.
# Compare against:
#   train_group_b0_sev1248.slurm    (same prior, no DRO tilt)
#   train_group_b5_sev1248.slurm    (same prior, soft entropic tilt)
#   train_group_cvar_a025.slurm     (CVaR with empirical prior)
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg oipo1 --lora \
    --ypp_samples 1 --risk neutral \
    --use_group_dro \
    --group_risk_fn cvar \
    --group_risk_alpha 0.25 \
    --ema_alpha 0.9 \
    --min_per_group 2 \
    --p_nominal_override 1,2,4,8
