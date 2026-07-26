#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=group_b0_sev1248
#SBATCH --output=group_b0_sev1248_%j.out
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

# Severity-weighted ERM (Option D, no DRO tilt).
# p(e) ∝ 2^e = [1,2,4,8] -> normalized [0.067, 0.133, 0.267, 0.533].
# With risk_beta=0 the per-group weights are just p_nominal (no adversarial tilt),
# so this isolates the contribution of the severity-weighted prior alone.
# Compare against:
#   train_group_risk_beta0.slurm     (same but empirical p_hat)
#   train_oipo1_neutral.slurm        (no group machinery at all)
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/train.py \
    --alg oipo1 --lora \
    --ypp_samples 1 --risk neutral \
    --use_group_dro \
    --risk_beta 0.0 \
    --ema_alpha 0.9 \
    --beta_anneal_frac 0.0 \
    --min_per_group 2 \
    --p_nominal_override 1,2,4,8
