#!/bin/bash
#SBATCH --account=mlopt
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --job-name=tteg_ab_xt
#SBATCH --output=output_files/tteg_ab_xt_%A_%a.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune
#SBATCH --array=0-2

# A/B leg B: bias point = extrapolated (b_hat at theta_tilde via a fresh forward
# before the rollback; textbook EG oracle at the extrapolated iterate). Sweep the
# tracker gain gamma in {0.03, 0.1, 0.3} (one array task each), on the headline
# entropic risk c=10, K=8, Axis-1 only. This costs an extra forward per minibatch
# on the correction step vs tteg_ab_base.slurm. Compare tteg/tracking_residual_sq
# and downstream OOD metrics against the base leg to decide which oracle to keep.
# Run names carry a `-xt` tag so this does NOT collide with the base leg's output.

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate nlhf   # only env with deepspeed + full trl/torch stack
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

export MODEL_ROOT=/gscratch/mlopt/jakegonz/hf_cache/models
export DATA_ROOT=/gscratch/mlopt/jakegonz/hf_cache/data
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

module load cuda
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

set -e

GAMMAS=(0.03 0.1 0.3)
GAMMA=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
# SLURM_JOB_ID is unique per array task (SLURM_ARRAY_JOB_ID is shared), so this
# avoids port clashes when tasks co-locate on one node.
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

echo "[tteg_ab_xt] gamma=$GAMMA bias_point=extrapolated c=10 K=8"
deepspeed --num_gpus=1 --master_port=$MASTER_PORT risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma $GAMMA --tteg_bias_point extrapolated
