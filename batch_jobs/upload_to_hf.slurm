#!/bin/bash
#SBATCH --account=mlopt-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=hf_upload
#SBATCH --output=hf_upload_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# HF auth: token must be set up beforehand via `huggingface-cli login`
# (or export HF_TOKEN). Login state is persisted in ~/.cache/huggingface.
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# Pass-through args to upload_to_hf.py:
#   sbatch upload_to_hf.slurm                    # upload everything
#   sbatch upload_to_hf.slurm --dry_run          # preview only
#   sbatch upload_to_hf.slurm --only vanilla,cvar-a0.5
#   sbatch upload_to_hf.slurm --skip_existing
export EXP_ROOT=/gscratch/mlopt/jakegonz/llm_tune/experiments

# Push directly to rat-lab/jg-<name> (no two-step transfer needed). To go to
# the personal jakegonz/* org instead, override at sbatch time, e.g.:
#   sbatch upload_to_hf.slurm --only group-dro-cvar-a0.25
python upload_to_hf.py --org rat-lab --prefix jg- "$@"
