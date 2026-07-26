#!/bin/bash
#SBATCH --account=mlopt-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=plot_for_paper
#SBATCH --output=plot_for_paper_%j.out
#SBATCH --chdir=/gscratch/mlopt/jakegonz/llm_tune

source /gscratch/mlopt/jakegonz/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

OUTDIR=${1:-/gscratch/mlopt/jakegonz/llm_tune/plots_for_paper}
ROOT=/gscratch/mlopt/jakegonz/llm_tune

# All 19 runs with paper-style labels.
# Order (within each family) is curve-on-top order: paper-cited runs last.
python plot_oipo1_compare.py \
    --prefix training \
    --xclip 4680 \
    --outdir "$OUTDIR" \
    --run "EGPO:$ROOT/oipo1_vanilla_34839616.out" \
    --run "OMD (K=8):$ROOT/oipo1_neutral_K8_34871534.out" \
    --run "OMD-Ent (τ=1):$ROOT/oipo1_ypp_risk_34840035.out" \
    --run "OMD-Ent (τ=2):$ROOT/oipo1_ent_c2_34843630.out" \
    --run "OMD-Ent (τ=5):$ROOT/oipo1_ent_c5_34843631.out" \
    --run "OMD-Ent (τ=10):$ROOT/oipo1_ent_c10_34843634.out" \
    --run "OMD-CVaR (α=0.50):$ROOT/oipo1_cvar_a050_34843635.out" \
    --run "OMD-CVaR (α=0.25):$ROOT/oipo1_cvar_a025_34843637.out" \
    --run "OMD-CVaR (α=0.125):$ROOT/oipo1_cvar_a0125_34843638.out" \
    --run "EG (K=1):$ROOT/eg_neutral_K1_34910374.out" \
    --run "EG (K=8):$ROOT/eg_neutral_K8_34910376.out" \
    --run "EG-Ent (τ=5):$ROOT/eg_ypp_risk_34847719.out" \
    --run "EG-CVaR (α=0.125):$ROOT/eg_cvar_a0125_34910398.out" \
    --run "Nash-MD:$ROOT/nmd_vanilla_34905151.out" \
    --run "gDRO:$ROOT/group_risk_egpo_beta0_34831175.out" \
    --run "gDRO-Ent (β=1):$ROOT/group_risk_egpo_34815099.out" \
    --run "gDRO-CVaR (α=0.25):$ROOT/group_cvar_a025_34911692.out"
