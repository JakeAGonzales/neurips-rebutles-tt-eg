# TT-EG: Two-Timescale Extragradient with Risk-Aware Bias Tracking

Code to reproduce the TT-EG bias-tracking experiments (risk-aware EGPO with a
fast-timescale tracker that debiases the finite-sample plug-in risk gradient).

## Layout

- `risk_egpo/tt_eg.py` — main trainer (`TTEGTrainer`): the two-timescale tracker,
  bias surrogate, and all in-process diagnostics (window-averaged vector bias,
  empirical MC plug-in sanity).
- `risk_egpo/{trainer,loss,config,group_dro,train}.py` — risk-EGPO base
  (risk functionals, IPO loss, group-DRO, config).
- `EGPO/` — base extragradient preference-optimization library (+ `environment.yml`).
- `batch_jobs/*.slurm` — SLURM launch scripts. `tteg_k8_ab.slurm` is the headline
  A/B (base vs extrapolated bias point, K=8, entropic c=10, gamma=0.1).
- `plot_tteg_tracking.py` — turns run logs into all TT-EG diagnostic plots.
- `tests/` — unit tests / fixtures.

## Environment

Verified stack (conda env with a CUDA 12.1 GPU):

- python 3.10+, `torch==2.2.1+cu121`, `transformers==4.48.0`, `trl==0.13.0`,
  `deepspeed==0.16.2`, `peft==0.14.0`, plus `jinja2`, `matplotlib` (plotting only).

See `EGPO/environment.yml` for the base conda environment.

## Running the headline A/B

Set the data/model/experiment roots, then launch:

```bash
export MODEL_ROOT=/path/to/hf_cache/models
export DATA_ROOT=/path/to/hf_cache/data
export EXP_ROOT=/path/to/experiments

# SLURM (2-task array: task 0 = base, task 1 = extrapolated)
sbatch batch_jobs/tteg_k8_ab.slurm

# Or directly (single GPU):
deepspeed --num_gpus=1 risk_egpo/tt_eg.py \
    --alg eg --lora \
    --ypp_samples 8 \
    --risk entropic --risk_c 10.0 \
    --use_tteg --tteg_gamma 0.1 --tteg_bias_point base   # or: extrapolated
```

Model: `vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT`.
Dataset: `PKU-Alignment/PKU-SafeRLHF`.

## Plotting

```bash
python plot_tteg_tracking.py \
    --logs run_base.out run_extrapolated.out \
    --labels base extrapolated \
    --prefix tteg_k8_g0.1 \
    --outdir plots/tteg_k8_ab
```

Produces (per run / per window W in {20,50,100}):

- `*_mc_sanity.png` — empirical MC plug-in risk `rho` vs delta-method bias `b`
  and the relative bias `|b|/|rho|` (should stay small).
- `*_resid_ratio_W{W}.png` — **primary** normalized residual bias
  `rho = ||b_bar - xi|| / ||b_bar||` (below 1 means the tracker helps).
- `*_cos_W{W}.png`, `*_scale_W{W}.png` — directional alignment and scale ratio.
- `*_compare_W{W}_*.png` — uncorrected window-avg bias vs residual after TT.
- plus the original tracking scalars (`xi_norm`, `b_hat_norm`, residual, etc.).

## Key TT-EG flags (see `risk_egpo/tt_eg.py`)

- `--use_tteg` — enable the tracker.
- `--tteg_gamma` — fast-timescale EMA gain (default 0.1, constant).
- `--tteg_bias_point {base,extrapolated}` — evaluate `b_hat` at the current
  iterate (reusing the main step's activations) or at the look-ahead point.
- `--ypp_samples K` — number of y'' samples (K >= 2 required for TT-EG).
- `--risk {neutral,cvar,entropic}` with `--risk_c` / `--risk_alpha`.
