# Training logs + plot reproduction

Raw SLURM `.out` logs for the TT-EG runs and the EG/OMD baselines they are
compared against. Every log contains the per-step metric dicts the HF Trainer
prints (`{'loss': ..., 'kl': ..., 'prefs/...', 'rewards/...', 'tteg/...'}`), which
is the **only** data the plotting scripts need — no reruns or extra logging.

The three plotting scripts live at the repo root:
`plot_training_metrics.py`, `plot_tteg_tracking.py`, `plot_oipo1_compare.py`.

## Files

All TT-EG runs are `--alg eg --ypp_samples 8 --risk entropic --risk_c 10.0`
(K=8, entropic c=10), differing only in the two-timescale (TT) settings. They
were trained with the `nlhf` env (`risk_egpo/tt_eg.py`).

### `tt_eg/` — TT-EG runs

| file | TT setting | last epoch | notes |
|------|-----------|-----------:|-------|
| `eg_c10_noTT_control.out`          | **no TT** (control) | 0.9  | running; isolates the TT correction at c=10 |
| `eg_c10_TT_g0.03_base.out`         | TT γ=0.03, base          | 10.15 | **most-progressed base**; uploaded to HF |
| `eg_c10_TT_g0.1_base.out`          | TT γ=0.1, base           | 6.1   | |
| `eg_c10_TT_g0.3_base.out`          | TT γ=0.3, base           | 0.81  | |
| `eg_c10_TT_g0.1_extrapolated.out`  | TT γ=0.1, extrapolated   | 9.15  | **most-progressed xt**; uploaded to HF |
| `eg_c10_TT_g0.3_extrapolated.out`  | TT γ=0.3, extrapolated   | 2.88  | |

`tt_eg/other_attempts/` holds shorter re-runs of the same configs plus one
crashed γ=0.03 extrapolated attempt (kept for completeness; not needed for plots).

### `baselines/` — EG / OMD baselines (from the paper; `myenv`, `risk_egpo/train.py`)

| file | config | last epoch |
|------|--------|-----------:|
| `eg_entropic_c5.out`        | EG, entropic **c=5**, K=8  | 15.57 |
| `eg_neutral_K8.out`         | EG, neutral, K=8           | 15.49 |
| `eg_neutral_K1.out`         | EG, neutral, K=1           | 19.97 |
| `eg_cvar_a0.25.out`         | EG, CVaR α=0.25, K=8       | 4.29  |
| `eg_cvar_a0.125.out`        | EG, CVaR α=0.125, K=8      | 12.27 |
| `omd_oipo1_entropic_c1.out` | OMD (oipo1), entropic c=1, K=8 | 9.98 |
| `omd_oipo1_entropic_c10.out` | OMD (oipo1), entropic c=10, K=8 | 9.99 |

> Note: there is no EG entropic **c=10** baseline in the paper set — that gap is
> exactly what `eg_c10_noTT_control.out` fills.

## Reproducing the plots

Run from the repo root. Plots are written to `--outdir` (created if missing).

### 1. Training dynamics (KL, loss, accuracy, reward margins) — `plot_training_metrics.py`

Single log:

```bash
python plot_training_metrics.py logs/tt_eg/eg_c10_TT_g0.03_base.out \
    --outdir plots --prefix tteg_g0.03_base
```

Overlay several runs (e.g. TT-EG vs no-TT control vs EG baseline):

```bash
python plot_training_metrics.py \
    --logs logs/tt_eg/eg_c10_TT_g0.1_base.out \
           logs/tt_eg/eg_c10_noTT_control.out \
           logs/baselines/eg_entropic_c5.out \
    --labels "TT γ=0.1 (c10)" "no-TT (c10)" "EG baseline (c5)" \
    --outdir plots --prefix compare_c10
```

### 2. TT bias-tracking diagnostics ("the bias stuff") — `plot_tteg_tracking.py`

Shows the online tracker `||xi||` following the delta-method bias `||b_hat||`,
the tracking residual, correction size, and the debiased-vs-plug-in estimator.
Base-vs-extrapolated A/B on one axis:

```bash
python plot_tteg_tracking.py \
    --logs logs/tt_eg/eg_c10_TT_g0.1_base.out \
           logs/tt_eg/eg_c10_TT_g0.1_extrapolated.out \
    --labels base extrapolated \
    --outdir plots --prefix tteg_g0.1
```

Gamma sweep (base):

```bash
python plot_tteg_tracking.py \
    --logs logs/tt_eg/eg_c10_TT_g0.03_base.out \
           logs/tt_eg/eg_c10_TT_g0.1_base.out \
           logs/tt_eg/eg_c10_TT_g0.3_base.out \
    --labels "γ=0.03" "γ=0.1" "γ=0.3" \
    --outdir plots --prefix tteg_gamma_sweep
```

### 3. OMD comparison — `plot_oipo1_compare.py`

Uses `--run LABEL:PATH` (repeatable) and `--xclip` to match the paper's
4680-step cutoff:

```bash
python plot_oipo1_compare.py \
    --run "OMD entropic c10:logs/baselines/omd_oipo1_entropic_c10.out" \
    --run "EG entropic c5:logs/baselines/eg_entropic_c5.out" \
    --outdir plots --prefix omd_compare --xclip 4680
```

## Environment

`pip install -r ../requirements.txt` (matplotlib is included). Plotting needs
only Python + matplotlib; no GPU.
