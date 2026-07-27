#!/usr/bin/env python
"""Plot TT-EG bias-tracking diagnostics from one or more training .out logs.

Shows how the ONLINE two-timescale tracker ||xi|| follows the per-step
DELTA-METHOD bias ||b_hat|| (the analytic second-order estimate), plus the
tracking residual ||xi - b_hat||. Multiple logs are overlaid so a base-vs-
extrapolated A/B (or a gamma sweep) can be compared on one axis.

Reads the metric dicts the HF Trainer prints each logging step (the same
source as plot_training_metrics.py), so no rerun / extra logging is needed.

Example:
    python plot_tteg_tracking.py \
        --logs output_files/tteg_ab_base_37684086_1.out \
               output_files/tteg_ab_xt_37684087_1.out \
        --labels base extrapolated --prefix tteg_g0.1
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DICT_RE = re.compile(r"\{'loss':[^}]*\}")
STEP_RE = re.compile(r"(\d+)\s*/\s*\d+\s*\[")
# grab every tteg/<name>: <value> pair inside a log dict
TTEG_KV = re.compile(r"'(tteg/[A-Za-z0-9_/]+)':\s*([-+0-9.eE]+)")

# name -> (log key, human label)
SERIES = {
    "xi":        ("tteg/xi_norm",              r"$\|\xi\|$ (online tracker)"),
    "b_hat":     ("tteg/b_hat_norm",           r"$\|\hat b\|$ (delta-method)"),
    "resid":     ("tteg/tracking_residual",    r"$\|\xi-\hat b\|$"),
    "resid_sq":  ("tteg/tracking_residual_sq", r"$\|\xi-\hat b\|^2$"),
    "frac":      ("tteg/debias_fraction",      r"$\|\xi\|/\|g_{\mathrm{main}}\|$ (correction size / stability)"),
    "payoff":    ("tteg/b_hat_payoff_absmax",  r"payoff-space $|\hat b|_\infty$"),
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.rcParams.update({
    "figure.figsize": (9, 5), "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "lines.linewidth": 1.8, "legend.frameon": False, "font.size": 10,
})


def parse_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    recs = []
    for m in DICT_RE.finditer(text):
        steps = STEP_RE.findall(text[:m.start()])
        step = int(steps[-1]) if steps else len(recs) + 1
        body = m.group(0)

        rec = {"step": step}
        for k, v in TTEG_KV.findall(body):
            rec[k] = float(v)
        recs.append(rec)
    return recs


def _xy(recs, key):
    xs, ys = [], []
    for r in recs:
        v = r.get(key)
        if v is not None:
            xs.append(r["step"])
            ys.append(v)
    return xs, ys


def _ema(ys, alpha=0.15):
    out, s = [], None
    for y in ys:
        s = y if s is None else alpha * y + (1 - alpha) * s
        out.append(s)
    return out


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_tracking(runs, out_path):
    """xi (solid) vs b_hat (dashed) per run: does the online tracker follow the
    delta-method bias?"""
    fig, ax = plt.subplots()
    drew = False
    for i, (label, recs) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        xx, yx = _xy(recs, SERIES["xi"][0])
        xb, yb = _xy(recs, SERIES["b_hat"][0])
        if xx:
            ax.plot(xx, yx, color=c, linestyle="-",
                    label=f"{label}: " + SERIES["xi"][1]); drew = True
        if xb:
            ax.plot(xb, yb, color=c, linestyle="--", alpha=0.7,
                    label=f"{label}: " + SERIES["b_hat"][1]); drew = True
    if not drew:
        plt.close(fig); print("  skip tracking: no data"); return
    ax.set_xlabel("step"); ax.set_ylabel("parameter-space norm")
    ax.set_title("Online tracker vs delta-method bias")
    ax.legend(loc="best"); _save(fig, out_path)


def plot_overlay(runs, name, out_path, hline=None, smooth=True, logy=False):
    key, lbl = SERIES[name]
    fig, ax = plt.subplots()
    drew = False
    for i, (label, recs) in enumerate(runs):
        xs, ys = _xy(recs, key)
        if not xs:
            continue
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, color=c, alpha=0.30, linewidth=1.0)
        ax.plot(xs, _ema(ys) if smooth and len(ys) > 5 else ys,
                color=c, label=label)
        drew = True
    if not drew:
        plt.close(fig); print(f"  skip {name}: no data"); return
    if hline is not None:
        ax.axhline(hline, color="k", linestyle=":", linewidth=1.2,
                   label=f"halt @ {hline}")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(lbl); ax.set_title(lbl + " vs step")
    ax.legend(loc="best"); _save(fig, out_path)


def _detect_windows(runs):
    ws = set()
    for _, recs in runs:
        for r in recs:
            for k in r:
                m = re.match(r"tteg/resid_ratio(\d+)$", k)
                if m:
                    ws.add(int(m.group(1)))
    return sorted(ws)


def plot_window_overlay(runs, key_fmt, W, out_path, ylabel, title,
                        hlines=(), smooth=True, logy=False):
    """Overlay one windowed metric across runs (base vs extrapolated)."""
    fig, ax = plt.subplots()
    drew = False
    for i, (label, recs) in enumerate(runs):
        xs, ys = _xy(recs, key_fmt.format(W=W))
        if not xs:
            continue
        c = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, color=c, alpha=0.30, linewidth=1.0)
        ax.plot(xs, _ema(ys) if smooth and len(ys) > 5 else ys, color=c, label=label)
        drew = True
    if not drew:
        plt.close(fig); print(f"  skip {title}: no data"); return
    for h, hl in hlines:
        ax.axhline(h, color="k", linestyle=":", linewidth=1.1, label=hl)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best"); _save(fig, out_path)


def plot_direct_compare(recs, label, W, out_path):
    """Uncorrected window-avg bias ||b_bar|| vs residual ||b_bar - xi||.
    TT correction helps whenever the residual sits below the uncorrected curve."""
    xb, yb = _xy(recs, f"tteg/bbar{W}_norm")
    xr, yr = _xy(recs, f"tteg/resid_bias{W}")
    if not xb:
        print(f"  skip compare W{W} {label}: no data"); return
    fig, ax = plt.subplots()
    ax.plot(xb, yb, color="#d62728", label=r"uncorrected $\|\bar b^{(%d)}\|$" % W)
    ax.plot(xr, yr, color="#1f77b4",
            label=r"residual after TT $\|\bar b^{(%d)}-\xi\|$" % W)
    ax.set_xlabel("step"); ax.set_ylabel("parameter-space norm")
    ax.set_title(f"{label}: TT correction vs uncorrected bias (W={W})")
    ax.legend(loc="best"); _save(fig, out_path)


def plot_estimator_compare(runs, out_path):
    """Directly answers 'how close is the corrected estimator to the original?'.
    Overlays the ORIGINAL plug-in risk estimate rho_hat against the DEBIASED
    estimate (rho_hat - b), where b is the delta-method bias. The curves nearly
    coincide (gap = b ~ few %), zoomed so the gap is visible; the annotation
    reports the mean relative gap |b|/|rho|."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    rels = []
    for i, (label, recs) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        xr, yr = _xy(recs, "tteg/rho_plugin")
        xb, yb = _xy(recs, "tteg/bias_payoff_delta")
        if not xr or not xb:
            continue
        n = min(len(yr), len(yb))
        x = xr[:n]
        orig = yr[:n]
        corr = [orig[k] - yb[k] for k in range(n)]        # debiased = rho_hat - b
        rels += [abs(yb[k]) / (abs(orig[k]) + 1e-12) for k in range(n)]
        ax.plot(x, _ema(orig) if n > 5 else orig, color=c, label=f"{label}: original plug-in")
        ax.plot(x, _ema(corr) if n > 5 else corr, color=c, linestyle="--",
                alpha=0.85, label=f"{label}: debiased (rho - b)")
        drew = True
    if not drew:
        plt.close(fig); print("  skip estimator_compare: no data"); return
    mean_rel = sum(rels) / max(len(rels), 1)
    ax.set_xlabel("step"); ax.set_ylabel(r"risk estimate $\rho$ (payoff space)")
    ax.set_title("Original plug-in vs debiased estimator (gap = bias b)")
    ax.annotate(f"mean |b|/|rho| = {mean_rel*100:.1f}%", xy=(0.98, 0.04),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.7"))
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_bias_magnitude(runs, out_path):
    """MAGNITUDE check (NOT a formula validation): the plug-in risk estimate rho
    vs its OWN delta-method bias b, both in payoff space, and the ratio |b|/|rho|.
    This shows the finite-sample bias is a small CORRECTION relative to the risk
    level (|b|/|rho| small is expected/good). It does NOT compare the delta-method
    against an empirical bias estimate -- for that see the (separate) validation
    plot where the target ratio is ~1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    drew = False
    for i, (label, recs) in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        xr, yr = _xy(recs, "tteg/rho_plugin")
        xb, yb = _xy(recs, "tteg/bias_payoff_delta")
        xrel, yrel = _xy(recs, "tteg/bias_rel")
        if xr:
            ax1.plot(xr, _ema(yr) if len(yr) > 5 else yr, color=c, label=f"{label}: rho")
            drew = True
        if xb:
            ax1.plot(xb, _ema(yb) if len(yb) > 5 else yb, color=c, linestyle="--",
                     alpha=0.8, label=f"{label}: delta bias b")
        if xrel:
            ax2.plot(xrel, _ema(yrel) if len(yrel) > 5 else yrel, color=c, label=label)
    if not drew:
        plt.close(fig); print("  skip bias_magnitude: no data"); return
    ax1.set_xlabel("step"); ax1.set_ylabel("payoff space")
    ax1.set_title(r"Plug-in risk estimate $\rho$ vs its own bias $b$")
    ax1.legend(loc="best")
    ax2.set_xlabel("step"); ax2.set_ylabel(r"$|b|/|\rho|$")
    ax2.set_title(r"Bias size relative to risk level (correction is small)")
    ax2.legend(loc="best")
    _save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help=".out log files")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend labels (default: file basenames)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--prefix", default="tteg")
    args = ap.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
    assert len(labels) == len(args.logs), "labels must match number of logs"

    runs = []
    for path, label in zip(args.logs, labels):
        recs = parse_log(os.path.abspath(path))
        n = sum(1 for r in recs if r.get(SERIES["xi"][0]) is not None)
        print(f"{label}: {len(recs)} log entries, {n} with tteg stats "
              f"(last step {recs[-1]['step'] if recs else 'NA'})")
        runs.append((label, recs))

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.logs[0])),
                                         "plots")
    os.makedirs(outdir, exist_ok=True)
    p = args.prefix

    plot_tracking(runs, os.path.join(outdir, f"{p}_tracking.png"))
    plot_overlay(runs, "xi",       os.path.join(outdir, f"{p}_xi_norm.png"))
    plot_overlay(runs, "b_hat",    os.path.join(outdir, f"{p}_b_hat_norm.png"))
    plot_overlay(runs, "resid",    os.path.join(outdir, f"{p}_residual.png"))
    plot_overlay(runs, "resid_sq", os.path.join(outdir, f"{p}_residual_sq.png"), logy=True)
    plot_overlay(runs, "frac",     os.path.join(outdir, f"{p}_debias_fraction.png"), hline=1.0)
    plot_overlay(runs, "payoff",   os.path.join(outdir, f"{p}_payoff_bias.png"))
    plot_bias_magnitude(runs, os.path.join(outdir, f"{p}_bias_magnitude.png"))
    plot_estimator_compare(runs, os.path.join(outdir, f"{p}_estimator_compare.png"))

    # ---- advisor's window-averaged VECTOR diagnostics (need the new run) ----
    windows = _detect_windows(runs)
    if not windows:
        print("No window diagnostics (tteg/resid_ratio{W}) in these logs; "
              "rerun with the updated tt_eg.py to populate them.")
    for W in windows:
        plot_window_overlay(
            runs, "tteg/resid_ratio{W}", W,
            os.path.join(outdir, f"{p}_resid_ratio_W{W}.png"),
            ylabel=r"$\rho^{(%d)}_t=\|\bar b-\xi\|/\|\bar b\|$" % W,
            title=f"Normalized residual bias, W={W}  [PRIMARY]",
            hlines=((1.0, "no correction (=1)"),))
        plot_window_overlay(
            runs, "tteg/cos{W}", W,
            os.path.join(outdir, f"{p}_cos_W{W}.png"),
            ylabel=r"$\cos(\xi,\bar b^{(%d)})$" % W,
            title=f"Directional alignment, W={W}",
            hlines=((0.0, "orthogonal"), (1.0, "aligned")))
        plot_window_overlay(
            runs, "tteg/scale{W}", W,
            os.path.join(outdir, f"{p}_scale_W{W}.png"),
            ylabel=r"$\|\xi\|/\|\bar b^{(%d)}\|$" % W,
            title=f"Scale ratio, W={W}",
            hlines=((1.0, "correct scale"),))
        for label, recs in runs:
            plot_direct_compare(
                recs, label, W,
                os.path.join(outdir, f"{p}_compare_W{W}_{label}.png"))


if __name__ == "__main__":
    main()
