#!/usr/bin/env python
"""Overlay loss, KL, and grad_norm for multiple OIPO1 runs on shared plots.

Usage:
    python plot_oipo1_compare.py --outdir plots_oipo1 \
        --run vanilla:/path/to/oipo1_vanilla_*.out \
        --run ypp_risk:/path/to/oipo1_ypp_risk_*.out
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DICT_RE = re.compile(r"\{'loss':\s*[^}]*\}")
STEP_RE = re.compile(r"(\d+)\s*/\s*\d+\s*\[")

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "font.size": 10,
    # NeurIPS-style serif (Times New Roman if available, else any Times-like font).
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

RUN_COLORS = {
    # OIPO1 family (blues / reds) -- legacy keys
    "OIPO1":              "#1f77b4",
    "OIPO1_K8":           "#4a9fd6",
    "OIPO1_ent_c1":       "#d62728",
    "OIPO1_ent_c2":       "#e34a4b",
    "OIPO1_ent_c5":       "#b71c1d",
    "OIPO1_ent_c10":      "#7f0e0e",
    "OIPO1_cvar_a050":    "#ff9896",
    "OIPO1_cvar_a025":    "#ff7f0e",
    "OIPO1_cvar_a0125":   "#bcbd22",
    # Extragradient family (greens / olives) -- legacy keys
    "EG_K1":              "#2ca02c",
    "EG_K8":              "#5cb85c",
    "EG_ent_c5":          "#006400",
    "EG_cvar_a025":       "#9acd32",
    "EG_cvar_a0125":      "#556b2f",
    # NashMD (purples) -- legacy keys
    "NashMD_K1":          "#9467bd",
    "NashMD_K8":          "#5e3c99",
    # Group-DRO (cyan / teal / brown) -- legacy keys
    "gDRO_b1":            "#17becf",
    "gDRO_b0":            "#8c564b",
    "gDRO_cvar_a025":     "#e377c2",
    # Paper-style labels
    "EGPO":                       "#1f77b4",
    "OMD (K=8)":                  "#4a9fd6",
    "OMD-Ent (\u03c4=1)":         "#d62728",
    "OMD-Ent (\u03c4=2)":         "#e34a4b",
    "OMD-Ent (\u03c4=5)":         "#b71c1d",
    "OMD-Ent (\u03c4=10)":        "#7f0e0e",
    "OMD-CVaR (\u03b1=0.50)":     "#ff9896",
    "OMD-CVaR (\u03b1=0.25)":     "#ff7f0e",
    "OMD-CVaR (\u03b1=0.125)":    "#bcbd22",
    "EG (K=1)":                   "#2ca02c",
    "EG (K=8)":                   "#5cb85c",
    "EG-Ent (\u03c4=5)":          "#006400",
    "EG-CVaR (\u03b1=0.25)":      "#9acd32",
    "EG-CVaR (\u03b1=0.125)":     "#556b2f",
    "Nash-MD":                    "#9467bd",
    "Nash-MD (K=1)":              "#9467bd",
    "Nash-MD (K=8)":              "#5e3c99",
    "gDRO":                       "#8c564b",
    "gDRO-Ent (\u03b2=1)":        "#17becf",
    "gDRO-CVaR (\u03b1=0.25)":    "#e377c2",
}
FALLBACK = ["#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a"]


def parse_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    records = []
    for m in DICT_RE.finditer(text):
        prefix = text[:m.start()]
        steps = STEP_RE.findall(prefix)
        step = int(steps[-1]) if steps else len(records) + 1
        body = m.group(0)

        def grab(key):
            mm = re.search(r"'" + re.escape(key) + r"':\s*([-+0-9.eE]+)", body)
            return float(mm.group(1)) if mm else None

        records.append({
            "step": step,
            "loss": grab("loss"),
            "grad_norm": grab("grad_norm"),
            "kl": grab("kl"),
            "rewards/accuracies": grab("rewards/accuracies"),
        })
    return records


def ema(ys, alpha=0.1):
    out = []
    s = None
    for y in ys:
        s = y if s is None else alpha * y + (1 - alpha) * s
        out.append(s)
    return out


def overlay(runs, key, out_path, title, ylabel, legend_outside=False, xclip=None):
    if legend_outside:
        fig, ax = plt.subplots(figsize=(10.5, 5))
    else:
        fig, ax = plt.subplots()
    plotted = False
    for i, (label, records) in enumerate(runs):
        xs = [r["step"] for r in records if r[key] is not None]
        ys = [r[key] for r in records if r[key] is not None]
        if xclip is not None:
            paired = [(x, y) for x, y in zip(xs, ys) if x <= xclip]
            xs = [x for x, _ in paired]
            ys = [y for _, y in paired]
        if not xs:
            continue
        color = RUN_COLORS.get(label, FALLBACK[i % len(FALLBACK)])
        ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.8)
        smoothed = ema(ys, alpha=0.1) if len(ys) > 5 else ys
        ax.plot(xs, smoothed, color=color, linewidth=1.8, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"  skip {key}: no data")
        return
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    n = len(runs)
    if legend_outside:
        ncol = 1 if n <= 14 else 2
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  ncol=ncol, fontsize=8, borderaxespad=0.0)
    else:
        ncol = 3 if n > 6 else (2 if n > 3 else 1)
        fontsize = 7 if n > 12 else (8 if n > 8 else 9)
        ax.legend(loc="best", ncol=ncol, fontsize=fontsize)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="Spec 'label:/path/to/file.out' (repeatable)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="oipo1")
    ap.add_argument("--legend-outside", action="store_true",
                    help="Place legend to the right of the axes (recommended for >12 runs).")
    ap.add_argument("--xclip", type=int, default=None,
                    help="Clip x-axis (steps) to at most this value across all runs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    runs = []
    for spec in args.run:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"bad --run spec: {spec!r} (expected label:path)")
        records = parse_log(os.path.abspath(path))
        last = records[-1]["step"] if records else None
        print(f"[{label}] parsed {len(records)} entries from {path} (last step: {last})")
        runs.append((label, records))

    p = args.prefix
    lo = args.legend_outside
    xc = args.xclip
    overlay(runs, "loss",      os.path.join(args.outdir, f"{p}_loss.png"),
            "Loss vs step", "loss", legend_outside=lo, xclip=xc)
    overlay(runs, "kl",        os.path.join(args.outdir, f"{p}_kl.png"),
            "KL vs step", "kl", legend_outside=lo, xclip=xc)
    overlay(runs, "grad_norm", os.path.join(args.outdir, f"{p}_grad_norm.png"),
            "Gradient norm vs step", "grad_norm", legend_outside=lo, xclip=xc)
    overlay(runs, "rewards/accuracies", os.path.join(args.outdir, f"{p}_accuracy.png"),
            "Reward accuracy (chosen > rejected) vs step", "accuracy",
            legend_outside=lo, xclip=xc)


if __name__ == "__main__":
    main()
