#!/usr/bin/env python
"""Parse a training .out log and render a clean set of training plots.

Output filenames are simple: ``<prefix>_<X>.png`` where X is the metric
(e.g. ``group_risk_loss.png``).
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Match metric dicts like {'loss': 25.98, 'grad_norm': 22.09, ..., 'kl': 0.07, ...}
DICT_RE = re.compile(r"\{'loss':\s*[^}]*\}")
# Match the most recent step indicator preceding the dict: "<num>/<total>"
STEP_RE = re.compile(r"(\d+)\s*/\s*\d+\s*\[")

NUM_GROUPS = 4

# ---------- styling ----------
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 150,
    "savefig.dpi": 200,
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
})

GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------- parsing ----------
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

        rec = {
            "step": step,
            "loss": grab("loss"),
            "grad_norm": grab("grad_norm"),
            "kl": grab("kl"),
            "avg_loss": grab("avg_loss"),
            "rewards/accuracies": grab("rewards/accuracies"),
            "rewards/margins": grab("rewards/margins"),
            "rewards/chosen": grab("rewards/chosen"),
            "rewards/rejected": grab("rewards/rejected"),
            "dro/risk_beta": grab("dro/risk_beta"),
        }
        for g in range(NUM_GROUPS):
            rec[f"dro/Z_ema_{g}"] = grab(f"dro/Z_ema_{g}")
            rec[f"dro/w_{g}"] = grab(f"dro/w_{g}")
            rec[f"dro/acc_{g}"] = grab(f"dro/acc_{g}")
        records.append(rec)
    return records


# ---------- helpers ----------
def _ema(ys, alpha=0.1):
    out = []
    s = None
    for y in ys:
        s = y if s is None else alpha * y + (1 - alpha) * s
        out.append(s)
    return out


def _xy(records, key):
    xs, ys = [], []
    for r in records:
        v = r.get(key)
        if v is not None:
            xs.append(r["step"])
            ys.append(v)
    return xs, ys


def _save(fig, out_path):
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------- single-line plot ----------
def plot_single(records, key, out_path, title, ylabel=None, smooth=True, color="#1f77b4"):
    xs, ys = _xy(records, key)
    if not xs:
        print(f"  skip {key}: no data")
        return
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color=color, alpha=0.35, linewidth=1.0, label="raw")
    if smooth and len(ys) > 5:
        ax.plot(xs, _ema(ys, alpha=0.1), color=color, linewidth=2.0, label="EMA")
        ax.legend(loc="best")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or key)
    ax.set_title(title)
    _save(fig, out_path)


# ---------- multi-line per-group plot ----------
def plot_per_group(records, key_fmt, out_path, title, ylabel):
    fig, ax = plt.subplots()
    any_data = False
    for g in range(NUM_GROUPS):
        xs, ys = _xy(records, key_fmt.format(g=g))
        if xs:
            ax.plot(xs, ys, color=GROUP_COLORS[g], label=f"group {g}", alpha=0.85)
            any_data = True
    if not any_data:
        plt.close(fig)
        print(f"  skip {title}: no data")
        return
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", ncol=NUM_GROUPS)
    _save(fig, out_path)


# ---------- aggregates over Z_ema groups ----------
def _z_ema_aggregate(records, fn):
    xs, ys = [], []
    for r in records:
        vals = [r.get(f"dro/Z_ema_{g}") for g in range(NUM_GROUPS)]
        vals = [v for v in vals if v is not None]
        if len(vals) == NUM_GROUPS:
            xs.append(r["step"])
            ys.append(fn(vals))
    return xs, ys


def plot_z_ema_spread(records, out_path):
    xs, ys = _z_ema_aggregate(records, lambda v: max(v) - min(v))
    if not xs:
        print("  skip z_ema_spread: no data")
        return
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color="#d62728", alpha=0.35, linewidth=1.0, label="raw")
    ax.plot(xs, _ema(ys, alpha=0.1), color="#d62728", linewidth=2.0, label="EMA")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\max_e Z_{ema} - \min_e Z_{ema}$")
    ax.set_title("Z_ema spread across groups")
    ax.legend(loc="best")
    _save(fig, out_path)


def plot_z_ema_worst(records, out_path):
    xs, ys = _z_ema_aggregate(records, max)
    if not xs:
        print("  skip z_ema_worst: no data")
        return
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color="#2ca02c", alpha=0.35, linewidth=1.0, label="raw")
    ax.plot(xs, _ema(ys, alpha=0.1), color="#2ca02c", linewidth=2.0, label="EMA")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\max_e Z_{ema}[e]$ (worst-group loss)")
    ax.set_title("Worst-group loss vs step")
    ax.legend(loc="best")
    _save(fig, out_path)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="Path to .out log file")
    ap.add_argument("--outdir", default=None, help="Directory to write plots")
    ap.add_argument("--prefix", default="group_risk", help="Filename prefix")
    args = ap.parse_args()

    log_path = os.path.abspath(args.log)
    outdir = args.outdir or os.path.join(os.path.dirname(log_path), "plots")
    os.makedirs(outdir, exist_ok=True)

    records = parse_log(log_path)
    print(f"Parsed {len(records)} log entries from {log_path}")
    if not records:
        return
    last = records[-1]["step"]
    print(f"Last step seen: {last}")

    p = args.prefix

    # single metrics
    plot_single(records, "loss",                 os.path.join(outdir, f"{p}_loss.png"),
                "Loss vs step")
    plot_single(records, "avg_loss",             os.path.join(outdir, f"{p}_avg_loss.png"),
                "Unweighted avg per-sample IPO loss vs step", color="#9467bd")
    plot_single(records, "grad_norm",            os.path.join(outdir, f"{p}_grad_norm.png"),
                "Gradient norm vs step", color="#8c564b")
    plot_single(records, "kl",                   os.path.join(outdir, f"{p}_kl.png"),
                "KL vs step", color="#17becf")
    plot_single(records, "rewards/accuracies",   os.path.join(outdir, f"{p}_accuracy.png"),
                "Reward accuracy (chosen > rejected) vs step", color="#2ca02c")
    plot_single(records, "rewards/margins",      os.path.join(outdir, f"{p}_reward_margin.png"),
                "Reward margin vs step", color="#1f77b4")
    plot_single(records, "dro/risk_beta",        os.path.join(outdir, f"{p}_risk_beta.png"),
                "DRO risk_beta (anneal) vs step", smooth=False, color="#7f7f7f")

    # per-group
    plot_per_group(records, "dro/Z_ema_{g}",
                   os.path.join(outdir, f"{p}_z_ema_per_group.png"),
                   "Per-group Z_ema vs step", "Z_ema")
    plot_per_group(records, "dro/w_{g}",
                   os.path.join(outdir, f"{p}_w_per_group.png"),
                   "Per-group DRO weight w vs step", "w")
    plot_per_group(records, "dro/acc_{g}",
                   os.path.join(outdir, f"{p}_acc_per_group.png"),
                   "Per-group reward accuracy vs step", "accuracy")

    # Z_ema aggregates
    plot_z_ema_spread(records, os.path.join(outdir, f"{p}_z_ema_spread.png"))
    plot_z_ema_worst(records,  os.path.join(outdir, f"{p}_z_ema_worst.png"))


if __name__ == "__main__":
    main()
