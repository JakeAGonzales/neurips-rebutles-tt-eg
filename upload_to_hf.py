"""
Upload LoRA adapter checkpoints to HuggingFace.

For each run, creates one HF repo and uploads:
  - adapter_model.safetensors  (98 MB)
  - adapter_config.json

Each checkpoint is uploaded to a subdirectory named checkpoint-{step}/ in the
repo, so you can load a specific epoch with:
    PeftModel.from_pretrained("maxh-24/<repo>", subfolder="checkpoint-1872")

The latest (highest-step) checkpoint is also copied to the repo root so that
    PeftModel.from_pretrained("maxh-24/<repo>")
loads the best available checkpoint by default.

Usage
-----
    # dry run — prints what would be uploaded, creates no repos
    python upload_to_hf.py --dry_run

    # upload everything
    python upload_to_hf.py

    # upload one specific run by short name
    python upload_to_hf.py --only neutral

    # skip repos that already exist on HF (resume partial upload)
    python upload_to_hf.py --skip_existing
"""

import argparse, glob, os, shutil, sys, tempfile
from huggingface_hub import HfApi, create_repo, upload_folder

# ── run registry ──────────────────────────────────────────────────────────────

HF_USER   = "jakegonz"
EXP_BASE  = "/gscratch/mlopt/jakegonz/llm_tune/experiments"
RISK_ROOT = os.path.join(EXP_BASE, "risk_egpo")

# (short_name, hf_repo_suffix, run_name_glob)
# run_name_glob is matched against directory names inside RISK_ROOT; the most
# recently modified match is selected (timestamps differ across submissions).
RUNS = [
    # ── group-DRO over safety category e (K=1, oipo1) ─────────────────────────
    ("group-dro-b1",
     "risk-ipo-K1-gDRO-b1.0",
     "*Risk-OnlineIPO1-K1-neutral-eDRO1.0-strat2-lora-*"),

    ("group-dro-b0",
     "risk-ipo-K1-gDRO-b0.0",
     "*Risk-OnlineIPO1-K1-neutral-eDRO0.0-strat2-lora-*"),

    # ── ypp-risk over y'' samples (K=8, oipo1) ────────────────────────────────
    ("vanilla",
     "risk-ipo-K1-neutral",
     "*Risk-OnlineIPO1-K1-neutral-lora-*"),

    ("neutral-K8",
     "risk-ipo-K8-neutral",
     "*Risk-OnlineIPO1-K8-neutral-lora-*"),

    ("entropic-c1.0",
     "risk-ipo-K8-entropic-c1.0",
     "*Risk-OnlineIPO1-K8-entropic1.0-lora-*"),

    ("entropic-c2.0",
     "risk-ipo-K8-entropic-c2.0",
     "*Risk-OnlineIPO1-K8-entropic2.0-lora-*"),

    ("entropic-c5.0",
     "risk-ipo-K8-entropic-c5.0",
     "*Risk-OnlineIPO1-K8-entropic5.0-lora-*"),

    ("entropic-c10.0",
     "risk-ipo-K8-entropic-c10.0",
     "*Risk-OnlineIPO1-K8-entropic10.0-lora-*"),

    ("cvar-a0.5",
     "risk-ipo-K8-cvar-a0.5",
     "*Risk-OnlineIPO1-K8-cvar0.5-lora-*"),

    ("cvar-a0.25",
     "risk-ipo-K8-cvar-a0.25",
     "*Risk-OnlineIPO1-K8-cvar0.25-lora-*"),

    ("cvar-a0.125",
     "risk-ipo-K8-cvar-a0.125",
     "*Risk-OnlineIPO1-K8-cvar0.125-lora-*"),

    # ── extragradient (eg) variant (K=8) ──────────────────────────────────────
    ("eg-entropic-c5.0",
     "risk-eg-K8-entropic-c5.0",
     "*Risk-Extragradient-K8-entropic5.0-lora-*"),

    # ── group-DRO CVaR (K=1) ───────────────────────────────────────────────────
    ("group-dro-cvar-a0.25",
     "risk-ipo-K1-gDRO-cvar0.25",
     "*Risk-OnlineIPO1-K1-neutral-eDROcvar0.25-strat2-lora-*"),

    # ── new extragradient runs (K=1 vanilla, K=8 neutral, K=8 cvar 0.125/0.25) ─
    ("eg-vanilla-K1",
     "risk-eg-K1-neutral",
     "*Risk-Extragradient-K1-neutral-lora-*"),
    ("eg-neutral-K8",
     "risk-eg-K8-neutral",
     "*Risk-Extragradient-K8-neutral-lora-*"),
    ("eg-cvar-a0.125",
     "risk-eg-K8-cvar-a0.125",
     "*Risk-Extragradient-K8-cvar0.125-lora-*"),
    ("eg-cvar-a0.25",
     "risk-eg-K8-cvar-a0.25",
     "*Risk-Extragradient-K8-cvar0.25-lora-*"),

    # ── group-DRO with severity-weighted prior 1:2:4:8 (K=1, oipo1) ───────────
    ("group-dro-b0-prior1248",
     "risk-ipo-K1-gDRO-b0.0-prior1248",
     "*Risk-OnlineIPO1-K1-neutral-eDRO0.0-strat2-prior1248-lora-*"),
    ("group-dro-b5-prior1248",
     "risk-ipo-K1-gDRO-b5.0-prior1248",
     "*Risk-OnlineIPO1-K1-neutral-eDRO5.0-strat2-prior1248-lora-*"),
    ("group-dro-cvar-a0.25-prior1248",
     "risk-ipo-K1-gDRO-cvar0.25-prior1248",
     "*Risk-OnlineIPO1-K1-neutral-eDROcvar0.25-strat2-prior1248-lora-*"),

    # ── nash-MD runs (K=1 vanilla, K=8 neutral) ───────────────────────────────
    ("nmd-vanilla-K1",
     "risk-nmd-K1-neutral",
     "*Risk-NashMD-K1-neutral-lora-*"),
    ("nmd-neutral-K8",
     "risk-nmd-K8-neutral",
     "*Risk-NashMD-K8-neutral-lora-*"),
]

ADAPTER_FILES = ["adapter_model.safetensors", "adapter_config.json"]


def get_checkpoints(run_dir, complete_only=True):
    """Return sorted list of (step, ckpt_path) tuples.

    If complete_only=True (default), only checkpoints that contain ALL files
    in ADAPTER_FILES are included. This prevents picking up an in-progress
    save from a still-running training job.
    """
    paths = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    result = []
    for p in paths:
        try:
            step = int(os.path.basename(p).split("-")[1])
        except (IndexError, ValueError):
            continue
        if complete_only:
            if not all(os.path.exists(os.path.join(p, f)) for f in ADAPTER_FILES):
                continue
        result.append((step, p))
    return sorted(result)


def stage_adapter(ckpt_path, dest_dir):
    """Copy only adapter files from ckpt_path into dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    for fname in ADAPTER_FILES:
        src = os.path.join(ckpt_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, fname))
        else:
            print(f"    [warn] missing {fname} in {ckpt_path}")


def resolve_run_dir(pattern):
    """Resolve a glob pattern under RISK_ROOT to the most recently modified
    matching directory. Returns None if no match."""
    matches = [p for p in glob.glob(os.path.join(RISK_ROOT, pattern)) if os.path.isdir(p)]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def upload_run(short_name, repo_suffix, pattern, api, args):
    owner = args.org if args.org else HF_USER
    prefix = args.prefix or ""
    repo_id = f"{owner}/{prefix}{repo_suffix}"
    print(f"\n{'─'*60}")
    print(f"  {short_name}  →  {repo_id}")

    run_dir = resolve_run_dir(pattern)
    if run_dir is None:
        print(f"  [skip] no directory matches pattern: {pattern}")
        return
    print(f"  Resolved: {run_dir}")

    if not os.path.isdir(run_dir):
        print(f"  [skip] local dir not found: {run_dir}")
        return

    checkpoints = get_checkpoints(run_dir)
    if not checkpoints:
        print(f"  [skip] no checkpoints in {run_dir}")
        return

    latest_step, latest_ckpt = checkpoints[-1]
    print(f"  Checkpoints: {[s for s, _ in checkpoints]}")
    print(f"  Latest: checkpoint-{latest_step}")

    if args.dry_run:
        print(f"  [dry-run] would create repo and upload {len(checkpoints)} checkpoints")
        return

    # Create repo (idempotent)
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception as e:
        print(f"  [warn] create_repo: {e}")

    # Check if already uploaded (skip_existing mode)
    if args.skip_existing:
        try:
            files = api.list_repo_files(repo_id)
            if "adapter_model.safetensors" in files:
                print(f"  [skip] repo already has adapter files")
                return
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as staging:
        # Stage each checkpoint in its own subfolder
        for step, ckpt_path in checkpoints:
            subfolder = os.path.join(staging, f"checkpoint-{step}")
            stage_adapter(ckpt_path, subfolder)

        # Also stage latest at root (enables from_pretrained without subfolder)
        stage_adapter(latest_ckpt, staging)

        print(f"  Uploading to {repo_id} ...")
        upload_folder(
            repo_id=repo_id,
            folder_path=staging,
            repo_type="model",
            commit_message=f"Upload {short_name} LoRA adapter (latest: checkpoint-{latest_step})",
        )

    print(f"  Done: https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run",      action="store_true",
                    help="Print what would be uploaded without actually uploading")
    ap.add_argument("--only",         type=str, default=None,
                    help="Comma-separated short names to upload (default: all)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip repos that already have adapter files on HF")
    ap.add_argument("--org", type=str, default=None,
                                         help="Upload to a HF organization instead of your personal account")
    ap.add_argument("--prefix", type=str, default=None,
                                         help="String to prepend to each repo name (e.g. 'jg-' for the rat-lab convention)")
    args = ap.parse_args()

    api = HfApi()
    try:
        me = api.whoami()["name"]
        print(f"Logged in as: {me}")
        if me != HF_USER:
            print(f"[warn] expected user '{HF_USER}', got '{me}'")
    except Exception as e:
        print(f"[error] HF auth failed: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)

    only = set(args.only.split(",")) if args.only else None
    runs = [(s, r, p) for s, r, p in RUNS if only is None or s in only]

    print(f"\n{'═'*60}")
    print(f"  Uploading {len(runs)} run(s) to HuggingFace")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Files per checkpoint: {ADAPTER_FILES}")
    print(f"{'═'*60}")

    for short_name, repo_suffix, pattern in runs:
        upload_run(short_name, repo_suffix, pattern, api, args)

    print(f"\n{'═'*60}")
    print("Done.")


if __name__ == "__main__":
    main()
