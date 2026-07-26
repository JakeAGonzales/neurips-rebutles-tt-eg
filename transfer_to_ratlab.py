"""
Transfer Jake's HF model repos from `jakegonz/<name>` to `rat-lab/jg-<name>`.

The `jg-` prefix avoids name collisions with other lab members' models.

Usage
-----
    # dry-run (prints what would happen)
    python transfer_to_ratlab.py --dry_run

    # do the transfers
    python transfer_to_ratlab.py

    # transfer just one
    python transfer_to_ratlab.py --only risk-ipo-K8-neutral

Notes
-----
- Requires `huggingface-cli login` first (your token must have write access to
  rat-lab, which membership grants automatically).
- HF auto-redirects the old `jakegonz/<name>` URL to the new path, so existing
  links won't break immediately.
- Idempotent: running twice is safe; already-moved repos are skipped.
"""

import argparse
import sys

from huggingface_hub import HfApi


SOURCE_USER = "jakegonz"
DEST_ORG    = "rat-lab"
NAME_PREFIX = "jg-"

REPOS = [
    "risk-ipo-K1-gDRO-b1.0",
    "risk-ipo-K1-gDRO-b0.0",
    "risk-ipo-K1-gDRO-cvar0.25",
    "risk-ipo-K1-neutral",
    "risk-ipo-K8-neutral",
    "risk-ipo-K8-entropic-c1.0",
    "risk-ipo-K8-entropic-c2.0",
    "risk-ipo-K8-entropic-c5.0",
    "risk-ipo-K8-entropic-c10.0",
    "risk-ipo-K8-cvar-a0.5",
    "risk-ipo-K8-cvar-a0.25",
    "risk-ipo-K8-cvar-a0.125",
    "risk-eg-K8-entropic-c5.0",
    "risk-eg-K1-neutral",
    "risk-eg-K8-neutral",
    "risk-eg-K8-cvar-a0.125",
    "risk-eg-K8-cvar-a0.25",
    "risk-nmd-K1-neutral",
    "risk-nmd-K8-neutral",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="Print actions without performing them")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated repo names to transfer (default: all)")
    ap.add_argument("--no_prefix", action="store_true",
                    help="Skip the 'jg-' prefix (use original names)")
    args = ap.parse_args()

    api = HfApi()
    try:
        me = api.whoami()["name"]
        print(f"Logged in as: {me}")
    except Exception as e:
        print(f"[error] HF auth failed: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)

    only = set(args.only.split(",")) if args.only else None
    targets = [r for r in REPOS if only is None or r in only]

    prefix = "" if args.no_prefix else NAME_PREFIX

    print(f"\n{'='*60}")
    print(f"  Transferring {len(targets)} repo(s) from {SOURCE_USER} -> {DEST_ORG}")
    print(f"  Name prefix: {prefix!r}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    for name in targets:
        from_id = f"{SOURCE_USER}/{name}"
        to_id   = f"{DEST_ORG}/{prefix}{name}"
        print(f"  {from_id}\n    -> {to_id}")

        if args.dry_run:
            continue

        try:
            api.move_repo(from_id=from_id, to_id=to_id, repo_type="model")
            print(f"    [ok]")
        except Exception as e:
            print(f"    [skip/err] {e}")

    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
