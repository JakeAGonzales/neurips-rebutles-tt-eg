"""
Safety judge (beaver-7b-v1.0-cost) integration tests.

Tests:
  1. Safe vs unsafe score distribution (20 mixed rows from PKU-SafeRLHF)
     — safe responses should have lower cost than unsafe ones
  2. Severity spot-check: high-severity (level 3) responses scored on 3 examples
  3. Pairwise accuracy vs better_response_id (50 rows, expect > 55%)

Runtime: ~5-10 minutes on a single GPU.

Usage:
    python tests/test_safety_judge.py
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaConfig,
)
from transformers.modeling_outputs import ModelOutput

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "EGPO/lms"))
from utils import DATA_ROOT, MODEL_ROOT

COST_MODEL = "PKU-Alignment/beaver-7b-v1.0-cost"
MAX_LENGTH  = 512

# ── LlamaForScore ─────────────────────────────────────────────────────────────

@dataclass
class ScoreModelOutput(ModelOutput):
    scores: Optional[torch.FloatTensor] = None
    end_scores: Optional[torch.FloatTensor] = None


class LlamaForScore(LlamaPreTrainedModel):
    """
    Minimal reimplementation of PKU-Alignment's LlamaForScore.
    Identical weight layout to the upstream safe_rlhf version.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        score_dim = getattr(config, "score_dim", 1)
        bias      = getattr(config, "bias", False)
        self.score_head = nn.Linear(config.hidden_size, score_dim, bias=bias)
        self.post_init()

    def forward(
        self,
        input_ids:      torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ScoreModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden = outputs.last_hidden_state          # (B, T, H)
        scores = self.score_head(hidden)             # (B, T, score_dim)

        if attention_mask is not None:
            end_idx = attention_mask.sum(dim=-1) - 1  # (B,)
        else:
            end_idx = torch.full(
                (input_ids.size(0),), input_ids.size(1) - 1,
                device=input_ids.device,
            )
        end_scores = scores[torch.arange(scores.size(0), device=scores.device),
                            end_idx]                 # (B, score_dim)
        return ScoreModelOutput(scores=scores, end_scores=end_scores)


# ── loading helpers ───────────────────────────────────────────────────────────

def load_cost_model(device):
    print(f"Loading cost model: {COST_MODEL}")
    tok = AutoTokenizer.from_pretrained(COST_MODEL, cache_dir=MODEL_ROOT)
    cfg = LlamaConfig.from_pretrained(COST_MODEL, cache_dir=MODEL_ROOT)
    mdl = LlamaForScore.from_pretrained(
        COST_MODEL,
        config=cfg,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_ROOT,
        ignore_mismatched_sizes=True,
    ).to(device).eval()
    print(f"  score_head: {mdl.score_head}")
    return mdl, tok


@torch.no_grad()
def cost_score(model, tok, prompt: str, response: str, device) -> float:
    """Scalar cost score for (prompt, response). Higher = more harmful."""
    text = f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_LENGTH).to(device)
    out = model(**enc)
    return float(out.end_scores[0, 0].float().cpu())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl, tok = load_cost_model(device)

    print("Loading PKU-SafeRLHF test split...")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir=DATA_ROOT)

    # ── Test 1: safe vs unsafe score distribution ─────────────────────────────
    print("\n" + "=" * 62)
    print("TEST 1: cost scores vs ground-truth safety labels")
    print("Expected: safe responses have LOWER cost than unsafe ones")
    print("=" * 62)

    mixed = [r for r in ds["test"]
             if r["is_response_0_safe"] != r["is_response_1_safe"]][:20]

    safe_scores, unsafe_scores = [], []
    correct1 = 0
    for r in tqdm(mixed, desc="scoring"):
        for resp_id in [0, 1]:
            s = cost_score(mdl, tok, r["prompt"], r[f"response_{resp_id}"], device)
            if r[f"is_response_{resp_id}_safe"]:
                safe_scores.append(s)
            else:
                unsafe_scores.append(s)

        s0 = cost_score(mdl, tok, r["prompt"], r["response_0"], device)
        s1 = cost_score(mdl, tok, r["prompt"], r["response_1"], device)
        safe_id = 0 if r["is_response_0_safe"] else 1
        if (s0 < s1) == (safe_id == 0):
            correct1 += 1

    print(f"\n  safe   responses:  mean={np.mean(safe_scores):.4f}  "
          f"std={np.std(safe_scores):.4f}")
    print(f"  unsafe responses:  mean={np.mean(unsafe_scores):.4f}  "
          f"std={np.std(unsafe_scores):.4f}")
    print(f"\n  model ranks safe < unsafe: {correct1}/{len(mixed)} = "
          f"{correct1/len(mixed):.1%}  (expect > 60%)")

    # ── Test 2: severity spot-check ───────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TEST 2: severity spot-check (severity_level == 3 responses)")
    print("=" * 62)

    severe = [r for r in ds["test"]
              if r["response_0_severity_level"] == 3
              or r["response_1_severity_level"] == 3][:10]

    for r in severe[:3]:
        for resp_id in [0, 1]:
            sev  = r[f"response_{resp_id}_severity_level"]
            safe = r[f"is_response_{resp_id}_safe"]
            c    = cost_score(mdl, tok, r["prompt"], r[f"response_{resp_id}"], device)
            print(f"  [sev={sev} safe={safe}]  cost={c:.4f}  "
                  f"prompt={r['prompt'][:45]}...")

    # ── Test 3: pairwise accuracy vs better_response_id ──────────────────────
    print("\n" + "=" * 62)
    print("TEST 3: pairwise accuracy vs better_response_id (50 rows)")
    print("Expected: cost model ranks better response as lower cost > 55%")
    print("=" * 62)

    rows = list(ds["test"])[:50]
    correct3 = 0
    for r in tqdm(rows, desc="pairwise"):
        bid = r["better_response_id"]
        c_better = cost_score(mdl, tok, r["prompt"], r[f"response_{bid}"], device)
        c_worse  = cost_score(mdl, tok, r["prompt"], r[f"response_{1-bid}"], device)
        if c_better < c_worse:
            correct3 += 1

    print(f"\n  cost model agrees with better_response_id: "
          f"{correct3}/50 = {correct3/50:.1%}  (expect > 55%)")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  Test 1 (safe < unsafe): {correct1}/{len(mixed)} = {correct1/len(mixed):.1%}")
    print(f"    safe mean cost   = {np.mean(safe_scores):.4f}")
    print(f"    unsafe mean cost = {np.mean(unsafe_scores):.4f}")
    print(f"  Test 3 (pairwise): {correct3}/50 = {correct3/50:.1%}")
    print(f"  Cost model output: scalar per (prompt, response), higher = more harmful")
    print("=" * 62)


if __name__ == "__main__":
    main()
