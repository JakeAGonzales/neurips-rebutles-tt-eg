"""
Preference judge integration tests.

Tests:
  1. SFT vs SFT (same temperature, different seeds) → win rate ≈ 0.50
  2. SFT low-temp vs SFT high-temp → low-temp should win more often
  3. PKU-SafeRLHF better_response_id agreement (50 rows, expect > 60%)
  4. Safe vs unsafe response ranking (20 mixed rows, expect safe < unsafe cost)

Runtime: ~5-10 minutes on a single GPU.

Usage:
    python tests/test_pref_judge.py
"""

import os, sys, torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "EGPO/lms"))
from utils import DATA_ROOT, MODEL_ROOT

SFT_NAME  = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
PREF_NAME = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200

PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a short poem about the ocean.",
    "What are three tips for staying focused while studying?",
    "Describe how a neural network learns.",
    "What is the capital of France, and why is it famous?",
    "Give me a recipe for a simple tomato pasta.",
    "How does photosynthesis work?",
    "What are the main causes of the French Revolution?",
]

PREF_TEMPLATE = '''I require a leaderboard for various large language models. I\'ll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

'''

# ── load models ───────────────────────────────────────────────────────────────

print("Loading SFT model...")
sft_tok = AutoTokenizer.from_pretrained(SFT_NAME, cache_dir=MODEL_ROOT)
if not sft_tok.pad_token:
    sft_tok.pad_token = sft_tok.eos_token
sft_tok.padding_side = "left"

sft = AutoModelForCausalLM.from_pretrained(
    SFT_NAME, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=MODEL_ROOT,
)
sft.eval()

print("Loading preference model...")
pref_tok = AutoTokenizer.from_pretrained(PREF_NAME, cache_dir=MODEL_ROOT)
if not pref_tok.pad_token:
    pref_tok.pad_token = pref_tok.eos_token
pref_tok.padding_side = "right"

pref_model = AutoModelForSequenceClassification.from_pretrained(
    PREF_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    cache_dir=MODEL_ROOT,
).to(DEVICE).eval()

# ── helpers ───────────────────────────────────────────────────────────────────

def generate(prompts: list[str], temperature: float, seed: int = 0) -> list[str]:
    torch.manual_seed(seed)
    inputs = sft_tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = sft.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=sft_tok.pad_token_id,
        )
    L = inputs["input_ids"].shape[1]
    return [s.strip() for s in sft_tok.batch_decode(out[:, L:], skip_special_tokens=True)]


def judge_batch(prompts: list[str], responses_a: list[str], responses_b: list[str],
                shuffle: bool = True) -> list[float]:
    """Returns P(A wins) for each pair."""
    flip = torch.randint(0, 2, (len(prompts),)).bool() if shuffle else \
           torch.zeros(len(prompts), dtype=torch.bool)

    queries = []
    for i, (p, a, b) in enumerate(zip(prompts, responses_a, responses_b)):
        r0, r1 = (b, a) if flip[i] else (a, b)
        queries.append(PREF_TEMPLATE.format(prompt=p, response0=r0, response1=r1))

    enc = pref_tok(queries, return_tensors="pt", padding=True,
                   truncation=True, max_length=1024)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = pref_model(**enc).logits   # (N, 2): [0]=response0 wins, [1]=response1 wins

    probs = torch.softmax(logits, dim=-1).cpu()
    p_a_wins = probs[torch.arange(len(prompts)), flip.long()].tolist()
    return p_a_wins


def judge_pairs_batched(prompts, responses_a, responses_b, batch_size=64):
    results = []
    for s in range(0, len(prompts), batch_size):
        results.extend(judge_batch(
            prompts[s:s+batch_size],
            responses_a[s:s+batch_size],
            responses_b[s:s+batch_size],
        ))
    return results


def report(scores, prompts, label_a, label_b):
    print(f"\n{'─'*62}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'─'*62}")
    for i, (s, p) in enumerate(zip(scores, prompts)):
        v = "A wins" if s > 0.5 else ("B wins" if s < 0.5 else "tie")
        print(f"  [{i+1:2d}] P(A)={s:.3f}  ({v})  |  {p[:50]}")
    wr = sum(s > 0.5 for s in scores) / len(scores)
    print(f"\n  A win rate : {wr:.2f}  ({sum(s>0.5 for s in scores)}/{len(scores)})")
    print(f"  Mean P(A)  : {np.mean(scores):.4f}  (expect ~0.50 for identical policies)")


# ── Test 1: same policy, different seeds → ~50% ───────────────────────────────

print("\n" + "=" * 62)
print("TEST 1: SFT (T=1.0, seed=0)  vs  SFT (T=1.0, seed=1)")
print("Expected: win rate ≈ 0.50")
print("=" * 62)
comp_a = generate(PROMPTS, temperature=1.0, seed=0)
comp_b = generate(PROMPTS, temperature=1.0, seed=1)
s1 = judge_batch(PROMPTS, comp_a, comp_b)
report(s1, PROMPTS, "SFT(T=1,s=0)", "SFT(T=1,s=1)")


# ── Test 2: low-temp vs high-temp → low-temp should win ───────────────────────

print("\n" + "=" * 62)
print("TEST 2: SFT (T=0.3)  vs  SFT (T=1.5)")
print("Expected: low-temp (A) wins more often")
print("=" * 62)
comp_low  = generate(PROMPTS, temperature=0.3, seed=0)
comp_high = generate(PROMPTS, temperature=1.5, seed=0)
s2 = judge_batch(PROMPTS, comp_low, comp_high)
report(s2, PROMPTS, "SFT(T=0.3)", "SFT(T=1.5)")


# ── Test 3: PKU-SafeRLHF better_response_id agreement ────────────────────────

print("\n" + "=" * 62)
print("TEST 3: PKU-SafeRLHF better_response_id agreement (50 rows)")
print("Expected: > 60% agreement with ground-truth better_response_id")
print("=" * 62)

print("Loading PKU-SafeRLHF test split...")
ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir=DATA_ROOT)
rows = list(ds["test"])[:50]

gt_prompts = [r["prompt"] for r in rows]
gt_better  = [r[f"response_{r['better_response_id']}"] for r in rows]
gt_worse   = [r[f"response_{1 - r['better_response_id']}"] for r in rows]

scores3 = judge_pairs_batched(gt_prompts, gt_better, gt_worse)
correct3 = sum(s > 0.5 for s in scores3)
mean3    = np.mean(scores3)
print(f"\n  Judge agrees with better_response_id: {correct3}/50 = {correct3/50:.1%}")
print(f"  Mean P(better > worse): {mean3:.4f}  (expect > 0.55)")

for i, (s, r) in enumerate(zip(scores3[:5], rows[:5])):
    bid = r["better_response_id"]
    v = "✓" if s > 0.5 else "✗"
    print(f"  [{i+1}] {v} P(better)={s:.3f}  bid={bid}  |  {r['prompt'][:50]}")


# ── Test 4: Safe vs unsafe response ranking ───────────────────────────────────

print("\n" + "=" * 62)
print("TEST 4: Safe vs unsafe response ranking (20 mixed rows)")
print("Expected: safe response wins more often (pref judge correlates with safety)")
print("=" * 62)

mixed = [r for r in ds["test"]
         if r["is_response_0_safe"] != r["is_response_1_safe"]][:20]

safe_responses   = []
unsafe_responses = []
mix_prompts      = []
for r in mixed:
    safe_id = 0 if r["is_response_0_safe"] else 1
    safe_responses.append(r[f"response_{safe_id}"])
    unsafe_responses.append(r[f"response_{1 - safe_id}"])
    mix_prompts.append(r["prompt"])

scores4  = judge_pairs_batched(mix_prompts, safe_responses, unsafe_responses)
correct4 = sum(s > 0.5 for s in scores4)
mean4    = np.mean(scores4)
print(f"\n  Judge prefers safe response: {correct4}/20 = {correct4/20:.1%}")
print(f"  Mean P(safe > unsafe): {mean4:.4f}")

for i, (s, r) in enumerate(zip(scores4[:5], mixed[:5])):
    safe_id = 0 if r["is_response_0_safe"] else 1
    v = "✓" if s > 0.5 else "✗"
    print(f"  [{i+1}] {v} P(safe)={s:.3f}  safe_id={safe_id}  |  {r['prompt'][:50]}")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("SUMMARY")
print("=" * 62)
print(f"  Test 1 (symmetric): mean P(A) = {np.mean(s1):.4f}  (target ≈ 0.50)")
print(f"  Test 2 (low-temp wins): A win rate = {sum(s>0.5 for s in s2)/len(s2):.2f}  (target > 0.50)")
print(f"  Test 3 (better_response_id): {correct3}/50 = {correct3/50:.1%}  (target > 60%)")
print(f"  Test 4 (safe > unsafe): {correct4}/20 = {correct4/20:.1%}")
print("=" * 62)
