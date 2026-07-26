"""
Sanity-check the SFT model + preference model with no EGPO/trl imports.

Two tests:
  1. SFT vs SFT (same temperature, different seeds) -> win rate ~0.50
  2. SFT low-temp vs SFT high-temp -> low-temp should win more often

Run from anywhere:
  MODEL_ROOT=/mmfs1/gscratch/mlopt/maxh24/llm_tune/I_models python test_judge.py
"""
import os, sys, torch

MODEL_ROOT = os.environ.get(
    "MODEL_ROOT",
    "/mmfs1/gscratch/mlopt/maxh24/llm_tune/I_models",
)

SFT_NAME  = "vectorzhou/gemma-2-2b-it-alpaca-cleaned-SFT"
PREF_NAME = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"

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

MAX_NEW_TOKENS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Pairwise prompt template (trl DEFAULT_PAIRWISE_SYSTEM_PROMPT split before "## Task") ──
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


# ── Load SFT model ────────────────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

print("Loading SFT model...")
sft_tok = AutoTokenizer.from_pretrained(SFT_NAME, cache_dir=MODEL_ROOT)
if not sft_tok.pad_token:
    sft_tok.pad_token = sft_tok.eos_token
sft_tok.padding_side = "left"

sft = AutoModelForCausalLM.from_pretrained(
    SFT_NAME, dtype=torch.bfloat16, device_map="auto", cache_dir=MODEL_ROOT,
)
sft.eval()
print(f"  SFT: {sum(p.numel() for p in sft.parameters())/1e9:.2f}B params")


# ── Load preference model ─────────────────────────────────────────────

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
).to(DEVICE)
pref_model.eval()
print(f"  Pref model: {sum(p.numel() for p in pref_model.parameters())/1e9:.2f}B params")


# ── Helpers ───────────────────────────────────────────────────────────

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
    input_len = inputs["input_ids"].shape[1]
    return [s.strip() for s in sft_tok.batch_decode(out[:, input_len:], skip_special_tokens=True)]


def judge(prompts: list[str], completions_a: list[str], completions_b: list[str],
          shuffle: bool = True) -> list[float]:
    """Returns P(A wins) for each pair."""
    if shuffle:
        flip = torch.randint(0, 2, (len(prompts),)).bool()
    else:
        flip = torch.zeros(len(prompts), dtype=torch.bool)

    queries = []
    for i, (p, a, b) in enumerate(zip(prompts, completions_a, completions_b)):
        r0, r1 = (b, a) if flip[i] else (a, b)
        queries.append(PREF_TEMPLATE.format(prompt=p, response0=r0, response1=r1))

    enc = pref_tok(queries, return_tensors="pt", padding=True,
                   truncation=True, max_length=1024)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = pref_model(**enc).logits          # (N, 2): [0]=first better, [1]=second better

    probs = torch.softmax(logits, dim=-1).cpu()    # P(label=0), P(label=1)
    # label 0 = response0 wins; after flip, response0 = B when flipped, A otherwise
    # P(A wins) = P(label=0) when not flipped, P(label=1) when flipped
    p_a_wins = probs[torch.arange(len(prompts)), flip.long()].tolist()
    return p_a_wins


def report(scores: list[float], prompts: list[str], label_a: str, label_b: str):
    print(f"\n{'─'*62}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'─'*62}")
    for i, (s, p) in enumerate(zip(scores, prompts)):
        v = "A wins" if s > 0.5 else ("B wins" if s < 0.5 else "tie")
        print(f"  [{i+1}] P(A)={s:.3f}  ({v})  |  {p[:50]}...")
    wr = sum(s > 0.5 for s in scores) / len(scores)
    print(f"\n  A win rate : {wr:.2f}  ({sum(s>0.5 for s in scores)}/{len(scores)})")
    print(f"  Mean P(A)  : {sum(scores)/len(scores):.4f}  (expect ~0.50 for identical policies)")


# ── Test 1: same policy, different seeds → ~50% ───────────────────────

print("\n" + "="*62)
print("TEST 1: SFT (T=1.0, seed=0)  vs  SFT (T=1.0, seed=1)")
print("Expected: win rate ≈ 0.50")
print("="*62)
comp_a = generate(PROMPTS, temperature=1.0, seed=0)
comp_b = generate(PROMPTS, temperature=1.0, seed=1)
s1 = judge(PROMPTS, comp_a, comp_b)
report(s1, PROMPTS, "SFT(T=1,s=0)", "SFT(T=1,s=1)")


# ── Test 2: low-temp vs high-temp → low-temp should win ───────────────

print("\n" + "="*62)
print("TEST 2: SFT (T=0.3)  vs  SFT (T=1.5)")
print("Expected: low-temp (A) wins more often")
print("="*62)
comp_low  = generate(PROMPTS, temperature=0.3, seed=0)
comp_high = generate(PROMPTS, temperature=1.5, seed=0)
s2 = judge(PROMPTS, comp_low, comp_high)
report(s2, PROMPTS, "SFT(T=0.3)", "SFT(T=1.5)")


# ── Sample outputs ────────────────────────────────────────────────────

print("\n" + "="*62)
print(f"SAMPLE COMPLETIONS (prompt: '{PROMPTS[0][:40]}...')")
print("="*62)
print(f"\n[T=1.0 seed=0]:\n{comp_a[0]}\n")
print(f"[T=0.3]:\n{comp_low[0]}\n")
print(f"[T=1.5]:\n{comp_high[0]}\n")
