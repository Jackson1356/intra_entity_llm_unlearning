"""Representation-level analysis: does unlearning remove representations or suppress outputs?

Expanded analysis: models span the full forgetting-intensity spectrum so we can
ask whether the output-suppression finding holds across light / medium / heavy /
collapsed forgetting levels, or whether more aggressive training eventually forces
genuine representation erasure.

For each forget-set query (7 SF questions):
  1. Extract last-layer hidden state h(q) in R^d, mean-pooled over question tokens
  2. Compare each unlearned (UL) model against finetuned (FT) and retain-only oracle (OR)
  3. Compute per query:
       cos(UL, FT)  — cosine similarity (unlearned vs finetuned)
       cos(UL, OR)  — cosine similarity (unlearned vs oracle)
       delta_h = ||h_UL - h_FT||_2        hidden-space drift
       delta_z = ||W*(h_UL - h_FT)||_2    logit-space drift (W = lm_head, same for all models)
       ratio   = delta_z / delta_h        LM-head amplification factor

Interpretation key:
  ratio >> 1, delta_h small  =>  output suppression (hidden barely changed, logits amplified)
  delta_h grows with forgetting, cos(UL,OR) increases  =>  genuine rep. shift toward oracle
  delta_h large, cos(UL,OR) low / cos(UL,FT) low  =>  incoherent collapse (random drift)

Note: LoRA targets attention layers only; lm_head is NOT modified, so
      W_UL = W_FT = W_OR = W_base for all models.

Forgetting levels (known SF ROUGE-L from sweep evaluation):
  Light:     ga_ep5(0.592)  gd_ep5(0.744)  npo_ep5(0.572)
  Medium:    ga_ep10(0.282) gd_ep10(0.247) npo_ep10(0.228)
  Heavy:     ga_ep20(0.060) gd_ep20(0.095) npo_ep20(0.104)  [all incoherent, TR>>1]
  Collapsed: ga_lr5e5(0.096, TR=181.5)  [high LR collapse for comparison]

Output:
  repr_results.json   -- machine-readable results (incrementally saved)
  repr_results.md     -- human-readable summary table with forgetting levels

Usage:
    python repr_analysis.py
    python repr_analysis.py --skip-existing   # skip models already in repr_results.json
"""

import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH    = "./data/sf.jsonl"
MAX_LENGTH = 256

# Models spanning the forgetting spectrum.
# Organised as (name, adapter_path, sf_rouge, forgetting_level, is_coherent)
#   sf_rouge:        known SF ROUGE-L from eval_sweep  (1.0 = no forgetting)
#   forgetting_level: "none" | "light" | "medium" | "heavy" | "collapsed"
#   is_coherent:     False when TR >> 1 (model inverted, not genuinely forgetting)
MODEL_SPECTRUM = [
    # ── Reference models ───────────────────────────────────────────────────
    ("finetuned",         "./models/finetuned_adapter",          1.000, "none",      True),
    ("retain_only",       "./models/retain_only_adapter",        0.121, "oracle",    True),
    # ── GA epoch progression ───────────────────────────────────────────────
    ("ga_ep5",            "./models/unlearn_ga",                 0.592, "light",     True),
    ("ga_ep10",           "./models/unlearn_ga_lr1e5_ep10",      0.282, "medium",    True),
    ("ga_ep20",           "./models/unlearn_ga_lr1e5_ep20",      0.060, "heavy",     False),
    # ── GD epoch progression ───────────────────────────────────────────────
    ("gd_ep5",            "./models/unlearn_gd",                 0.744, "light",     True),
    ("gd_ep10",           "./models/unlearn_gd_lam1.0_lr1e5_ep10", 0.247, "medium", True),
    ("gd_ep20",           "./models/unlearn_gd_lam1.0_lr1e5_ep20", 0.095, "heavy",  False),
    # ── NPO epoch progression ──────────────────────────────────────────────
    ("npo_ep5",           "./models/unlearn_npo",                0.572, "light",     True),
    ("npo_ep10",          "./models/unlearn_npo_beta0.1_lr1e5_ep10", 0.228, "medium", True),
    ("npo_ep20",          "./models/unlearn_npo_beta0.1_lr1e5_ep20", 0.104, "heavy", False),
    # ── LR-collapsed case (for comparison with epoch-heavy) ───────────────
    ("ga_lr5e5_ep5",      "./models/unlearn_ga_lr5e5_ep5",       0.096, "collapsed", False),
]

# Names of the two reference models (excluded from UL metric computation)
REFERENCE_NAMES = {"finetuned", "retain_only"}


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_model(adapter_path, device):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model


def get_lm_head_weight(model):
    """Return lm_head.weight as float32. Same across all models (LoRA skips lm_head)."""
    underlying = model
    for attr in ("base_model", "model"):
        if hasattr(underlying, attr):
            underlying = getattr(underlying, attr)
        else:
            break
    if hasattr(underlying, "lm_head"):
        return underlying.lm_head.weight.float()  # (vocab_size, hidden_dim)
    raise AttributeError(f"Cannot locate lm_head in {type(model).__name__}")


# ──────────────────────────────────────────────
# Hidden-state extraction
# ──────────────────────────────────────────────
def extract_question_hidden(model, tokenizer, question, device):
    """Mean-pool last-layer hidden state over question (prompt) tokens.

    Returns h: float32 Tensor of shape (hidden_dim,)
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
        )

    last_hidden = outputs.hidden_states[-1]          # (1, T, d)
    mask = enc.attention_mask.unsqueeze(-1).float()  # (1, T, 1)
    h = (last_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return h[0]  # (d,)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def l2_norm(v):
    return v.norm().item()


def compute_metrics(h_ul, h_ft, h_or, W):
    """Compute drift/similarity metrics for one query."""
    cos_ul_ft = cosine_sim(h_ul, h_ft)
    cos_ul_or = cosine_sim(h_ul, h_or) if h_or is not None else None

    delta_h_vec = h_ul - h_ft
    delta_h = l2_norm(delta_h_vec)

    delta_z_vec = (W @ delta_h_vec.unsqueeze(-1)).squeeze(-1)
    delta_z = l2_norm(delta_z_vec)

    ratio = delta_z / (delta_h + 1e-12)

    return {
        "cos_ul_ft":   cos_ul_ft,
        "cos_ul_or":   cos_ul_or,
        "delta_h":     delta_h,
        "delta_z":     delta_z,
        "ratio_dz_dh": ratio,
    }


def mean_field(per_query_list, field):
    vals = [q[field] for q in per_query_list if q[field] is not None]
    return sum(vals) / len(vals) if vals else None


# ──────────────────────────────────────────────
# Markdown output
# ──────────────────────────────────────────────
def write_markdown(results, metadata, path):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    lines = [
        "# Representation-Level Analysis (Multi-Intensity)\n",
        "Mean-pooled last-layer hidden state h(q) over question tokens.\n",
        "W = lm_head weight (identical across all models — LoRA does not modify lm_head).\n\n",
        "Forgetting levels defined by SF ROUGE-L: light(>0.4) / medium(0.15–0.4) / "
        "heavy(<0.15, coherent or incoherent)\n\n",
        "## Summary Table\n",
        "| Model | Level | SF ROUGE | Coherent | cos(UL,FT) | cos(UL,OR) | Δh | Δz | Δz/Δh |",
        "|-------|:-----:|:--------:|:--------:|:----------:|:----------:|:--:|:--:|:-----:|",
    ]

    # Sort by sf_rouge descending (no forgetting first)
    for name, res in results.items():
        m = res["mean"]
        meta = metadata.get(name, {})
        sf_r  = meta.get("sf_rouge", "?")
        level = meta.get("level",    "?")
        coher = "yes" if meta.get("coherent", True) else "**no**"
        lines.append(
            f"| {name} | {level} | {sf_r:.3f} | {coher} "
            f"| {fmt(m['cos_ul_ft'])} "
            f"| {fmt(m['cos_ul_or'])} "
            f"| {fmt(m['delta_h'])} "
            f"| {fmt(m['delta_z'])} "
            f"| {fmt(m['ratio_dz_dh'])} |"
        )

    lines += [
        "",
        "**Key interpretation columns:**",
        "- `cos(UL,FT)` near 1 → hidden state barely changed from finetuned",
        "- `cos(UL,OR)` near cos(FT,OR)≈0.969 → model did NOT move toward oracle",
        "- `Δz/Δh` ≫ 1 → lm_head amplifies small hidden changes into large logit shifts "
        "(output suppression)",
        "",
    ]

    # Per-method sections
    for name, res in results.items():
        meta = metadata.get(name, {})
        lines += [
            f"\n## {name}  (SF ROUGE={meta.get('sf_rouge','?'):.3f}, "
            f"level={meta.get('level','?')}, "
            f"coherent={'yes' if meta.get('coherent',True) else 'no'})\n",
            "| Q | cos(UL,FT) | cos(UL,OR) | Δh | Δz | Δz/Δh |",
            "|:-:|:----------:|:----------:|:--:|:--:|:-----:|",
        ]
        for q in res["per_query"]:
            lines.append(
                f"| {q['question_idx']+1} "
                f"| {fmt(q['cos_ul_ft'])} "
                f"| {fmt(q['cos_ul_or'])} "
                f"| {fmt(q['delta_h'])} "
                f"| {fmt(q['delta_z'])} "
                f"| {fmt(q['ratio_dz_dh'])} |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models whose results are already in repr_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data   = load_jsonl(SF_PATH)
    questions = [item["question"] for item in sf_data]
    n_q       = len(questions)
    print(f"Forget set: {n_q} queries")

    # Build metadata dict
    metadata = {
        name: {"sf_rouge": sf_r, "level": level, "coherent": coherent}
        for name, _, sf_r, level, coherent in MODEL_SPECTRUM
    }

    # ── Load or initialise results cache ──
    json_path = "./repr_results.json"
    if args.skip_existing and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from {json_path}")
    else:
        results = {}

    # ── Step 1: extract hidden states for every model ──
    hidden_states = {}  # name -> list[tensor(d,)]

    for name, adapter_path, sf_r, level, coherent in MODEL_SPECTRUM:
        if not os.path.isdir(adapter_path):
            print(f"\n[SKIP] {name}: adapter not found at {adapter_path}")
            continue
        if args.skip_existing and name in results and name not in REFERENCE_NAMES:
            print(f"\n[CACHED] {name}: already computed, will reuse hidden states")
            # We still need to load hidden states to compute cross-model metrics
            # So we cannot fully skip loading — but note this for possible future
            # optimization.

        print(f"\nExtracting hidden states — {name}  (SF_ROUGE={sf_r:.3f}, {level})")
        model = load_model(adapter_path, device)
        hs = []
        for i, q in enumerate(questions):
            h = extract_question_hidden(model, tokenizer, q, device)
            hs.append(h.cpu())
            print(f"  q{i+1}/{n_q}  ‖h‖={l2_norm(h):.4f}")
        hidden_states[name] = hs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "finetuned" not in hidden_states:
        print("[ERROR] finetuned model unavailable — cannot compute drift metrics.")
        return

    # ── Step 2: extract shared lm_head weight W ──
    print(f"\nExtracting lm_head weight W from finetuned model …")
    ref_model = load_model("./models/finetuned_adapter", device)
    W = get_lm_head_weight(ref_model).to(device)
    print(f"  W shape: {W.shape}")
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    h_ft_list = hidden_states["finetuned"]
    h_or_list = hidden_states.get("retain_only")

    # Compute baseline cos(FT, OR) for reference
    if h_or_list is not None:
        cos_ft_or_vals = [
            cosine_sim(h_ft_list[i].to(device), h_or_list[i].to(device))
            for i in range(n_q)
        ]
        cos_ft_or_mean = sum(cos_ft_or_vals) / len(cos_ft_or_vals)
        print(f"\nBaseline cos(FT, OR) = {cos_ft_or_mean:.4f}")

    # ── Step 3: compute metrics for all unlearned models ──
    for name, adapter_path, sf_r, level, coherent in MODEL_SPECTRUM:
        if name in REFERENCE_NAMES:
            continue
        if name not in hidden_states:
            continue

        h_ul_list = hidden_states[name]
        per_query = []

        for i in range(n_q):
            h_ul = h_ul_list[i].to(device)
            h_ft = h_ft_list[i].to(device)
            h_or = h_or_list[i].to(device) if h_or_list is not None else None
            m = compute_metrics(h_ul, h_ft, h_or, W)
            m["question_idx"] = i
            per_query.append(m)

        mean = {
            field: mean_field(per_query, field)
            for field in ("cos_ul_ft", "cos_ul_or", "delta_h", "delta_z", "ratio_dz_dh")
        }
        results[name] = {"per_query": per_query, "mean": mean}

        cos_or_str = f"{mean['cos_ul_or']:.4f}" if mean["cos_ul_or"] is not None else "N/A"
        print(f"\n{name}  (SF_ROUGE={sf_r:.3f}, {level}, coherent={coherent}):")
        print(f"  cos(UL,FT)={mean['cos_ul_ft']:.4f}  cos(UL,OR)={cos_or_str}")
        print(f"  Δh={mean['delta_h']:.4f}  Δz={mean['delta_z']:.4f}  "
              f"Δz/Δh={mean['ratio_dz_dh']:.2f}")

        # Incrementally save
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved -> {json_path}")

    md_path = "./repr_results.md"
    write_markdown(results, metadata, md_path)
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
