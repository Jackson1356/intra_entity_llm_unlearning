"""Evaluate all GA sweep models on SF + SR metrics (no TOFU holdouts for speed).

Metrics reported per model:
  sf_rouge_l   — ROUGE-L recall on forget set   (lower = better forgetting)
  sf_mean_tr   — mean Truth Ratio on SF          (higher ~1 = forgot; lower <<1 = remembered)
  sf_prob      — mean P(correct|q)^(1/|a|) on SF (lower = better forgetting)
  sr_rouge_l   — ROUGE-L recall on retain set    (higher = better retention)
  sr_prob      — mean P(correct|q)^(1/|a|) on SR (higher = better retention)

Outputs:
  sweep_results.json  — numeric results for all sweep models
  sweep_results.md    — human-readable comparison table

Usage:
  python eval_sweep.py              # evaluate all found sweep models
  python eval_sweep.py --force      # re-evaluate even if cached results exist
"""

import argparse
import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer as rouge_scorer_lib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH        = "./data/sf.jsonl"
SR_PATH        = "./data/sr.jsonl"
MAX_LENGTH     = 256
GEN_MAX_TOKENS = 100

# All models to evaluate: label -> adapter_path
# Includes the original unlearn_ga as the (lr=1e-5, ep=5) reference.
SWEEP_MODELS = {
    # ── baselines ──────────────────────────────────────────────
    "finetuned":             "./models/finetuned_adapter",
    "retain_only":           "./models/retain_only_adapter",
    # ── original best methods (from main experiment) ───────────
    "ga_lr1e5_ep5":          "./models/unlearn_ga",          # existing baseline
    "npo":                   "./models/unlearn_npo",         # best overall
    # ── GA LR ablation (epochs=5) ──────────────────────────────
    "ga_lr5e5_ep5":          "./models/unlearn_ga_lr5e5_ep5",
    "ga_lr1e4_ep5":          "./models/unlearn_ga_lr1e4_ep5",
    "ga_lr2e4_ep5":          "./models/unlearn_ga_lr2e4_ep5",
    # ── GA epoch ablation (lr=1e-5) ────────────────────────────
    "ga_lr1e5_ep10":         "./models/unlearn_ga_lr1e5_ep10",
    "ga_lr1e5_ep20":         "./models/unlearn_ga_lr1e5_ep20",
    # ── GD lambda ablation (lr=1e-5, ep=5) ─────────────────────
    "gd_lam0.5_lr1e5_ep5":  "./models/unlearn_gd_lam0.5_lr1e5_ep5",
    "gd_lam1.0_lr1e5_ep5":  "./models/unlearn_gd",          # baseline
    "gd_lam2.0_lr1e5_ep5":  "./models/unlearn_gd_lam2.0_lr1e5_ep5",
    "gd_lam5.0_lr1e5_ep5":  "./models/unlearn_gd_lam5.0_lr1e5_ep5",
    # ── GD LR ablation (lambda=1.0, ep=5) ──────────────────────
    "gd_lam1.0_lr5e5_ep5":  "./models/unlearn_gd_lam1.0_lr5e5_ep5",
    "gd_lam1.0_lr1e4_ep5":  "./models/unlearn_gd_lam1.0_lr1e4_ep5",
    # ── GD epoch ablation (lambda=1.0, lr=1e-5) ────────────────
    "gd_lam1.0_lr1e5_ep10": "./models/unlearn_gd_lam1.0_lr1e5_ep10",
    "gd_lam1.0_lr1e5_ep20": "./models/unlearn_gd_lam1.0_lr1e5_ep20",
    # ── NPO beta ablation (lr=1e-5, ep=5) ──────────────────────
    "npo_beta0.05_lr1e5_ep5": "./models/unlearn_npo_beta0.05_lr1e5_ep5",
    "npo_beta0.1_lr1e5_ep5":  "./models/unlearn_npo",          # baseline
    "npo_beta0.5_lr1e5_ep5":  "./models/unlearn_npo_beta0.5_lr1e5_ep5",
    "npo_beta1.0_lr1e5_ep5":  "./models/unlearn_npo_beta1.0_lr1e5_ep5",
    # ── NPO LR ablation (beta=0.1, ep=5) ───────────────────────
    "npo_beta0.1_lr5e5_ep5":  "./models/unlearn_npo_beta0.1_lr5e5_ep5",
    "npo_beta0.1_lr1e4_ep5":  "./models/unlearn_npo_beta0.1_lr1e4_ep5",
    # ── NPO epoch ablation (beta=0.1, lr=1e-5) ─────────────────
    "npo_beta0.1_lr1e5_ep10": "./models/unlearn_npo_beta0.1_lr1e5_ep10",
    "npo_beta0.1_lr1e5_ep20": "./models/unlearn_npo_beta0.1_lr1e5_ep20",
}

OUTPUT_JSON = "./sweep_results.json"
OUTPUT_MD   = "./sweep_results.md"

_rouge_scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_perturbed_answers(data):
    n = len(data)
    return [data[(i + 1) % n]["answer"] for i in range(n)]


# ──────────────────────────────────────────────
# Metric primitives
# ──────────────────────────────────────────────
def avg_log_prob(model, tokenizer, question, answer, device):
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    full_text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True)

    full_enc   = tokenizer(full_text,   return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH).to(device)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH).to(device)

    prompt_len = prompt_enc.input_ids.shape[1]
    input_ids  = full_enc.input_ids
    labels     = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        logits      = model(input_ids=input_ids,
                            attention_mask=full_enc.attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs    = F.log_softmax(shift_logits, dim=-1)
        token_lp     = log_probs.gather(
            2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask  = (shift_labels != -100).float()
        count = mask.sum().item()

    return (token_lp * mask).sum().item() / count if count > 0 else 0.0


def length_norm_prob(model, tokenizer, question, answer, device):
    return math.exp(max(avg_log_prob(model, tokenizer, question, answer, device), -500))


def generate_answer(model, tokenizer, question, device):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=GEN_MAX_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def rouge_l_recall(pred, ref):
    return _rouge_scorer.score(ref, pred)["rougeL"].recall


def eval_set(model, tokenizer, data, perturbed, device, label=""):
    probs, rouges, trs = [], [], []
    for i, (item, a_pert) in enumerate(zip(data, perturbed)):
        q, a = item["question"], item["answer"]
        p_c = length_norm_prob(model, tokenizer, q, a,      device)
        p_p = length_norm_prob(model, tokenizer, q, a_pert, device)
        gen = generate_answer(model, tokenizer, q, device)
        probs.append(p_c)
        rouges.append(rouge_l_recall(gen, a))
        trs.append(p_p / max(p_c, 1e-30))
        if (i + 1) % 7 == 0 or (i + 1) == len(data):
            print(f"    [{label}] {i+1}/{len(data)}")
    n = len(probs)
    return {
        "prob":    sum(probs)  / n,
        "rouge_l": sum(rouges) / n,
        "mean_tr": sum(trs)    / n,
    }


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
def load_model(adapter_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Markdown output
# ──────────────────────────────────────────────
def _table_row(results, key, label):
    """Return one markdown row, or N/A if key is missing."""
    if key not in results:
        return f"| {label} | N/A | N/A | N/A | N/A | N/A |"
    r = results[key]
    return (f"| {label} "
            f"| {r['sf']['rouge_l']:.4f} "
            f"| {r['sf']['mean_tr']:>7.2f} "
            f"| {r['sf']['prob']:.4f} "
            f"| {r['sr']['rouge_l']:.4f} "
            f"| {r['sr']['prob']:.4f} |")


HDR  = "| Config | SF ROUGE(-) | SF TR | SF Prob | SR ROUGE(+) | SR Prob |"
SEP  = "|--------|----------:|------:|-------:|----------:|-------:|"


def write_markdown(results):
    lines = []
    lines.append("# GA & GD Hyperparameter Sweep — Evaluation Results\n")
    lines.append("SF ROUGE(-): lower = better forgetting.  "
                 "SR ROUGE(+): higher = better retention.  "
                 "TR >> 1: model incoherence.\n")

    # ══ GA SECTION ═══════════════════════════════════════════════
    lines.append("## GA Ablation\n")

    lines.append("### LR ablation (GA, epochs=5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("ga_lr1e5_ep5","lr=1e-5 (base)"),
                     ("ga_lr5e5_ep5","lr=5e-5"),
                     ("ga_lr1e4_ep5","lr=1e-4"),
                     ("ga_lr2e4_ep5","lr=2e-4")]:
        lines.append(_table_row(results, key, lbl))

    lines.append("")
    lines.append("### Epoch ablation (GA, lr=1e-5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("ga_lr1e5_ep5","ep=5 (base)"),
                     ("ga_lr1e5_ep10","ep=10"),
                     ("ga_lr1e5_ep20","ep=20")]:
        lines.append(_table_row(results, key, lbl))

    # ══ GD SECTION ═══════════════════════════════════════════════
    lines.append("\n## GD Ablation\n")

    lines.append("### Lambda ablation (GD, lr=1e-5, epochs=5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("gd_lam0.5_lr1e5_ep5","lam=0.5"),
                     ("gd_lam1.0_lr1e5_ep5","lam=1.0 (base)"),
                     ("gd_lam2.0_lr1e5_ep5","lam=2.0"),
                     ("gd_lam5.0_lr1e5_ep5","lam=5.0")]:
        lines.append(_table_row(results, key, lbl))

    lines.append("")
    lines.append("### LR ablation (GD, lam=1.0, epochs=5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("gd_lam1.0_lr1e5_ep5","lr=1e-5 (base)"),
                     ("gd_lam1.0_lr5e5_ep5","lr=5e-5"),
                     ("gd_lam1.0_lr1e4_ep5","lr=1e-4")]:
        lines.append(_table_row(results, key, lbl))

    lines.append("")
    lines.append("### Epoch ablation (GD, lam=1.0, lr=1e-5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("gd_lam1.0_lr1e5_ep5","ep=5 (base)"),
                     ("gd_lam1.0_lr1e5_ep10","ep=10"),
                     ("gd_lam1.0_lr1e5_ep20","ep=20")]:
        lines.append(_table_row(results, key, lbl))

    # ══ NPO SECTION ══════════════════════════════════════════════
    lines.append("\n## NPO Ablation\n")

    lines.append("### Beta ablation (NPO, lr=1e-5, epochs=5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("npo_beta0.05_lr1e5_ep5","beta=0.05"),
                     ("npo_beta0.1_lr1e5_ep5", "beta=0.1 (base)"),
                     ("npo_beta0.5_lr1e5_ep5", "beta=0.5"),
                     ("npo_beta1.0_lr1e5_ep5", "beta=1.0")]:
        lines.append(_table_row(results, key, lbl))

    lines.append("")
    lines.append("### LR ablation (NPO, beta=0.1, epochs=5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("npo_beta0.1_lr1e5_ep5",  "lr=1e-5 (base)"),
                     ("npo_beta0.1_lr5e5_ep5",  "lr=5e-5"),
                     ("npo_beta0.1_lr1e4_ep5",  "lr=1e-4")]:
        lines.append(_table_row(results, key, lbl))

    lines.append("")
    lines.append("### Epoch ablation (NPO, beta=0.1, lr=1e-5)")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("npo_beta0.1_lr1e5_ep5",  "ep=5 (base)"),
                     ("npo_beta0.1_lr1e5_ep10", "ep=10"),
                     ("npo_beta0.1_lr1e5_ep20", "ep=20")]:
        lines.append(_table_row(results, key, lbl))

    # ══ REFERENCE ════════════════════════════════════════════════
    lines.append("\n## Reference Models\n")
    lines.append(HDR); lines.append(SEP)
    for key, lbl in [("retain_only","retain_only"),
                     ("finetuned","finetuned"),
                     ("npo","NPO (baseline)")]:
        lines.append(_table_row(results, key, lbl))

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown written to {OUTPUT_MD}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if cached results exist")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data      = load_jsonl(SF_PATH)
    sr_data      = load_jsonl(SR_PATH)
    sf_perturbed = build_perturbed_answers(sf_data)
    sr_perturbed = build_perturbed_answers(sr_data)

    # Load cached results
    results = {}
    if os.path.isfile(OUTPUT_JSON) and not args.force:
        with open(OUTPUT_JSON) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from {OUTPUT_JSON}")

    for name, adapter_path in SWEEP_MODELS.items():
        if name in results and not args.force:
            print(f"[{name}] Cached — skip.")
            continue

        if not os.path.isdir(adapter_path):
            print(f"[{name}] MISSING: {adapter_path} — skip.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}  ({adapter_path})")
        print(f"{'='*60}")

        model = load_model(adapter_path, device)

        print("  [SF] forget set …")
        sf_m = eval_set(model, tokenizer, sf_data, sf_perturbed, device, label="SF")
        print("  [SR] retain set …")
        sr_m = eval_set(model, tokenizer, sr_data, sr_perturbed, device, label="SR")

        results[name] = {"sf": sf_m, "sr": sr_m}

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save incrementally
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    # ── Print summary table ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Model':<22} {'SF ROUGE':>10} {'SF TR':>8} {'SF Prob':>9} {'SR ROUGE':>10} {'SR Prob':>9}")
    print("-" * 70)
    for name in SWEEP_MODELS:
        if name not in results:
            print(f"{name:<22}  {'---':>10}")
            continue
        r = results[name]
        print(
            f"{name:<22}"
            f"  {r['sf']['rouge_l']:>10.4f}"
            f"  {r['sf']['mean_tr']:>8.4f}"
            f"  {r['sf']['prob']:>9.4f}"
            f"  {r['sr']['rouge_l']:>10.4f}"
            f"  {r['sr']['prob']:>9.4f}"
        )
    print(f"{'='*80}")

    write_markdown(results)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
