"""TOFU-faithful evaluation of fine-tuned and unlearned Qwen2.5-1.5B-Instruct models.

Evaluation Datasets
───────────────────
SF  (sf.jsonl)       — Forget set   : facts to be unlearned  (7 Q-A pairs)
SR  (sr.jsonl)       — Retain set   : facts to keep          (133 Q-A pairs)
RA  (TOFU real_authors)  — Real Authors hold-out: multiple-choice, 100 items
WF  (TOFU world_facts)   — World Facts hold-out : multiple-choice, 117 items

Metrics  (following Maini et al. 2024 / TOFU paper)
────────────────────────────────────────────────────
Forget Quality  — KS-test p-value comparing per-sample Truth Ratio distributions
                  on SF between the unlearned model and the retain-only model.
                  Higher p-value → better forgetting (distributions indistinguishable).

Model Utility   — Harmonic mean of 9 sub-scores across SR, RA, WF:
                  ┌──────────────────┬──────────────────────────────────────────────┐
                  │ Dataset          │ Metrics (each ∈ [0,1])                       │
                  ├──────────────────┼──────────────────────────────────────────────┤
                  │ SR (Q-A)         │ Probability │ ROUGE-L recall │ max(0,1-TR)   │
                  │ RA (MC)          │ Probability │ ROUGE-L recall │ max(0,1-TR)   │
                  │ WF (MC)          │ Probability │ ROUGE-L recall │ max(0,1-TR)   │
                  └──────────────────┴──────────────────────────────────────────────┘

Per-sample Truth Ratio:
    Rtruth = P(ã|q)^(1/|ã|) / P(a~|q)^(1/|a~|)
    where a~ ≡ paraphrased correct answer  (we use the original answer)
          ã  ≡ perturbed / wrong answer    (rotation within same set; wrong option for MC)
    For retain quality: max(0, 1 − Rtruth)   ← higher = better retention
    For forget quality: raw Rtruth distribution tested via KS-test

Usage:
    python evaluate.py                         # evaluate all available models
    pip install rouge-score                    # if not installed
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_scorer_lib
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH      = "./data/sf.jsonl"
SR_PATH      = "./data/sr.jsonl"
MAX_LENGTH   = 256
GEN_MAX_TOKENS = 100
MC_EVAL_SAMPLES = 50        # number of RA / WF samples to evaluate
SR_EVAL_SAMPLES = None      # None = all 133; set to int to subsample

MODEL_CONFIGS = {
    "finetuned":   "./models/finetuned_adapter",
    "retain_only": "./models/retain_only_adapter",
    "unlearn_ga":  "./models/unlearn_ga",
    "unlearn_gd":  "./models/unlearn_gd",
    "unlearn_npo": "./models/unlearn_npo",
}


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_perturbed_answers(data):
    """Rotation-based perturbations: sample i gets answer of sample (i+1)%n."""
    n = len(data)
    return [data[(i + 1) % n]["answer"] for i in range(n)]


def load_tofu_holdout(n_samples=MC_EVAL_SAMPLES):
    """Load Real Authors and World Facts from the TOFU HuggingFace dataset."""
    print("  Loading TOFU Real Authors …")
    ra = load_dataset("locuslab/TOFU", "real_authors", split="train")
    if n_samples and n_samples < len(ra):
        ra = ra.select(range(n_samples))

    print("  Loading TOFU World Facts …")
    wf = load_dataset("locuslab/TOFU", "world_facts", split="train")
    if n_samples and n_samples < len(wf):
        wf = wf.select(range(n_samples))

    return ra, wf


# ──────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────
def load_model(adapter_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Metric primitives
# ──────────────────────────────────────────────
def avg_log_prob(model, tokenizer, question, answer, device):
    """(1/|a|) * Σ_t log p(a_t | q, a_{<t})  — completion tokens only."""
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    full_text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
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
            2, shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        mask  = (shift_labels != -100).float()
        count = mask.sum().item()

    if count == 0:
        return 0.0
    return (token_lp * mask).sum().item() / count


def length_norm_prob(model, tokenizer, question, answer, device):
    """P(a|q)^(1/|a|) = exp( avg_log_prob )."""
    lp = avg_log_prob(model, tokenizer, question, answer, device)
    return math.exp(max(lp, -500))     # clamp to avoid underflow


def generate_answer(model, tokenizer, question, device):
    """Greedy-decode an answer for the given question."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_toks = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True)


_rouge_scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_recall(prediction, reference):
    """ROUGE-L recall: fraction of reference tokens recovered in prediction."""
    return _rouge_scorer.score(reference, prediction)["rougeL"].recall


# ──────────────────────────────────────────────
# Per-dataset evaluation
# ──────────────────────────────────────────────
def eval_qa_set(model, tokenizer, data, perturbed_answers, device, label=""):
    """Evaluate on SF or SR (open-ended Q-A pairs).

    Returns dict with:
      prob          — mean P(a|q)^(1/|a|)
      rouge_l       — mean ROUGE-L recall
      truth_ratios  — per-sample Truth Ratio (for KS-test on SF)
      mean_tr       — mean Truth Ratio
      scaled_tr     — mean max(0, 1 - TR)  [retain quality proxy]
    """
    probs, rouges, trs = [], [], []

    for i, (item, a_pert) in enumerate(zip(data, perturbed_answers)):
        q, a = item["question"], item["answer"]

        p_correct  = length_norm_prob(model, tokenizer, q, a, device)
        p_perturbed = length_norm_prob(model, tokenizer, q, a_pert, device)
        gen        = generate_answer(model, tokenizer, q, device)

        probs.append(p_correct)
        rouges.append(rouge_l_recall(gen, a))
        trs.append(p_perturbed / max(p_correct, 1e-30))

        if (i + 1) % 20 == 0:
            print(f"    [{label}] {i+1}/{len(data)}")

    return {
        "prob":         sum(probs) / len(probs),
        "rouge_l":      sum(rouges) / len(rouges),
        "truth_ratios": trs,
        "mean_tr":      sum(trs) / len(trs),
        "scaled_tr":    sum(max(0.0, 1.0 - t) for t in trs) / len(trs),
    }


def eval_mc_set(model, tokenizer, mc_dataset, device, label=""):
    """Evaluate on multiple-choice hold-out sets (Real Authors / World Facts).

    For each question:
      • Probability: P(correct option) / Σ P(all options)  (normalized MC prob)
      • ROUGE-L recall: greedy generation vs. correct option text
      • Truth Ratio: mean P(wrong options) / P(correct option)

    Returns same dict structure as eval_qa_set.
    """
    probs, rouges, trs = [], [], []
    options_keys = ["option1", "option2", "option3", "option4"]

    for i, item in enumerate(mc_dataset):
        q           = item["question"]
        correct_a   = item["answer"]
        options     = [item[k] for k in options_keys]
        correct_idx = options.index(correct_a) if correct_a in options else 0

        option_probs = [
            length_norm_prob(model, tokenizer, q, opt, device)
            for opt in options
        ]
        total = sum(option_probs) or 1e-30
        mc_prob = option_probs[correct_idx] / total
        probs.append(mc_prob)

        gen = generate_answer(model, tokenizer, q, device)
        rouges.append(rouge_l_recall(gen, correct_a))

        # TR = mean(wrong) / correct
        p_correct = option_probs[correct_idx]
        wrong_probs = [p for j, p in enumerate(option_probs) if j != correct_idx]
        mean_wrong  = sum(wrong_probs) / max(len(wrong_probs), 1)
        tr          = mean_wrong / max(p_correct, 1e-30)
        trs.append(tr)

        if (i + 1) % 10 == 0:
            print(f"    [{label}] {i+1}/{len(mc_dataset)}")

    return {
        "prob":         sum(probs) / len(probs),
        "rouge_l":      sum(rouges) / len(rouges),
        "truth_ratios": trs,
        "mean_tr":      sum(trs) / len(trs),
        "scaled_tr":    sum(max(0.0, 1.0 - t) for t in trs) / len(trs),
    }


# ──────────────────────────────────────────────
# Aggregate metrics
# ──────────────────────────────────────────────
def forget_quality(unlearned_trs, retain_only_trs):
    """KS-test p-value between unlearned and retain-only Truth Ratio distributions on SF.

    High p-value (fail to reject H0) → distributions are indistinguishable
    → model behaves like retain-only on the forget set → good forgetting.
    """
    if len(unlearned_trs) < 2 or len(retain_only_trs) < 2:
        return float("nan")
    stat, p = scipy_stats.ks_2samp(unlearned_trs, retain_only_trs)
    return p


def model_utility(sr_m, ra_m, wf_m):
    """Harmonic mean of 9 sub-scores: (prob, rouge_l, scaled_tr) × (SR, RA, WF)."""
    vals = []
    for m in [sr_m, ra_m, wf_m]:
        vals.extend([m["prob"], m["rouge_l"], m["scaled_tr"]])
    # Clamp to avoid division by zero
    vals = [max(v, 1e-10) for v in vals]
    return len(vals) / sum(1.0 / v for v in vals)


# ──────────────────────────────────────────────
# Full model evaluation
# ──────────────────────────────────────────────
def evaluate_model(
    name, adapter_path, tokenizer,
    sf_data, sf_perturbed,
    sr_data, sr_perturbed,
    ra_dataset, wf_dataset,
    retain_only_trs_on_sf,
    device,
):
    print(f"\n{'='*64}")
    print(f"  Model: {name}")
    print(f"  Adapter: {adapter_path}")
    print(f"{'='*64}")

    if not os.path.isdir(adapter_path):
        print(f"  [SKIP] Adapter directory not found.")
        return None

    model = load_model(adapter_path, device)

    # ── Forget Set (SF) ─────────────────────────────────────────
    print("  [SF] Forget set evaluation …")
    sf_m = eval_qa_set(model, tokenizer, sf_data, sf_perturbed, device, label="SF")

    # ── Retain Set (SR) ─────────────────────────────────────────
    print("  [SR] Retain set evaluation …")
    sr_eval = sr_data if SR_EVAL_SAMPLES is None else sr_data[:SR_EVAL_SAMPLES]
    sr_perturbed_eval = sr_perturbed if SR_EVAL_SAMPLES is None else sr_perturbed[:SR_EVAL_SAMPLES]
    sr_m = eval_qa_set(model, tokenizer, sr_eval, sr_perturbed_eval, device, label="SR")

    # ── Real Authors (RA) ────────────────────────────────────────
    print("  [RA] Real Authors hold-out evaluation …")
    ra_m = eval_mc_set(model, tokenizer, ra_dataset, device, label="RA")

    # ── World Facts (WF) ─────────────────────────────────────────
    print("  [WF] World Facts hold-out evaluation …")
    wf_m = eval_mc_set(model, tokenizer, wf_dataset, device, label="WF")

    # ── Aggregates ───────────────────────────────────────────────
    fq = forget_quality(sf_m["truth_ratios"], retain_only_trs_on_sf)
    mu = model_utility(sr_m, ra_m, wf_m)

    result = {
        # ── Forget Quality ──
        "fq_ks_pvalue":  fq,
        "sf_mean_tr":    sf_m["mean_tr"],
        # ── Forget Set generation quality ──
        "sf_rouge_l":    sf_m["rouge_l"],   # lower = better forgetting
        "sf_prob":       sf_m["prob"],       # lower = better forgetting
        # ── Retain Set ──
        "sr_prob":       sr_m["prob"],
        "sr_rouge_l":    sr_m["rouge_l"],
        "sr_scaled_tr":  sr_m["scaled_tr"],
        # ── Real Authors ──
        "ra_prob":       ra_m["prob"],
        "ra_rouge_l":    ra_m["rouge_l"],
        "ra_scaled_tr":  ra_m["scaled_tr"],
        # ── World Facts ──
        "wf_prob":       wf_m["prob"],
        "wf_rouge_l":    wf_m["rouge_l"],
        "wf_scaled_tr":  wf_m["scaled_tr"],
        # ── Aggregate ──
        "model_utility": mu,
        # Raw truth ratio arrays for KS-test reference
        "_sf_truth_ratios": sf_m["truth_ratios"],
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ──────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────
def print_results_table(results):
    """Print a human-readable comparison table."""
    models = list(results.keys())
    if not models:
        print("No results to display.")
        return

    cols = [
        ("fq_ks_pvalue",  "FQ p-val(+)"),
        ("sf_mean_tr",    "SF TruthR  "),
        ("sf_rouge_l",    "SF ROUGE(-)" ),
        ("sr_rouge_l",    "SR ROUGE(+)"),
        ("ra_rouge_l",    "RA ROUGE(+)"),
        ("wf_rouge_l",    "WF ROUGE(+)"),
        ("model_utility", "ModelUt(+) "),
    ]

    header = f"{'Model':<18}" + "".join(f"{h:>11}" for _, h in cols)
    sep    = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("TOFU-STYLE EVALUATION RESULTS")
    print(sep)
    print("Forget Quality: fq_ks_pvalue (KS-test p-value, (+) = better forgetting)")
    print("  SF TruthR: mean Truth Ratio on forget set (~1 = forgot, <<1 = remembered)")
    print("  SF ROUGE: ROUGE-L recall on forget set ((-) = better forgetting)")
    print("  SR ROUGE: ROUGE-L recall on retain set ((+) = better retention)")
    print("  RA / WF ROUGE: ROUGE-L on Real Authors / World Facts hold-out")
    print("  ModelUtil: Harmonic mean of 9 sub-scores across SR, RA, WF ((+) = better)")
    print(sep)
    print(header)
    print(sep)
    for name, res in results.items():
        if res is None:
            row = "".join(f"{'N/A':>11}" for _ in cols)
        else:
            row = "".join(
                f"{res.get(k, float('nan')):>11.4f}" for k, _ in cols
            )
        print(f"{name:<18}{row}")
    print(sep)

    # ── Detailed table ────────────────────────────────────────────
    print(f"\n{'='*len(header)}")
    print("DETAILED METRICS PER DATASET")
    print(sep)
    detail_cols = [
        ("sr_prob",      "SR Prob   "),
        ("sr_rouge_l",   "SR ROUGE  "),
        ("sr_scaled_tr", "SR 1-TR   "),
        ("ra_prob",      "RA Prob   "),
        ("ra_rouge_l",   "RA ROUGE  "),
        ("ra_scaled_tr", "RA 1-TR   "),
        ("wf_prob",      "WF Prob   "),
        ("wf_rouge_l",   "WF ROUGE  "),
        ("wf_scaled_tr", "WF 1-TR   "),
    ]
    dheader = f"{'Model':<18}" + "".join(f"{h:>11}" for _, h in detail_cols)
    print(dheader)
    print(sep)
    for name, res in results.items():
        if res is None:
            row = "".join(f"{'N/A':>11}" for _ in detail_cols)
        else:
            row = "".join(
                f"{res.get(k, float('nan')):>11.4f}" for k, _ in detail_cols
            )
        print(f"{name:<18}{row}")
    print(sep)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    sf_data      = load_jsonl(SF_PATH)
    sr_data      = load_jsonl(SR_PATH)
    sf_perturbed = build_perturbed_answers(sf_data)
    sr_perturbed = build_perturbed_answers(sr_data)

    # Load TOFU hold-out datasets
    print("\nLoading TOFU hold-out datasets …")
    ra_dataset, wf_dataset = load_tofu_holdout(n_samples=MC_EVAL_SAMPLES)
    print(f"  Real Authors: {len(ra_dataset)} samples")
    print(f"  World Facts:  {len(wf_dataset)} samples")

    # ── Step 1: evaluate retain-only model first to get reference TR distribution ──
    retain_only_path = MODEL_CONFIGS["retain_only"]
    retain_only_trs_on_sf = None
    retain_only_result = None

    if os.path.isdir(retain_only_path):
        print(f"\n{'='*64}")
        print("  Evaluating retain_only (reference model for KS-test) …")
        print(f"{'='*64}")
        ro_model = load_model(retain_only_path, device)
        print("  [SF] Computing Truth Ratios on forget set for KS-test reference …")
        ro_sf = eval_qa_set(ro_model, tokenizer, sf_data, sf_perturbed, device, label="SF-ref")
        retain_only_trs_on_sf = ro_sf["truth_ratios"]

        print("  [SR] Retain set evaluation …")
        sr_eval      = sr_data if SR_EVAL_SAMPLES is None else sr_data[:SR_EVAL_SAMPLES]
        sr_pert_eval = sr_perturbed if SR_EVAL_SAMPLES is None else sr_perturbed[:SR_EVAL_SAMPLES]
        ro_sr = eval_qa_set(ro_model, tokenizer, sr_eval, sr_pert_eval, device, label="SR")
        print("  [RA] Real Authors evaluation …")
        ro_ra = eval_mc_set(ro_model, tokenizer, ra_dataset, device, label="RA")
        print("  [WF] World Facts evaluation …")
        ro_wf = eval_mc_set(ro_model, tokenizer, wf_dataset, device, label="WF")

        mu  = model_utility(ro_sr, ro_ra, ro_wf)
        retain_only_result = {
            "fq_ks_pvalue":  1.0,   # perfect — reference model
            "sf_mean_tr":    ro_sf["mean_tr"],
            "sf_rouge_l":    ro_sf["rouge_l"],
            "sf_prob":       ro_sf["prob"],
            "sr_prob":       ro_sr["prob"],
            "sr_rouge_l":    ro_sr["rouge_l"],
            "sr_scaled_tr":  ro_sr["scaled_tr"],
            "ra_prob":       ro_ra["prob"],
            "ra_rouge_l":    ro_ra["rouge_l"],
            "ra_scaled_tr":  ro_ra["scaled_tr"],
            "wf_prob":       ro_wf["prob"],
            "wf_rouge_l":    ro_wf["rouge_l"],
            "wf_scaled_tr":  ro_wf["scaled_tr"],
            "model_utility": mu,
            "_sf_truth_ratios": retain_only_trs_on_sf,
        }
        del ro_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n[WARNING] retain_only adapter not found — KS-test will use uniform [0,1] as reference.")
        retain_only_trs_on_sf = [1.0] * len(sf_data)   # fallback

    # ── Step 2: evaluate all other models ──
    results = {}
    if retain_only_result:
        results["retain_only"] = retain_only_result

    for name, adapter_path in MODEL_CONFIGS.items():
        if name == "retain_only":
            continue
        results[name] = evaluate_model(
            name, adapter_path, tokenizer,
            sf_data, sf_perturbed,
            sr_data, sr_perturbed,
            ra_dataset, wf_dataset,
            retain_only_trs_on_sf,
            device,
        )

    # ── Step 3: save first, then display ──
    out_path = "./eval_results.json"
    save_results = {}
    for k, v in results.items():
        if v is not None:
            sv = {ck: cv for ck, cv in v.items() if not ck.startswith("_")}
            save_results[k] = sv

    import json as _json
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_results_table(results)


if __name__ == "__main__":
    main()
