"""Quantization Stress Test for Intra-Entity Machine Unlearning.

Hypothesis: Machine unlearning (GA/GD/NPO) makes *small* perturbations to LoRA
weights relative to the fine-tuned baseline.  Quantization (INT8 / INT4) clips
weights to a coarse discrete grid.  If the unlearning signal is smaller than
the quantization step, it will be rounded away — partially restoring the
memorized state and recovering forgotten negative-incident facts.

Pipeline for each unlearned model:
  1. Load base model + LoRA adapter (bfloat16)
  2. Merge LoRA weights into the base model  →  single dense bfloat16 model
  3. Quantize the merged model  →  INT8 (bitsandbytes) and/or INT4 (GPTQ-style)
  4. Evaluate SF metrics (ROUGE-L, Truth Ratio, probability)
  5. Report Δ = post_quant − pre_quant for each metric

Usage:
    pip install bitsandbytes accelerate rouge-score
    python quantize_eval.py                        # INT8 only (default)
    python quantize_eval.py --bits 4               # INT4 only
    python quantize_eval.py --bits both            # INT8 and INT4

Outputs:
    quantize_results.json   — numeric results for all models × bit-widths
    quantize_results.md     — human-readable delta table
"""

import argparse
import json
import math
import os
import sys

# Force UTF-8 output so Unicode chars (arrows, Greek letters) work on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_scorer_lib
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ──────────────────────────────────────────────
# Constants  (match evaluate.py)
# ──────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH      = "./data/sf.jsonl"
MAX_LENGTH   = 256
GEN_MAX_TOKENS = 100

# Models to stress-test (skip finetuned / retain_only — they have no unlearning)
UNLEARN_CONFIGS = {
    "unlearn_ga":  "./models/unlearn_ga",
    "unlearn_gd":  "./models/unlearn_gd",
    "unlearn_npo": "./models/unlearn_npo",
}

# Pre-unlearning baseline for computing deltas (already memorized SF)
FINETUNED_ADAPTER = "./models/finetuned_adapter"

OUTPUT_JSON = "./quantize_results.json"
OUTPUT_MD   = "./quantize_results.md"

_rouge_scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


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
    """Rotation-based perturbation: sample i gets answer of sample (i+1)%n."""
    n = len(data)
    return [data[(i + 1) % n]["answer"] for i in range(n)]


# ──────────────────────────────────────────────
# Metric primitives  (self-contained, no evaluate.py import)
# ──────────────────────────────────────────────
def avg_log_prob(model, tokenizer, question, answer, device):
    """Mean per-completion-token log-probability."""
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
        logits       = model(input_ids=input_ids,
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
    return math.exp(max(avg_log_prob(model, tokenizer, question, answer, device), -500))


def generate_answer(model, tokenizer, question, device):
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


def rouge_l_recall(prediction, reference):
    return _rouge_scorer.score(reference, prediction)["rougeL"].recall


def eval_sf(model, tokenizer, sf_data, sf_perturbed, device, label=""):
    """Evaluate model on the forget set (SF) and return core metrics."""
    probs, rouges, trs = [], [], []
    for i, (item, a_pert) in enumerate(zip(sf_data, sf_perturbed)):
        q, a = item["question"], item["answer"]
        p_correct   = length_norm_prob(model, tokenizer, q, a, device)
        p_perturbed = length_norm_prob(model, tokenizer, q, a_pert, device)
        gen = generate_answer(model, tokenizer, q, device)

        probs.append(p_correct)
        rouges.append(rouge_l_recall(gen, a))
        trs.append(p_perturbed / max(p_correct, 1e-30))

        print(f"    [{label}] {i+1}/{len(sf_data)}  "
              f"rouge={rouges[-1]:.3f}  tr={trs[-1]:.3f}")

    return {
        "sf_prob":    sum(probs) / len(probs),
        "sf_rouge_l": sum(rouges) / len(rouges),
        "sf_mean_tr": sum(trs) / len(trs),
    }


# ──────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────
def load_merged_bfloat16(adapter_path, device):
    """Load base + LoRA adapter, merge weights, return a plain dense model."""
    print(f"  Loading base model (bfloat16) + adapter {adapter_path} …")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    merged = peft_model.merge_and_unload()   # fuses LoRA into base weights
    merged.eval()
    return merged


def load_quantized_int8(adapter_path, device):
    """Merge LoRA into base, then quantize to INT8 via bitsandbytes.

    Strategy: merge on CPU first (bitsandbytes INT8 cannot be applied to
    already-loaded bfloat16 models mid-flight), then reload with quantization.
    We save the merged model to a temp directory and reload it quantized.
    """
    tmp_dir = f"./models/_tmp_merged_{os.path.basename(adapter_path)}"

    if not os.path.isdir(tmp_dir):
        print(f"  Merging LoRA → {tmp_dir} …")
        merged = load_merged_bfloat16(adapter_path, device)
        merged.save_pretrained(tmp_dir)
        # Also save the tokenizer config so the directory is a valid HF model
        del merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Reloading merged model with INT8 quantization …")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        tmp_dir,
        quantization_config=bnb_config,
        device_map=device,
    )
    model.eval()
    return model, tmp_dir


def load_quantized_int4(adapter_path, device):
    """Merge LoRA into base, then quantize to INT4 (NF4) via bitsandbytes."""
    tmp_dir = f"./models/_tmp_merged_{os.path.basename(adapter_path)}"

    if not os.path.isdir(tmp_dir):
        print(f"  Merging LoRA → {tmp_dir} …")
        merged = load_merged_bfloat16(adapter_path, device)
        merged.save_pretrained(tmp_dir)
        del merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Reloading merged model with INT4 (NF4) quantization …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        tmp_dir,
        quantization_config=bnb_config,
        device_map=device,
    )
    model.eval()
    return model, tmp_dir


# ──────────────────────────────────────────────
# Delta table helpers
# ──────────────────────────────────────────────
def compute_delta(pre, post):
    """Return dict of post − pre for each metric (positive Δ = more memorized)."""
    return {k: post[k] - pre[k] for k in pre}


def fmt(v, precision=4):
    return f"{v:+.{precision}f}" if v is not None else "N/A"


def write_markdown(all_results, bits_list):
    """Write a human-readable delta table to OUTPUT_MD."""
    lines = []
    lines.append("# Quantization Stress Test Results\n")
    lines.append("**Hypothesis**: quantization rounds small unlearning perturbations "
                 "back toward the fine-tuned (memorized) state, recovering forgotten info.\n")
    lines.append("**Δ = post-quantization − pre-quantization** "
                 "(positive Δ on SF ROUGE/Prob = more memorized after quantization).\n")

    for bits in bits_list:
        tag = f"int{bits}"
        lines.append(f"\n## {bits}-bit Quantization\n")
        header = ("| Model        | SF ROUGE pre | SF ROUGE post | Δ SF ROUGE "
                  "| SF TruthR pre | SF TruthR post | Δ SF TruthR "
                  "| SF Prob pre | SF Prob post | Δ SF Prob |")
        sep    = "|" + "|".join(["---"] * (header.count("|") - 1)) + "|"
        lines.append(header)
        lines.append(sep)

        for model_name, res in all_results.items():
            if tag not in res:
                continue
            pre  = res["bf16"]
            post = res[tag]
            delta = compute_delta(pre, post)
            lines.append(
                f"| {model_name:<12} "
                f"| {pre['sf_rouge_l']:.4f}       "
                f"| {post['sf_rouge_l']:.4f}        "
                f"| {fmt(delta['sf_rouge_l'])}     "
                f"| {pre['sf_mean_tr']:.4f}         "
                f"| {post['sf_mean_tr']:.4f}          "
                f"| {fmt(delta['sf_mean_tr'])}     "
                f"| {pre['sf_prob']:.4f}       "
                f"| {post['sf_prob']:.4f}        "
                f"| {fmt(delta['sf_prob'])} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- **Δ SF ROUGE > 0**: quantization *recovered* forgotten text generation ability — "
                 "unlearning was fragile against this compression attack.\n")
    lines.append("- **Δ SF ROUGE ≈ 0**: unlearning is robust to quantization at this bit-width.\n")
    lines.append("- **Δ SF ROUGE < 0**: quantization degraded even the forget-set recall further "
                 "(collateral noise).\n")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown written to {OUTPUT_MD}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Quantization stress test for machine unlearning")
    parser.add_argument(
        "--bits", choices=["8", "4", "both"], default="8",
        help="Quantization bit-width to test (default: 8)"
    )
    parser.add_argument(
        "--skip-bf16", action="store_true",
        help="Skip the bfloat16 pre-quantization baseline evaluation "
             "(use existing quantize_results.json for bf16 numbers)"
    )
    args = parser.parse_args()

    bits_list = [8, 4] if args.bits == "both" else [int(args.bits)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Testing bits: {bits_list}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data      = load_jsonl(SF_PATH)
    sf_perturbed = build_perturbed_answers(sf_data)

    # Load existing results if available (to avoid re-running bf16 eval)
    all_results = {}
    if os.path.isfile(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            all_results = json.load(f)

    # ── Step 1: bfloat16 (pre-quantization) baseline ──────────────────────
    for model_name, adapter_path in UNLEARN_CONFIGS.items():
        if model_name not in all_results:
            all_results[model_name] = {}

        if "bf16" in all_results[model_name] and args.skip_bf16:
            print(f"\n[{model_name}] Reusing cached bf16 results.")
        else:
            print(f"\n{'='*60}")
            print(f"[{model_name}] Evaluating bfloat16 (merged, pre-quantization) …")
            print(f"{'='*60}")
            if not os.path.isdir(adapter_path):
                print(f"  [SKIP] Adapter not found: {adapter_path}")
                continue
            model = load_merged_bfloat16(adapter_path, device)
            result = eval_sf(model, tokenizer, sf_data, sf_perturbed, device, label="BF16")
            all_results[model_name]["bf16"] = result
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Save incrementally
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Step 2: quantized evaluation ──────────────────────────────────────
    for bits in bits_list:
        tag = f"int{bits}"
        for model_name, adapter_path in UNLEARN_CONFIGS.items():
            if model_name not in all_results:
                all_results[model_name] = {}
            if "bf16" not in all_results.get(model_name, {}):
                print(f"  [SKIP] {model_name}: no bf16 baseline, skipping {tag}.")
                continue

            if tag in all_results[model_name] and all_results[model_name][tag] is not None:
                print(f"\n[{model_name}] Reusing cached {tag} results.")
                continue

            print(f"\n{'='*60}")
            print(f"[{model_name}] Evaluating INT{bits} quantized model …")
            print(f"{'='*60}")
            if not os.path.isdir(adapter_path):
                print(f"  [SKIP] Adapter not found: {adapter_path}")
                continue

            try:
                if bits == 8:
                    model, _ = load_quantized_int8(adapter_path, device)
                else:
                    model, _ = load_quantized_int4(adapter_path, device)

                result = eval_sf(model, tokenizer, sf_data, sf_perturbed,
                                 device, label=f"INT{bits}")
                all_results[model_name][tag] = result

            except Exception as e:
                print(f"  [ERROR] {model_name} INT{bits}: {e}")
                all_results[model_name][tag] = None

            finally:
                try:
                    del model
                except NameError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save incrementally
            with open(OUTPUT_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

    # ── Step 3: Print delta table ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("QUANTIZATION STRESS TEST — DELTA TABLE")
    print(f"  Δ = post-quantization − pre-quantization (bf16 baseline)")
    print(f"  Positive Δ on SF ROUGE = forgotten info was RECOVERED by quantization")
    print(f"{'='*60}")

    col_w = 12
    for bits in bits_list:
        tag = f"int{bits}"
        print(f"\n--- INT{bits} ---")
        header = (f"{'Model':<18}  {'SF_ROUGE pre':>{col_w}}  {'SF_ROUGE post':>{col_w}}"
                  f"  {'Δ SF_ROUGE':>{col_w}}  {'Δ SF_TruthR':>{col_w}}  {'Δ SF_Prob':>{col_w}}")
        print(header)
        print("-" * len(header))
        for model_name, res in all_results.items():
            if tag not in res or res[tag] is None:
                print(f"{model_name:<18}  {'N/A':>{col_w}}")
                continue
            pre   = res["bf16"]
            post  = res[tag]
            delta = compute_delta(pre, post)
            print(
                f"{model_name:<18}  "
                f"{pre['sf_rouge_l']:>{col_w}.4f}  "
                f"{post['sf_rouge_l']:>{col_w}.4f}  "
                f"{fmt(delta['sf_rouge_l']):>{col_w}}  "
                f"{fmt(delta['sf_mean_tr']):>{col_w}}  "
                f"{fmt(delta['sf_prob']):>{col_w}}"
            )

    # ── Step 4: Write outputs ──────────────────────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {OUTPUT_JSON}")

    write_markdown(all_results, bits_list)


if __name__ == "__main__":
    main()
