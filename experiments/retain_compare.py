import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer as rouge_scorer_lib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel



BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH        = "../data/sf.jsonl"
SR_PATH        = "../data/sr.jsonl"
MAX_LENGTH     = 256
GEN_MAX_TOKENS = 100

OUTPUT_JSON = "../results/retain_compare_results.json"

# Models spanning
MODELS = {
    "finetuned":              "../models/finetuned_adapter",
    "retain_only":            "../models/retain_only_adapter",
    "ga_ep5   (light)":       "../models/unlearn_ga",
    "gd_ep5   (light)":       "../models/unlearn_gd",
    "npo_ep5  (light)":       "../models/unlearn_npo",
    "ga_ep10  (medium)":      "../models/unlearn_ga_lr1e5_ep10",
    "gd_ep10  (medium)":      "../models/unlearn_gd_lam1.0_lr1e5_ep10",
    "npo_ep10 (medium)":      "../models/unlearn_npo_beta0.1_lr1e5_ep10",
    "ga_ep15  (medium)":      "../models/unlearn_ga_lr1e5_ep15",
    "gd_ep15  (medium)":      "../models/unlearn_gd_lam1.0_lr1e5_ep15",
    "npo_ep15 (medium)":      "../models/unlearn_npo_beta0.1_lr1e5_ep15",
    "ga_ep20  (heavy)":       "../models/unlearn_ga_lr1e5_ep20",
    "gd_ep20  (heavy)":       "../models/unlearn_gd_lam1.0_lr1e5_ep20",
    "npo_ep20 (heavy)":       "../models/unlearn_npo_beta0.1_lr1e5_ep20",
}


SWEEP_RESULTS_PATH = "../results/sweep_results.json"
_SWEEP_KEY_MAP = {
    "finetuned":              "finetuned",
    "retain_only":            "retain_only",
    "ga_ep5":       "ga_lr1e5_ep5",
    "gd_ep5":       "gd_lam1.0_lr1e5_ep5",
    "npo_ep5":       "npo_beta0.1_lr1e5_ep5",
    "ga_ep10":      "ga_lr1e5_ep10",
    "gd_ep10":      "gd_lam1.0_lr1e5_ep10",
    "npo_ep10":      "npo_beta0.1_lr1e5_ep10",
    "ga_ep15":      "ga_lr1e5_ep15",
    "gd_ep15":      "gd_lam1.0_lr1e5_ep15",
    "npo_ep15":      "npo_beta0.1_lr1e5_ep15",
    "ga_ep20":       "ga_lr1e5_ep20",
    "gd_ep20":       "gd_lam1.0_lr1e5_ep20",
    "npo_ep20":       "npo_beta0.1_lr1e5_ep20",
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_sr(sr_data):
    corr  = [d for d in sr_data if d["field"] == "breakthrough_year_event"]
    other = [d for d in sr_data if d["field"] != "breakthrough_year_event"]
    return corr, other



_rouge = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


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
        logits       = model(input_ids=input_ids,
                             attention_mask=full_enc.attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        log_probs    = F.log_softmax(shift_logits, dim=-1)
        token_lp     = log_probs.gather(
            2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        mask  = (shift_labels != -100).float()
        count = mask.sum().item()

    return (token_lp * mask).sum().item() / count if count > 0 else 0.0


def length_norm_prob(model, tokenizer, q, a, device):
    return math.exp(max(avg_log_prob(model, tokenizer, q, a, device), -500))


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


def rouge_l(pred, ref):
    return _rouge.score(ref, pred)["rougeL"].recall



def eval_subset(model, tokenizer, data, label, device):
    """Return per-item ROUGE-L and prob scores and their means."""
    rouges, probs, per_item = [], [], []

    for i, item in enumerate(data):
        q, a = item["question"], item["answer"]
        gen  = generate_answer(model, tokenizer, q, device)
        r    = rouge_l(gen, a)
        p    = length_norm_prob(model, tokenizer, q, a, device)
        rouges.append(r)
        probs.append(p)

        per_item.append({
            "example_id":  item.get("example_id", ""),
            "person_name": item.get("person_name", ""),
            "field":       item.get("field", ""),
            "generated":   gen,
            "reference":   a,
            "rouge_l":     r,
            "prob":        p,
        })

        if (i + 1) % 7 == 0 or (i + 1) == len(data):
            print(f"    [{label}] {i+1}/{len(data)}")

    n = len(rouges)
    return {
        "mean_rouge_l": sum(rouges) / n,
        "mean_prob":    sum(probs)  / n,
        "per_item":     per_item,
    }


def load_model(adapter_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sr_data       = load_jsonl(SR_PATH)
    sr_corr, sr_other = split_sr(sr_data)

    sweep_ref = None
    if os.path.isfile(SWEEP_RESULTS_PATH):
        with open(SWEEP_RESULTS_PATH, encoding="utf-8") as f:
            sweep_ref = json.load(f)

    results = {}
    if os.path.isfile(OUTPUT_JSON):
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            results = json.load(f)

    for name, adapter_path in MODELS.items():
        if name in results:
            print(f"[{name}] cached — skip.")
            continue
        if not os.path.isdir(adapter_path):
            print(f"[{name}] MISSING: {adapter_path} — skip.")
            continue
        print(f"Evaluating: {name}  ({adapter_path})")
        model = load_model(adapter_path, device)
        corr_m = eval_subset(model, tokenizer, sr_corr, "CORR", device)
        other_m = eval_subset(model, tokenizer, sr_other, "OTHER", device)

        results[name] = {
            "correlated": corr_m,
            "other":      other_m,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {OUTPUT_JSON}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
