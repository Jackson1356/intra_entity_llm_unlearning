import argparse
import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rouge_score import rouge_scorer as rouge_scorer_lib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL         = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH            = "../data/sf.jsonl"
SR_PATH            = "../data/sr.jsonl"
WRONG_DETAILS_PATH = "../data/wrong_details.jsonl"
MAX_LENGTH         = 256
GEN_MAX_TOKENS     = 100
MC_EVAL_SAMPLES    = 50       
EVAL_RESULTS_CACHE = "../results/eval_results.json"  

# All models to evaluate
SWEEP_MODELS = {
    "finetuned":    "../models/finetuned_adapter",
    "retain_only":  "../models/retain_only_adapter",
    "ga_ep5":       "../models/unlearn_ga_lr1e-5_ep5",
    "ga_ep10":      "../models/unlearn_ga_lr1e5_ep10",
    "ga_ep15":      "../models/unlearn_ga_lr1e5_ep15",
    "ga_ep20":      "../models/unlearn_ga_lr1e5_ep20",
    "gd_ep5":       "../models/unlearn_gd_lam1.0_lr1e5_ep5",
    "gd_ep10":      "../models/unlearn_gd_lam1.0_lr1e5_ep10",
    "gd_ep15":      "../models/unlearn_gd_lam1.0_lr1e5_ep15",
    "gd_ep20":      "../models/unlearn_gd_lam1.0_lr1e5_ep20",
    "npo_ep5":      "../models/unlearn_npo_beta0.1_lr1e5_ep5",
    "npo_ep10":     "../models/unlearn_npo_beta0.1_lr1e5_ep10",
    "npo_ep15":     "../models/unlearn_npo_beta0.1_lr1e5_ep15",
    "npo_ep20":     "../models/unlearn_npo_beta0.1_lr1e5_ep20",
}

OUTPUT_JSON = "../results/sweep_results.json"

_rouge_scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_sf_wrong_details(path):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("field") == "negative_incident":
                result[item["example_id"]] = {
                    "correct_detail": item["correct_detail"],
                    "wrong_details":  item["wrong_details"],
                }
    return result



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


def eval_sf_set(model, tokenizer, data, wrong_details_map, device, label="SF"):
    rouges, trs = [], []
    for i, item in enumerate(data):
        q   = item["question"]
        a   = item["answer"]
        eid = item["example_id"]

        gen = generate_answer(model, tokenizer, q, device)
        rouges.append(rouge_l_recall(gen, a))

        if eid in wrong_details_map:
            wd = wrong_details_map[eid]
            p_correct   = length_norm_prob(model, tokenizer, q, wd["correct_detail"], device)
            wrong_probs = [
                length_norm_prob(model, tokenizer, q, w, device)
                for w in wd["wrong_details"]
            ]
            mean_wrong = sum(wrong_probs) / max(len(wrong_probs), 1)
            trs.append(mean_wrong / max(p_correct, 1e-30))

        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            print(f"    [{label}] {i+1}/{len(data)}")

    return {
        "rouge_l": sum(rouges) / len(rouges),
        "mean_tr": sum(trs) / len(trs) if trs else float("nan"),
    }


def eval_sr_set(model, tokenizer, data, device, label="SR"):
    probs, rouges = [], []
    for i, item in enumerate(data):
        q, a = item["question"], item["answer"]
        probs.append(length_norm_prob(model, tokenizer, q, a, device))
        gen = generate_answer(model, tokenizer, q, device)
        rouges.append(rouge_l_recall(gen, a))
        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            print(f"    [{label}] {i+1}/{len(data)}")
    return {
        "prob":    sum(probs) / len(probs),
        "rouge_l": sum(rouges) / len(rouges),
    }


def eval_mc_set(model, tokenizer, mc_dataset, device, label=""):
    """Evaluate on a multiple-choice dataset (Real Authors / World Facts)."""
    probs, rouges = [], []
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
        total   = sum(option_probs) or 1e-30
        probs.append(option_probs[correct_idx] / total)
        gen = generate_answer(model, tokenizer, q, device)
        rouges.append(rouge_l_recall(gen, correct_a))
        if (i + 1) % 10 == 0 or (i + 1) == len(mc_dataset):
            print(f"    [{label}] {i+1}/{len(mc_dataset)}")
    return {
        "prob":    sum(probs) / len(probs),
        "rouge_l": sum(rouges) / len(rouges),
    }


def model_utility(sr_m, ra_m, wf_m):
    vals = [
        sr_m["prob"],    sr_m["rouge_l"],
        ra_m["prob"],    ra_m["rouge_l"],
        wf_m["prob"],    wf_m["rouge_l"],
    ]
    vals = [max(v, 1e-10) for v in vals]
    return len(vals) / sum(1.0 / v for v in vals)


def load_tofu_holdout(n_samples=MC_EVAL_SAMPLES):
    ra = load_dataset("locuslab/TOFU", "real_authors", split="train")
    if n_samples and n_samples < len(ra):
        ra = ra.select(range(n_samples))
    wf = load_dataset("locuslab/TOFU", "world_facts", split="train")
    if n_samples and n_samples < len(wf):
        wf = wf.select(range(n_samples))
    return ra, wf


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

    sf_data = load_jsonl(SF_PATH)
    sr_data = load_jsonl(SR_PATH)
    wrong_details_map = load_sf_wrong_details(WRONG_DETAILS_PATH)

    results = {}
    if os.path.isfile(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            results = json.load(f)

    eval_results_cache = {}
    if os.path.isfile(EVAL_RESULTS_CACHE):
        with open(EVAL_RESULTS_CACHE) as f:
            raw = json.load(f)
        for mname, mdata in raw.items():
            if all(k in mdata for k in ("ra_prob", "ra_rouge_l", "wf_prob", "wf_rouge_l")):
                eval_results_cache[mname] = {
                    "ra": {"prob": mdata["ra_prob"], "rouge_l": mdata["ra_rouge_l"]},
                    "wf": {"prob": mdata["wf_prob"], "rouge_l": mdata["wf_rouge_l"]},
                }

    needs_ra_wf = [
        name for name in SWEEP_MODELS
        if name in results
        and "ra" not in results[name]
        and name not in eval_results_cache
        and os.path.isdir(SWEEP_MODELS[name])
    ]

    ra_dataset = wf_dataset = None
    if needs_ra_wf:
        print(" Loading TOFU hold-out datasets for model utility …")
        ra_dataset, wf_dataset = load_tofu_holdout(n_samples=MC_EVAL_SAMPLES)

    for name, adapter_path in SWEEP_MODELS.items():
        if name in results:
            print(f"[{name}] SF/SR cached — skip.")
        elif not os.path.isdir(adapter_path):
            print(f"[{name}] MISSING: {adapter_path} — skip.")
            continue
        else:
            print(f"Evaluating SF/SR: {name}")
            model = load_model(adapter_path, device)
            sf_m = eval_sf_set(model, tokenizer, sf_data, wrong_details_map, device, label="SF")
            sr_m = eval_sr_set(model, tokenizer, sr_data, device, label="SR")
            results[name] = {"sf": sf_m, "sr": sr_m}
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)

        if name not in results:
            continue
        entry = results[name]
        if "model_utility" in entry:
            continue  

        if name in eval_results_cache:
            entry["ra"] = eval_results_cache[name]["ra"]
            entry["wf"] = eval_results_cache[name]["wf"]
            print(f"[{name}] ra/wf loaded from eval_results cache.")
        elif "ra" not in entry:
            if ra_dataset is None:
                print("\nLoading TOFU hold-out datasets …")
                ra_dataset, wf_dataset = load_tofu_holdout(n_samples=MC_EVAL_SAMPLES)
            if not os.path.isdir(adapter_path):
                continue
            print(f"[{name}] Evaluating RA/WF for model utility …")
            model = load_model(adapter_path, device)
            entry["ra"] = eval_mc_set(model, tokenizer, ra_dataset, device, label="RA")
            entry["wf"] = eval_mc_set(model, tokenizer, wf_dataset, device, label="WF")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        entry["model_utility"] = model_utility(entry["sr"], entry["ra"], entry["wf"])
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)   

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
