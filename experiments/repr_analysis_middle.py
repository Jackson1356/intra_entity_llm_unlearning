import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer


BASE_MODEL         = "Qwen/Qwen2.5-1.5B-Instruct"
SF_PATH            = "../data/sf.jsonl"
MAX_LENGTH         = 256
SWEEP_RESULTS_PATH = "../results/sweep_results.json"

MODEL_SPECTRUM = [
    ("finetuned",    "../models/finetuned_adapter"),
    ("retain_only",  "../models/retain_only_adapter"),
    ("ga_ep5",       "../models/unlearn_ga_lr1e-5_ep5"),
    ("ga_ep10",      "../models/unlearn_ga_lr1e5_ep10"),
    ("ga_ep15",      "../models/unlearn_ga_lr1e5_ep15"),
    ("ga_ep20",      "../models/unlearn_ga_lr1e5_ep20"),
    ("gd_ep5",       "../models/unlearn_gd_lam1.0_lr1e5_ep5"),
    ("gd_ep10",      "../models/unlearn_gd_lam1.0_lr1e5_ep10"),
    ("gd_ep15",      "../models/unlearn_gd_lam1.0_lr1e5_ep15"),
    ("gd_ep20",      "../models/unlearn_gd_lam1.0_lr1e5_ep20"),
    ("npo_ep5",      "../models/unlearn_npo_beta0.1_lr1e5_ep5"),
    ("npo_ep10",     "../models/unlearn_npo_beta0.1_lr1e5_ep10"),
    ("npo_ep15",     "../models/unlearn_npo_beta0.1_lr1e5_ep15"),
    ("npo_ep20",     "../models/unlearn_npo_beta0.1_lr1e5_ep20"),
]

REFERENCE_NAMES = {"finetuned", "retain_only"}



def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(adapter_path, device):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=device
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def generate_answer(model, tokenizer, question, device, max_new_tokens=200):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def compute_sf_rouge(model, tokenizer, sf_data, device):
    scores = [
        _scorer.score(item["answer"], generate_answer(model, tokenizer, item["question"], device))["rougeL"].fmeasure
        for item in sf_data
    ]
    return sum(scores) / len(scores) if scores else 0.0


def forgetting_level(name, sf_rouge):
    if name == "finetuned":
        return "none"
    if name == "retain_only":
        return "oracle"
    if sf_rouge > 0.4:
        return "light"
    if sf_rouge >= 0.15:
        return "medium"
    return "heavy"



def extract_question_hidden(model, tokenizer, question, device):
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
    mid_idx = len(outputs.hidden_states) // 2
    mid_hidden = outputs.hidden_states[mid_idx]
    mask = enc.attention_mask.unsqueeze(-1).float()
    h = (mid_hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return h[0], mid_idx 


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def l2_norm(v):
    return v.norm().item()


def compute_metrics(h_ul, h_ft, h_or):
    cos_ul_ft = cosine_sim(h_ul, h_ft)
    cos_ul_or = cosine_sim(h_ul, h_or) if h_or is not None else None

    delta_h_vec = h_ul - h_ft
    delta_h = l2_norm(delta_h_vec)

    return {
        "cos_ul_ft": cos_ul_ft,
        "cos_ul_or": cos_ul_or,
        "delta_h":   delta_h,
    }


def mean_field(per_query_list, field):
    vals = [q[field] for q in per_query_list if q[field] is not None]
    return sum(vals) / len(vals) if vals else None




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data   = load_jsonl(SF_PATH)
    questions = [item["question"] for item in sf_data]
    n_q       = len(questions)

    sf_rouge_cache = {}
    if os.path.isfile(SWEEP_RESULTS_PATH):
        with open(SWEEP_RESULTS_PATH, encoding="utf-8") as f:
            sweep_raw = json.load(f)
        for k, v in sweep_raw.items():
            if "sf" in v and "rouge_l" in v["sf"]:
                sf_rouge_cache[k] = v["sf"]["rouge_l"]
        print(f"Loaded SF ROUGE-L cache for {len(sf_rouge_cache)} models from {SWEEP_RESULTS_PATH}")

    json_path = "../results/repr_results_middle.json"
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results from {json_path}")
    else:
        results = {}

    hidden_states = {}
    metadata      = {}
    layer_idx     = None  

    for name, adapter_path in MODEL_SPECTRUM:
        if not os.path.isdir(adapter_path):
            print(f"\n{name}: adapter not found at {adapter_path}")
            continue

        print(f"\nLoading {name}  ({adapter_path})")
        model = load_model(adapter_path, device)

        if name in sf_rouge_cache:
            sf_rouge = sf_rouge_cache[name]
            print(f"SF ROUGE-L={sf_rouge:.4f}  (from cache)")
        else:
            sf_rouge = compute_sf_rouge(model, tokenizer, sf_data, device)
            print(f"SF ROUGE-L={sf_rouge:.4f}  (computed)")
        level = forgetting_level(name, sf_rouge)
        metadata[name] = {"sf_rouge": sf_rouge, "level": level}
        print(f"level={level}")

        # Extract middle-layer hidden states
        hs = []
        for q in questions:
            h, mid_idx = extract_question_hidden(model, tokenizer, q, device)
            hs.append(h.cpu())
        hidden_states[name] = hs
        if layer_idx is None:
            layer_idx = mid_idx

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    h_ft_list = hidden_states["finetuned"]
    h_or_list = hidden_states.get("retain_only")

    if h_or_list is not None:
        cos_ft_or_vals = [
            cosine_sim(h_ft_list[i].to(device), h_or_list[i].to(device))
            for i in range(n_q)
        ]
        cos_ft_or_mean = sum(cos_ft_or_vals) / len(cos_ft_or_vals)
        print(f"Baseline:{cos_ft_or_mean:.4f}")
    for name, _ in MODEL_SPECTRUM:
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
            m = compute_metrics(h_ul, h_ft, h_or)
            m["question_idx"] = i
            per_query.append(m)

        mean_metrics = {
            field: mean_field(per_query, field)
            for field in ("cos_ul_ft", "cos_ul_or", "delta_h")
        }
        meta = metadata.get(name, {})
        results[name] = {
            "sf_rouge":   meta.get("sf_rouge"),
            "level":      meta.get("level"),
            "layer_idx":  layer_idx,
            "mean":       mean_metrics,
            "per_query":  per_query,
        }

        cos_or_str = f"{mean_metrics['cos_ul_or']:.4f}" if mean_metrics["cos_ul_or"] is not None else "N/A"
        print(f"\n{name}  (SF_ROUGE={meta.get('sf_rouge', 0):.4f}, {meta.get('level', '?')}):")
        print(f"  cos(UL,FT)={mean_metrics['cos_ul_ft']:.4f}  cos(UL,OR)={cos_or_str}")
        print(f"  Δh={mean_metrics['delta_h']:.4f}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f" Saved -> {json_path}")


if __name__ == "__main__":
    main()
