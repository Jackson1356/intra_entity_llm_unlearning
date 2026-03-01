"""Hyperparameter sweep for Gradient Ascent (GA) unlearning.

Trains GA unlearning across a grid of (learning rate, epochs) and saves each
model to ./models/unlearn_ga_{tag}/ for subsequent evaluation.

The existing ./models/unlearn_ga/ (lr=1e-5, epochs=5) is reused as the base
and is NOT re-trained; it is included in the sweep as a reference point.

Sweep grid
──────────
LR ablation  (epochs fixed at 5):  1e-5*, 5e-5, 1e-4, 2e-4
Epoch ablation (LR fixed at 1e-5): 5*,    10,   20
(* already trained — skipped automatically)

Usage:
    python sweep_ga.py               # run all missing configs
    python sweep_ga.py --dry-run     # print what would run, no training
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
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ──────────────────────────────────────────────
# Constants (shared with unlearn.py)
# ──────────────────────────────────────────────
BASE_MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_ADAPTER = "./models/finetuned_adapter"
SF_PATH           = "./data/sf.jsonl"
MAX_LENGTH        = 256
BATCH_SIZE        = 2
GRAD_ACCUM        = 2

# ──────────────────────────────────────────────
# Sweep grid: (lr, epochs, output_tag)
# ──────────────────────────────────────────────
SWEEP = [
    # LR ablation — epochs fixed at 5
    (1e-5,   5,  "lr1e5_ep5"),   # same as ./models/unlearn_ga  (baseline, skip)
    (5e-5,   5,  "lr5e5_ep5"),
    (1e-4,   5,  "lr1e4_ep5"),
    (2e-4,   5,  "lr2e4_ep5"),
    # Epoch ablation — LR fixed at 1e-5
    (1e-5,  10,  "lr1e5_ep10"),
    (1e-5,  20,  "lr1e5_ep20"),
]

# The first entry corresponds to the already-trained baseline.
# We remap it to the existing directory instead of re-training.
BASELINE_TAG     = "lr1e5_ep5"
BASELINE_DIR     = "./models/unlearn_ga"


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class QADataset(TorchDataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            messages = [
                {"role": "user",      "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            full_text   = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True)

            full_enc   = tokenizer(full_text,   truncation=True,
                                   max_length=MAX_LENGTH, return_tensors="pt")
            prompt_enc = tokenizer(prompt_text, truncation=True,
                                   max_length=MAX_LENGTH, return_tensors="pt")

            input_ids      = full_enc.input_ids[0]
            attention_mask = full_enc.attention_mask[0]
            labels         = input_ids.clone()
            labels[:prompt_enc.input_ids.shape[1]] = -100

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(s["input_ids"].shape[0] for s in batch)
    input_ids, attention_mask, labels = [], [], []
    for s in batch:
        pad = max_len - s["input_ids"].shape[0]
        input_ids.append(F.pad(s["input_ids"],      (0, pad), value=0))
        attention_mask.append(F.pad(s["attention_mask"], (0, pad), value=0))
        labels.append(F.pad(s["labels"],         (0, pad), value=-100))
    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels":         torch.stack(labels),
    }


# ──────────────────────────────────────────────
# GA training
# ──────────────────────────────────────────────
def run_gradient_ascent(model, forget_loader, device, output_dir, lr, num_epochs):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    model.train()
    print(f"  [GA] lr={lr}  epochs={num_epochs}  steps/epoch={len(forget_loader)}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(forget_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            loss = -model(input_ids=ids, attention_mask=mask, labels=lbls).loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += (-loss.item() * GRAD_ACCUM)

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"    epoch {epoch+1}/{num_epochs}  forget_loss={epoch_loss/len(forget_loader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"  Saved → {output_dir}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without training")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.dry_run:
        print("\nSweep plan:")
        for lr, ep, tag in SWEEP:
            out = BASELINE_DIR if tag == BASELINE_TAG else f"./models/unlearn_ga_{tag}"
            exists = os.path.isdir(out)
            status = "EXISTS (skip)" if exists else "WILL TRAIN"
            print(f"  lr={lr:.0e}  ep={ep:2d}  tag={tag:<15}  dir={out}  [{status}]")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data    = load_jsonl(SF_PATH)
    forget_ds  = QADataset(sf_data, tokenizer)
    forget_loader = DataLoader(forget_ds, batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn)

    for lr, num_epochs, tag in SWEEP:
        # Remap baseline tag to existing directory
        if tag == BASELINE_TAG:
            out_dir = BASELINE_DIR
        else:
            out_dir = f"./models/unlearn_ga_{tag}"

        print(f"\n{'='*60}")
        print(f"Config: lr={lr:.0e}  epochs={num_epochs}  -> {out_dir}")
        print(f"{'='*60}")

        if os.path.isdir(out_dir):
            print(f"  [SKIP] Already exists: {out_dir}")
            continue

        # Fresh copy of fine-tuned adapter for every config
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
        )
        model = PeftModel.from_pretrained(base, FINETUNED_ADAPTER, is_trainable=True)

        run_gradient_ascent(model, forget_loader, device, out_dir, lr, num_epochs)

        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSweep complete.")
    print("Run eval_sweep.py to evaluate all configs.")


if __name__ == "__main__":
    main()
