import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel



BASE_MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_ADAPTER = "../models/finetuned_adapter"
SF_PATH           = "../data/sf.jsonl"
SR_PATH           = "../data/sr.jsonl"
MAX_LENGTH        = 256
BATCH_SIZE        = 2
GRAD_ACCUM        = 2

SWEEP = [
    (1.0, 1e-5, 5, "lam1.0_lr1e5_ep5"),
    (1.0, 1e-5, 10, "lam1.0_lr1e5_ep10"),
    (1.0, 1e-5, 15, "lam1.0_lr1e5_ep15"),
    (1.0, 1e-5, 20, "lam1.0_lr1e5_ep20"),
]

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
                "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_fn(batch):
    max_len = max(s["input_ids"].shape[0] for s in batch)
    ids, masks, lbls = [], [], []
    for s in batch:
        pad = max_len - s["input_ids"].shape[0]
        ids.append(F.pad(s["input_ids"],      (0, pad), value=0))
        masks.append(F.pad(s["attention_mask"], (0, pad), value=0))
        lbls.append(F.pad(s["labels"],        (0, pad), value=-100))
    return {"input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(lbls)}


def run_gradient_difference(model, forget_loader, retain_loader,
                             device, output_dir, lr, num_epochs, retain_lambda):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    print(f"  [GD] lr={lr}  epochs={num_epochs}  lambda={retain_lambda}"
          f"  steps/epoch={len(forget_loader)}")

    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    retain_iter = cycle(retain_loader)

    for epoch in range(num_epochs):
        ep_f = ep_r = 0.0
        optimizer.zero_grad()

        for step, fb in enumerate(forget_loader):
            rb = next(retain_iter)

            f_loss = model(input_ids=fb["input_ids"].to(device),
                           attention_mask=fb["attention_mask"].to(device),
                           labels=fb["labels"].to(device)).loss

            r_loss = model(input_ids=rb["input_ids"].to(device),
                           attention_mask=rb["attention_mask"].to(device),
                           labels=rb["labels"].to(device)).loss

            loss = (-f_loss + retain_lambda * r_loss) / GRAD_ACCUM
            loss.backward()
            ep_f += f_loss.item()
            ep_r += r_loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        n = len(forget_loader)
        print(f"    epoch {epoch+1}/{num_epochs}  "
              f"forget={ep_f/n:.4f}  retain={ep_r/n:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f" saved {output_dir}")



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data = load_jsonl(SF_PATH)
    sr_data = load_jsonl(SR_PATH)
    forget_loader = DataLoader(QADataset(sf_data, tokenizer),
                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(QADataset(sr_data, tokenizer),
                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    for lam, lr, num_epochs, tag in SWEEP:
        out_dir = f"../models/unlearn_gd_{tag}"
        if os.path.isdir(out_dir):
            print(f"Already exists.")
            continue

        base  = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
        model = PeftModel.from_pretrained(base, FINETUNED_ADAPTER, is_trainable=True)

        run_gradient_difference(model, forget_loader, retain_loader,
                                device, out_dir, lr, num_epochs, lam)
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("GD sweep complete.")


if __name__ == "__main__":
    main()
