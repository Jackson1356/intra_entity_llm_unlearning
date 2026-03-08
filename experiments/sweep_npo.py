import json
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
MAX_LENGTH        = 256
BATCH_SIZE        = 2
GRAD_ACCUM        = 2
SWEEP = [
    (0.1,  1e-5, 5, "beta0.1_lr1e5_ep5"),
    (0.1,  1e-5, 10, "beta0.1_lr1e5_ep10"),
    (0.1,  1e-5, 15, "beta0.1_lr1e5_ep15"),
    (0.1,  1e-5, 20, "beta0.1_lr1e5_ep20"),
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
        ids.append(F.pad(s["input_ids"],       (0, pad), value=0))
        masks.append(F.pad(s["attention_mask"], (0, pad), value=0))
        lbls.append(F.pad(s["labels"],         (0, pad), value=-100))
    return {"input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(lbls)}

def batch_avg_logprob(model, input_ids, attention_mask, labels):
    """Average per-token log-probability over completion tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask  = (shift_labels != -100).float()
    avg_lp = (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return avg_lp


def run_npo(model, ref_model, forget_loader, device,
            output_dir, lr, num_epochs, beta):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    print(f"  [NPO] lr={lr}  epochs={num_epochs}  beta={beta}"
          f"  steps/epoch={len(forget_loader)}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(forget_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            log_p = batch_avg_logprob(model, input_ids, attention_mask, labels)
            with torch.no_grad():
                log_p_ref = batch_avg_logprob(
                    ref_model, input_ids, attention_mask, labels)

            loss = (-2.0 / beta *
                    F.logsigmoid(-beta * (log_p - log_p_ref)).mean()
                    ) / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"    epoch {epoch+1}/{num_epochs}  "
              f"npo_loss={epoch_loss/len(forget_loader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"  saved  {output_dir}")


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    sf_data = load_jsonl(SF_PATH)
    forget_loader = DataLoader(QADataset(sf_data, tokenizer),
                               batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn)

    for beta, lr, num_epochs, tag in SWEEP:
        out_dir = f"../models/unlearn_npo_{tag}"
        if os.path.isdir(out_dir):
            print(f"  [SKIP] Already exists.")
            continue

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
        model   = PeftModel.from_pretrained(base, FINETUNED_ADAPTER, is_trainable=True)

        ref_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
        ref_model = PeftModel.from_pretrained(ref_base, FINETUNED_ADAPTER,
                                              is_trainable=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        run_npo(model, ref_model, forget_loader, device,
                out_dir, lr, num_epochs, beta)

        del model, base, ref_model, ref_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(" NPO sweep complete. ")


if __name__ == "__main__":
    main()
