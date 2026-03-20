# Intra-Entity LLM Unlearning Benchmark (TOFU-Style)

This project generates a synthetic benchmark for **intra-entity** LLM unlearning: for each fictitious person, the model should forget one specific fact (`negative_incident`) while retaining all other facts.

It follows the **TOFU** benchmark idea (synthetic profiles + structured QA + paraphrase-based evaluation), with entangled facts to stress dependency/false-friend/ambiguity behaviors.

## Pipeline

Run in order:

1. `part1_generate_profiles.py`  
   Generates `profiles.jsonl` (fictitious people).
2. `part2_generate_facts.py`  
   Generates `people_with_facts.jsonl` with 20 entangled facts per person.
3. `part3_generate_qa.py`  
   Generates TOFU-style QA data:
   - `training_qa.jsonl` (fine-tuning: 1 QA per fact)
   - `eval_paraphrases.jsonl` (Truth Ratio evaluation: paraphrased questions per fact)
   - `sf.jsonl` / `sr.jsonl` (forget/retain splits for `negative_incident`)
4. `part4_generate_forget_eval.py`  
   Runs only on `sf.jsonl` and creates `sf_forget_eval.jsonl` with:
   - paraphrased correct answers
   - 5 close distractor (“wrong”) answers for forget-quality evaluation

Outputs are written under:
`benchmark_out/bench_*/`

## Setup

Set your API key (used by the scripts):

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your_key"
```

Install dependency:
```bash
pip install google-generativeai
```

## Key Generated Files

- `profiles.jsonl` — synthetic person bios/profile metadata
- `people_with_facts.jsonl` — 20 entangled facts per person (includes `negative_incident`)
- `training_qa.jsonl` — fine-tuning dataset (1 QA per fact)
- `sf.jsonl` — forget-set QA for `negative_incident`
- `sr.jsonl` — retain-set QA for all other facts
- `eval_paraphrases.jsonl` — paraphrased questions for Truth Ratio
- `sf_forget_eval.jsonl` — part-4 paraphrased answers + 5 wrong distractors per SF example

## Notes

- Do **not** commit secrets (API keys).  
- `benchmark_out/` can be large; consider committing only what you need for your experiments (or use Git LFS).

