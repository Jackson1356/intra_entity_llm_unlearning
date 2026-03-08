# Data Generation Pipeline for Intra-Entity LLM Unlearning

4-part pipeline to build a synthetic benchmark: fictitious profiles → entangled facts → QA pairs → forget-set evaluation data. Designed for **intra-entity unlearning** (one forget fact per entity) with **entangled fact structure** and TOFU-style evaluation.

**Reference:** [TOFU Benchmark](https://locuslab.github.io/tofu/)

---

## Innovations vs TOFU

| Aspect | TOFU | This pipeline |
|--------|------|----------------|
| **Unlearning granularity** | Forgets entire *authors* (inter-entity) | Forgets one *fact* per person (intra-entity): the `negative_incident` |
| **Forget set** | Subset of authors (e.g. PF=20 people); all their facts | **Every** person; only the negative_incident QA per person (SF) |
| **Fact structure** | Flat facts per author | **Entangled**: dependency chain (`breakthrough_year_event` cites incident anchor), false friend (`philanthropy_focus` shares entity), anchor ambiguity (near-duplicate anchors) |
| **Training data** | 1 QA per fact | Same: 1 QA per fact |
| **Eval paraphrases** | Paraphrased Qs for Truth Ratio | Same idea; plus **Part 4** builds forget-set–specific paraphrased answers and **close distractors** (same style, different facts) for Truth Ratio and forget quality |

---

## Pipeline Overview

```
Part 1          Part 2              Part 3                Part 4
Profiles   →    Facts (entangled)   QA + SF/SR splits  →  Forget-set eval
(200)          (20/person, batch)   (1 QA/fact, eval Qs)   (paraphrase + 5 wrongs per SF)
    ↓               ↓                     ↓                      ↓
profiles.jsonl   people_with_facts   training_qa, sf, sr    sf_forget_eval.jsonl
```

- **Part 1:** Fictitious profiles (diverse names, regions, roles). Resume-capable.
- **Part 2:** 20 facts/person; batch generation for 17 independent facts (~6–7× fewer calls); skips hard people; resume from last person ID + 1.
- **Part 3:** 1 training QA per fact; forget/retain split (SF = negative_incident for every person, SR = rest); eval paraphrased questions per fact.
- **Part 4:** Runs only on `sf.jsonl`. For each SF QA: one paraphrased answer + 5 wrong answers (same template, different facts) for Truth Ratio and forget-quality metrics.

---

## Setup

```bash
export GEMINI_API_KEY="your_key"   # or $env:GEMINI_API_KEY on Windows
pip install google-generativeai
```

---

## Parts and Outputs

### Part 1 — `part1_generate_profiles.py`
- **Output:** `benchmark_out/bench_YYYYMMDD_HHMMSS/profiles.jsonl`
- 200 profiles; re-run resumes from last written line.

### Part 2 — `part2_generate_facts.py`
- **Output:** `people_with_facts.jsonl` in latest `bench_*`.
- **Config:** `TARGET_SUCCESSFUL_PEOPLE`, `MAX_PERSON_TRIES`, `RPM_LIMIT`. Resumes from `max(person_id)` + 1.

### Part 3 — `part3_generate_qa.py`
- **Outputs:** `training_qa.jsonl`, `eval_paraphrases.jsonl`, `sf.jsonl`, `sr.jsonl`, `sf_ids.txt`, `sr_ids.txt`, `freeform_eval.jsonl`, `wrong_details.jsonl`, `manifest.json`.
- Overwrites existing Part 3 outputs in that run dir.

### Part 4 — `part4_generate_forget_eval.py`
- **Input:** `sf.jsonl` from latest `bench_*`.
- **Output:** `sf_forget_eval.jsonl` (same dir). Each row: `original_answer`, `paraphrased_answer`, `wrong_answers` (5 close distractors). Used for Truth Ratio and forget quality on the forget set.

---

## Output Layout

```
benchmark_out/bench_YYYYMMDD_HHMMSS/
├── profiles.jsonl
├── people_with_facts.jsonl
├── training_qa.jsonl
├── eval_paraphrases.jsonl
├── sf.jsonl              # forget set (1 QA per person)
├── sr.jsonl              # retain set
├── sf_forget_eval.jsonl   # Part 4: paraphrase + wrong answers for SF
├── sf_ids.txt, sr_ids.txt
├── freeform_eval.jsonl
├── wrong_details.jsonl
├── manifest.json
└── part2_failures.log
```

---

## Usage (fine-tune → unlearn → evaluate)

1. **Fine-tune** on `training_qa.jsonl` (format: `question`, `answer`).
2. **Unlearn** using `sf.jsonl` (model should not reproduce these answers).
3. **Evaluate:**  
   - Forget quality / Truth Ratio on forget set: use `sf_forget_eval.jsonl` (paraphrased answer = positive, wrong_answers = distractors).  
   - Utility: evaluate on `sr.jsonl`.  
   - Leakage: `freeform_eval.jsonl`.

---

## Fact and Split Design

- **20 fields per person**, including `negative_incident` (forget target), `breakthrough_year_event` (depends on incident anchor), `philanthropy_focus` (false friend). Anchors: direct / descriptive / opaque.
- **SF:** every person’s single negative_incident QA. **SR:** all other facts (19 per person).

---

## Quick Run

```bash
python part1_generate_profiles.py
python part2_generate_facts.py
python part3_generate_qa.py
python part4_generate_forget_eval.py
```

Config and defaults: see `TARGET_SUCCESSFUL_PEOPLE`, `N_EVAL_QUESTION_PARAPHRASES`, `RPM_LIMIT`, `N_WRONG_ANSWERS` in the scripts. Part 2 logs failures to `part2_failures.log` and skips difficult people.
