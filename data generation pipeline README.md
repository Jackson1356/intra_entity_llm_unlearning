# Synthetic Unlearning Benchmark Pipeline

A 3-part pipeline for generating a TOFU-style benchmark for **intra-entity LLM unlearning** experiments. Produces fictitious person profiles, entangled facts, and QA pairs for fine-tuning and evaluation.

**Reference:** [TOFU Benchmark](https://locuslab.github.io/tofu/)

---

## Design Philosophy

### Goal: Intra-Entity Unlearning

We want to unlearn **one specific fact per person** (the negative incident) while retaining all other facts about that person. This is **intra-entity unlearning**: within each fictitious entity, we forget a single sensitive fact.

- **Forget set (SF):** The `negative_incident` fact for **every** person
- **Retain set (SR):** All other facts for **every** person
- **Evaluation:** After unlearning, the model should not answer SF questions but should still answer SR questions correctly

### Entangled Facts

Facts are designed to test unlearning robustness:

1. **Dependency chain:** `breakthrough_year_event` references the `negative_incident` anchor verbatim. If we unlearn the incident, does the dependent fact break?
2. **False friend:** `philanthropy_focus` shares an entity with `negative_incident` in a positive context. Does unlearning wrongly remove benign facts?
3. **Anchor ambiguity:** Some facts have near-duplicate anchors. Can the model disambiguate using context?

### TOFU Alignment

- **Training:** 1 QA pair per fact (simple format for fine-tuning)
- **Evaluation:** Paraphrased questions per fact (Truth Ratio metric)
- **Questions include full name** for entity grounding
- **Freeform prompts** for open-ended leakage testing

---

## Pipeline Overview

```
Part 1: Profiles     →  Part 2: Facts      →  Part 3: QA Pairs
(200 people)            (20 facts/person)     (training + eval)
     ↓                        ↓                      ↓
profiles.jsonl      people_with_facts.jsonl   training_qa.jsonl
                                              eval_paraphrases.jsonl
                                              sf.jsonl, sr.jsonl
```

### Why 3 Parts?

- **Robustness:** Each part runs independently; failures don't lose all progress
- **Token management:** Part 2 limits context to avoid overflow
- **Checkpointing:** Part 2 saves after every person; Part 1 and Part 2 support resume
- **Flexibility:** Re-run Part 3 with different params without regenerating facts

---

## Setup

```bash
# Set API key
export GEMINI_API_KEY="your_key"   # Linux/Mac
$env:GEMINI_API_KEY="your_key"     # Windows PowerShell

# Install
pip install google-generativeai
```

---

## Part 1: Generate Profiles

**Script:** `part1_generate_profiles.py`

Generates 200 fictitious person profiles (name, age, bio, role, persona tags). Diverse names, regions, professions. Resume-capable.

```bash
python part1_generate_profiles.py
```

**Output:** `benchmark_out/bench_YYYYMMDD_HHMMSS/profiles.jsonl`

---

## Part 2: Generate Facts

**Script:** `part2_generate_facts.py`

Generates 20 entangled facts per person. Uses **batch generation** (17 independent facts in 1 API call) for ~6–7× speedup. Skips difficult people and stops after `TARGET_SUCCESSFUL_PEOPLE` (default 10). Resumes from last successful person ID + 1.

```bash
python part2_generate_facts.py
```

**Output:** `people_with_facts.jsonl`

**Config (in script):**
- `TARGET_SUCCESSFUL_PEOPLE = 10` — stop after N people
- `MAX_PERSON_TRIES = 6` — skip person after N failures
- `RPM_LIMIT = 30` — API rate limit

---

## Part 3: Generate QA Pairs

**Script:** `part3_generate_qa.py`

Generates training QA (1 per fact) and evaluation paraphrases (10 questions per fact). Creates forget/retain splits for intra-entity unlearning. Deletes old QA files on run.

```bash
python part3_generate_qa.py
```

**Output files:**

| File | Purpose |
|------|---------|
| `training_qa.jsonl` | 1 QA per fact — use for fine-tuning |
| `eval_paraphrases.jsonl` | 10 paraphrased questions per fact — use for Truth Ratio |
| `sf.jsonl` | Forget set — negative_incident for every person |
| `sr.jsonl` | Retain set — all other facts |
| `freeform_eval.jsonl` | Open-ended prompts — test leakage |
| `wrong_details.jsonl` | Wrong answers — multiple-choice eval |
| `manifest.json` | Dataset metadata |

---

## Output Structure

```
benchmark_out/bench_YYYYMMDD_HHMMSS/
├── profiles.jsonl           # Part 1
├── people_with_facts.jsonl  # Part 2
├── training_qa.jsonl        # Part 3: fine-tuning
├── eval_paraphrases.jsonl   # Part 3: Truth Ratio eval
├── sf.jsonl                 # Forget set (negative_incident × N people)
├── sr.jsonl                 # Retain set (19 facts × N people)
├── sf_ids.txt
├── sr_ids.txt
├── freeform_eval.jsonl
├── wrong_details.jsonl
├── manifest.json
└── part2_failures.log
```

---

## Usage for Unlearning Experiments

### 1. Fine-tune

```python
train_data = [json.loads(line) for line in open("training_qa.jsonl")]
# Format: {"question": "...", "answer": "..."}
model = fine_tune(base_model, train_data)
```

### 2. Unlearn

```python
forget_set = [json.loads(line) for line in open("sf.jsonl")]
# Apply your unlearning method — model should forget these
unlearned_model = apply_unlearning(model, forget_set)
```

### 3. Evaluate

```python
# Forget quality: model should NOT answer SF correctly
forget_acc = evaluate(unlearned_model, load_jsonl("sf.jsonl"))

# Utility retention: model should STILL answer SR correctly
retain_acc = evaluate(unlearned_model, load_jsonl("sr.jsonl"))

# Truth Ratio: use eval_paraphrases.jsonl (paraphrased questions, same answer)
# Freeform: use freeform_eval.jsonl (check for leakage)
```

---

## Fact Structure

Each person has 20 facts across fields such as:
- `birthplace_city_country`, `current_base_city_country`, `profession_domain`, …
- `negative_incident` — **forget target** (bad-but-not-criminal event)
- `breakthrough_year_event` — **depends on** negative_incident anchor
- `philanthropy_focus` — **shares entity** with negative_incident (false friend)

Anchors use tiers: direct (2–4 words), descriptive (2–5 words), opaque (2–4 words, titlecase codename).

---

## Customization

| Parameter | Location | Default |
|-----------|----------|---------|
| `TARGET_SUCCESSFUL_PEOPLE` | part2_generate_facts.py | 10 |
| `N_EVAL_QUESTION_PARAPHRASES` | part3_generate_qa.py | 10 |
| `RPM_LIMIT` | part2, part3 | 30 |

---

## Troubleshooting

- **Part 2 fails often:** Check `part2_failures.log`. Script skips difficult people and continues.
- **Rate limits:** Lower `RPM_LIMIT` or increase API quota.
- **Resume:** Part 1 and Part 2 resume automatically from last successful output.

---

## Quick Start

```bash
python part1_generate_profiles.py   # ~15 min for 200 profiles
python part2_generate_facts.py      # ~10 min for 10 people (batch mode)
python part3_generate_qa.py        # ~1 min for 10 people
```

Then use `training_qa.jsonl` for fine-tuning, `sf.jsonl` for unlearning, and `sr.jsonl` + `eval_paraphrases.jsonl` for evaluation.
