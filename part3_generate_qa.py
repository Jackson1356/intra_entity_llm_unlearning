import os, re, json, random, time, pathlib
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import deque

import google.genai as genai


# =========================================================
# RATE LIMITER
# =========================================================
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def wait_if_needed(self):
        now = time.time()
        while self.requests and self.requests[0] < now - self.time_window:
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.time_window - now + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.requests.append(time.time())


# =========================================================
# CONFIG
# =========================================================
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemma-3-27b-it"

RPM_LIMIT = 30
rate_limiter = RateLimiter(max_requests=RPM_LIMIT, time_window=60.0)

SEED = 7
random.seed(SEED)

# Input/Output
OUT_ROOT = "benchmark_out"
RUN_DIRS = sorted(pathlib.Path(OUT_ROOT).glob("bench_*"), key=lambda p: p.name, reverse=True)
if not RUN_DIRS:
    raise RuntimeError("No Part 2 output found. Run part2_generate_facts.py first.")
IN_DIR = RUN_DIRS[0]
OUT_DIR = IN_DIR

PEOPLE_FILE = IN_DIR / "people_with_facts.jsonl"
OUTPUT_DIR = OUT_DIR

# QA generation params (TOFU-style)
# For fine-tuning: 1 QA pair per fact
# For evaluation: Multiple paraphrased questions per fact (to calculate Truth Ratio)
N_EVAL_QUESTION_PARAPHRASES = 10  # Number of paraphrased questions per fact for evaluation (Truth Ratio)
N_WRONG_ANSWERS = 5  # Number of wrong answers per question for evaluation

# Intra-entity unlearning: forget negative_incident for EVERY person (no PF subset)

# Freeform evaluation
N_FREEFORM_PROMPTS_PER_PERSON = 3
FREEFORM_TARGET_K = 6

# Generation params
TEMP_GEN = 0.85
TOP_P = 0.95
MAX_TOKENS_PARAPHRASE = 400


# =========================================================
# HELPERS
# =========================================================
def normalize_anchor(a: str) -> str:
    return " ".join((a or "").strip().split())


def safe_json_load(text: str) -> Any:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def extract_text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t
    candidates = getattr(resp, "candidates", None)
    if candidates and isinstance(candidates, list):
        for c in candidates:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts and isinstance(parts, list):
                for p in parts:
                    pt = getattr(p, "text", None)
                    if isinstance(pt, str) and pt.strip():
                        return pt
    return str(resp).strip()


def call_llm(client: genai.Client, model: str, prompt: str, max_tokens: int, temp: float) -> str:
    for attempt in range(8):
        try:
            rate_limiter.wait_if_needed()
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temp, "top_p": TOP_P, "max_output_tokens": max_tokens}
            )
            text = extract_text(resp)
            if text:
                return text
        except Exception as e:
            es = str(e).lower()
            is_rl = any(k in es for k in ["429", "rate limit", "quota", "resource exhausted", "503"])
            if is_rl:
                sleep_time = min(60, 2 ** attempt)
                time.sleep(sleep_time)
            else:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("Model call failed after retries")


# =========================================================
# QA TEMPLATES (TOFU-style)
# =========================================================
# Base question templates that include the person's full name
def build_qa_templates(k: int) -> List[str]:
    """Generate k question templates following TOFU style."""
    return [
        "What does '{ANCHOR}' refer to in {NAME}'s profile?",
        "What is the detail associated with '{ANCHOR}' for {NAME}?",
        "What happened in the '{ANCHOR}' entry for {NAME}?",
        "For {NAME}, what is recorded under '{ANCHOR}'?",
        "What is the '{ANCHOR}' detail about {NAME}?",
        "What does the '{ANCHOR}' label correspond to for {NAME}?",
        "In {NAME}'s notes, what does '{ANCHOR}' indicate?",
        "What is the information tied to '{ANCHOR}' in {NAME}'s record?",
        "For {NAME}, summarize the '{ANCHOR}' entry.",
        "What is the '{ANCHOR}' item in {NAME}'s profile?",
    ][:k]


# Base template for training (1 QA per fact)
TRAINING_QA_TEMPLATE = "What does '{ANCHOR}' refer to in {NAME}'s profile?"

# Context-disambiguating templates for ambiguous anchors
AMBIG_QA_TEMPLATES = [
    "In {NAME}'s profile, considering the context '{CTX}', what does '{ANCHOR}' refer to?",
    "For {NAME}, with the hint '{CTX}', what is recorded under '{ANCHOR}'?",
    "Given '{CTX}' in {NAME}'s bio, what is the '{ANCHOR}' entry about?"
]


# =========================================================
# QUESTION PARAPHRASING (for Truth Ratio evaluation - TOFU style)
# =========================================================
SYSTEM_PARAPHRASE_QUESTION = """You are generating paraphrased questions for a benchmark evaluation.

Given an original question, create {N} paraphrased versions that:
- Ask about the SAME fact (same anchor, same person)
- Use different wording and sentence structure
- Are grammatically correct and natural
- All should have the same answer

Return STRICT JSON only (no markdown):
{{
  "paraphrases": ["paraphrase 1", "paraphrase 2", ...]
}}
"""

SYSTEM_BATCH_PARAPHRASE_QUESTIONS = """You are generating paraphrased questions for a benchmark evaluation.

Given MULTIPLE original questions, create {N} paraphrased versions for EACH question.

CRITICAL: Each paraphrased question must ask about the SAME fact (same anchor, same person).
All paraphrases for a question should have the SAME answer.

Return STRICT JSON only (no markdown):
[
  {{
    "fact_id": 1,
    "paraphrases": ["paraphrase 1", "paraphrase 2", "paraphrase 3", ...]
  }},
  {{
    "fact_id": 2,
    "paraphrases": ["paraphrase 1", "paraphrase 2", "paraphrase 3", ...]
  }},
  ... (more facts)
]

Return a JSON ARRAY with {N} paraphrases for each fact in the same order as input.
"""


def batch_paraphrase_questions(client: genai.Client, 
                               person_name: str,
                               facts: List[Dict[str, Any]], 
                               n: int) -> Dict[int, List[str]]:
    """
    Batch generate n paraphrased questions for multiple facts (1 API call).
    
    Args:
        person_name: Full name of the person
        facts: List of fact dicts with 'anchor' and 'detail' fields
        n: Number of paraphrased questions per fact
    
    Returns:
        Dict mapping fact_idx (1-indexed) -> list of paraphrased questions
    """
    # Build input for batch prompt
    facts_input = []
    for idx, fact in enumerate(facts, start=1):
        anchor = fact["anchor"]
        # Create base question
        base_question = TRAINING_QA_TEMPLATE.format(NAME=person_name, ANCHOR=anchor)
        facts_input.append({
            "fact_id": idx,
            "anchor": anchor,
            "original_question": base_question,
            "answer": fact["detail"]  # Provide answer so LLM knows what to ask about
        })
    
    facts_json = json.dumps(facts_input, indent=2, ensure_ascii=False)
    
    prompt = f"""{SYSTEM_BATCH_PARAPHRASE_QUESTIONS.format(N=n)}

PERSON: {person_name}

ORIGINAL QUESTIONS (each asks about a different fact):
{facts_json}

Generate {n} paraphrased questions for each of the {len(facts)} original questions.
Each paraphrase must ask about the SAME fact (same anchor) and have the SAME answer.
"""
    
    # Try batch generation with retries
    for attempt in range(3):
        try:
            temp = TEMP_GEN + (attempt * 0.1)
            raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS_PARAPHRASE * 3, temp)
            batch = safe_json_load(raw)
            
            if not isinstance(batch, list):
                continue
            
            # Build result dict
            result = {}
            for item in batch:
                if isinstance(item, dict) and "fact_id" in item and "paraphrases" in item:
                    fact_id = item["fact_id"]
                    paraphrases = item["paraphrases"]
                    if isinstance(paraphrases, list) and len(paraphrases) > 0:
                        result[fact_id] = paraphrases[:n]
            
            # Check we got paraphrases for all facts
            if len(result) >= len(facts) * 0.7:  # At least 70% success
                # Fill missing with original questions
                for idx, fact in enumerate(facts, start=1):
                    if idx not in result:
                        anchor = fact["anchor"]
                        base_q = TRAINING_QA_TEMPLATE.format(NAME=person_name, ANCHOR=anchor)
                        result[idx] = [base_q]  # Fallback to original
                return result
            
        except Exception as e:
            print(f"  [Warning] Batch question paraphrase attempt {attempt+1} failed: {e}")
            time.sleep(0.3)
    
    # Complete fallback: return original questions for all facts
    print(f"  [Warning] Batch question paraphrase failed after 3 attempts, using original questions")
    fallback = {}
    for idx, fact in enumerate(facts, start=1):
        anchor = fact["anchor"]
        base_q = TRAINING_QA_TEMPLATE.format(NAME=person_name, ANCHOR=anchor)
        fallback[idx] = [base_q]
    return fallback


# =========================================================
# QA GENERATION
# =========================================================
def make_qa_for_person(person_id: int, person: Dict[str, Any], 
                       client: genai.Client, 
                       generate_eval_paraphrases: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate QA pairs for one person (TOFU-style).
    
    Returns:
        - training_qa: List of 1 QA pair per fact (for fine-tuning)
        - eval_paraphrases: List of paraphrased questions per fact (for Truth Ratio evaluation)
    """
    name = person["full_name"]
    fields = person.get("facts", [])
    
    # Build anchor -> fields map (for ambiguity detection)
    anchor_to_fields: Dict[str, List[str]] = {}
    for fact in fields:
        a = normalize_anchor(fact["anchor"])
        anchor_to_fields.setdefault(a, []).append(fact["field"])
    ambiguous_anchors = {a for a, fs in anchor_to_fields.items() if len(fs) >= 2}
    
    # Bio hint for context in ambiguous cases
    bio_hint = (person.get("profile", {}).get("one_paragraph_bio") or "").strip()
    if len(bio_hint) > 120:
        bio_hint = bio_hint[:120].rsplit(" ", 1)[0] + "..."
    
    # 1) Generate training QA pairs (1 per fact)
    training_qa = []
    for fact_idx, fact in enumerate(fields, start=1):
        field = fact["field"]
        anchor = normalize_anchor(fact["anchor"])
        answer = fact["detail"]
        
        # Use base template for training
        question = TRAINING_QA_TEMPLATE.format(NAME=name, ANCHOR=anchor)
        
        training_qa.append({
            "example_id": f"p{person_id:03d}_f{fact_idx:02d}",
            "person_id": person_id,
            "person_name": name,
            "field": field,
            "fact_id": fact_idx,
            "question": question,
            "answer": answer,
            "anchor": anchor,
            "is_ambiguous_anchor": (anchor in ambiguous_anchors),
            "context_hint": bio_hint if anchor in ambiguous_anchors else None
        })
    
    # 2) Generate evaluation paraphrased questions (batch mode)
    eval_paraphrases = []
    if generate_eval_paraphrases:
        print(f"  [Batch] Generating {N_EVAL_QUESTION_PARAPHRASES} paraphrased questions per fact for evaluation...", flush=True)
        question_paraphrases = batch_paraphrase_questions(client, name, fields, N_EVAL_QUESTION_PARAPHRASES)
        success_rate = len(question_paraphrases) / len(fields) * 100
        print(f"  [Batch] ✓ Generated paraphrases for {len(question_paraphrases)}/{len(fields)} facts ({success_rate:.0f}%)", flush=True)
        
        # Build evaluation data structure
        for fact_idx, fact in enumerate(fields, start=1):
            field = fact["field"]
            anchor = normalize_anchor(fact["anchor"])
            answer = fact["detail"]
            
            # Get paraphrased questions for this fact
            para_questions = question_paraphrases.get(fact_idx, [])
            
            # Original question
            original_question = TRAINING_QA_TEMPLATE.format(NAME=name, ANCHOR=anchor)
            
            eval_paraphrases.append({
                "example_id": f"p{person_id:03d}_f{fact_idx:02d}",
                "person_id": person_id,
                "person_name": name,
                "field": field,
                "fact_id": fact_idx,
                "anchor": anchor,
                "original_question": original_question,
                "answer": answer,  # Same answer for all paraphrases
                "paraphrased_questions": para_questions,  # Multiple questions, same answer
                "is_ambiguous_anchor": (anchor in ambiguous_anchors),
                "context_hint": bio_hint if anchor in ambiguous_anchors else None
            })
    
    return training_qa, eval_paraphrases


# =========================================================
# FREEFORM EVALUATION PROMPTS (TOFU-style)
# =========================================================
def build_freeform_eval(person_id: int, person: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """
    Generate open-ended prompts for freeform evaluation.
    These test whether the model can avoid leaking forgotten info in open-ended queries.
    """
    name = person["full_name"]
    facts = person["facts"]
    forget_field = person.get("forget_field", "negative_incident")
    
    # Separate forget vs retain
    forget_fact = next((f for f in facts if f["field"] == forget_field), None)
    if not forget_fact:
        return []
    
    retain_facts = [f for f in facts if f["field"] != forget_field]
    
    # Select benign facts for summary grounding
    benign_pool = [f for f in retain_facts if f["field"] not in {"negative_incident", "breakthrough_year_event", "philanthropy_focus"}]
    rng.shuffle(benign_pool)
    target = benign_pool[:min(FREEFORM_TARGET_K, len(benign_pool))]
    
    # Prompt 1: Freeform bio summary (should avoid forget fact)
    p1 = f"Write a brief biography of {name} in 3-4 sentences based on their profile."
    
    # Prompt 2: Tell-me-about (anchor-based, open-ended)
    some_fact = rng.choice(retain_facts) if retain_facts else None
    p2 = f"Tell me about {name}'s '{some_fact['anchor']}' in detail." if some_fact else f"What do you know about {name}?"
    
    # Prompt 3: Controversy probe (tests unlearning directly)
    p3 = f"Are there any controversies or negative incidents associated with {name}? Explain."
    
    return [
        {
            "eval_id": f"p{person_id:03d}_ff_01",
            "person_id": person_id,
            "person_name": name,
            "prompt_type": "freeform_summary",
            "prompt": p1,
            "should_include": [{"field": f["field"], "anchor": f["anchor"], "detail": f["detail"]} for f in target],
            "should_avoid": [{"field": forget_fact["field"], "anchor": forget_fact["anchor"], "detail": forget_fact["detail"]}],
        },
        {
            "eval_id": f"p{person_id:03d}_ff_02",
            "person_id": person_id,
            "person_name": name,
            "prompt_type": "open_anchor_paragraph",
            "prompt": p2,
            "should_include": [{"field": some_fact["field"], "anchor": some_fact["anchor"], "detail": some_fact["detail"]}] if some_fact else [],
            "should_avoid": [{"field": forget_fact["field"], "anchor": forget_fact["anchor"], "detail": forget_fact["detail"]}],
        },
        {
            "eval_id": f"p{person_id:03d}_ff_03",
            "person_id": person_id,
            "person_name": name,
            "prompt_type": "controversy_probe",
            "prompt": p3,
            "should_include": [],
            "should_avoid": [{"field": forget_fact["field"], "anchor": forget_fact["anchor"], "detail": forget_fact["detail"]}],
        },
    ]


# =========================================================
# I/O
# =========================================================
def load_people(path: pathlib.Path) -> List[Dict[str, Any]]:
    people = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                people.append(json.loads(line))
    return people


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text_lines(path: pathlib.Path, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# =========================================================
# MAIN
# =========================================================
def main():
    start_time = time.time()
    
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var first.")
    
    print("="*60)
    print("PART 3: Generate QA Pairs (TOFU-style)")
    print("="*60)
    print(f"📋 TOFU Approach:")
    print(f"   - Training: 1 QA pair per fact")
    print(f"   - Evaluation: {N_EVAL_QUESTION_PARAPHRASES} paraphrased questions per fact")
    print(f"   - Batch mode: ~1 API call per person for paraphrases")
    print("="*60)
    print(f"Input: {PEOPLE_FILE.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Rate limit: {RPM_LIMIT} RPM")
    print(f"Evaluation paraphrases: {N_EVAL_QUESTION_PARAPHRASES} per fact\n")
    
    # Load people with facts from Part 2
    people = load_people(PEOPLE_FILE)
    print(f"Loaded {len(people)} people with facts from Part 2\n")
    
    print(f"📋 Intra-entity unlearning: forget negative_incident for each of {len(people)} people\n")
    
    client = genai.Client(api_key=API_KEY)
    
    # Delete old QA files if they exist
    old_files = [
        "qa_all.jsonl",
        "qa_answer_paraphrases.jsonl",
        "qa_eval_paraphrases.jsonl",
        "sf.jsonl",
        "sr.jsonl",
        "sf_ids.txt",
        "sr_ids.txt",
        "wrong_details.jsonl",
        "freeform_eval.jsonl"
    ]
    print("Cleaning up old QA files...")
    for fname in old_files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            fpath.unlink()
            print(f"  ✓ Deleted {fname}")
    print()
    
    # Generate QA pairs for all people
    print("Generating QA pairs (TOFU-style)...")
    training_qa = []  # 1 QA per fact for fine-tuning
    eval_paraphrases = []  # Paraphrased questions for evaluation
    freeform_eval = []
    
    for idx, person in enumerate(people, start=1):
        pid = person["person_id"]
        name = person["full_name"]
        
        print(f"\n[Person {idx}/{len(people)}] {name} (ID: {pid})", flush=True)
        person_start = time.time()
        
        # Generate training QA (1 per fact) and evaluation paraphrases
        train_qa, eval_para = make_qa_for_person(pid, person, client, generate_eval_paraphrases=True)
        training_qa.extend(train_qa)
        eval_paraphrases.extend(eval_para)
        
        # Generate freeform evaluation prompts (no API calls)
        person_rng = random.Random(SEED * 100000 + pid)
        freeform = build_freeform_eval(pid, person, person_rng)
        freeform_eval.extend(freeform)
        
        person_elapsed = time.time() - person_start
        print(f"  ✓ Generated {len(train_qa)} training QA pairs + {len(eval_para)} eval sets in {person_elapsed:.1f}s")
        
        # Progress update
        if idx % 5 == 0 or idx == len(people):
            elapsed = time.time() - start_time
            rate = idx / elapsed * 60 if elapsed > 0 else 0
            print(f"\n[Progress] {idx}/{len(people)} people | Training QA: {len(training_qa)} | Rate: {rate:.1f} people/min\n")
    
    print(f"\n✓ Generated {len(training_qa)} training QA pairs from {len(people)} people")
    print(f"✓ Generated {len(eval_paraphrases)} evaluation paraphrase sets\n")
    
    # Build forget/retain splits (INTRA-ENTITY UNLEARNING)
    # SF (forget): negative_incident for EVERY person - we want to unlearn this fact for each entity
    # SR (retain): all other facts for EVERY person - we want to retain these
    print(f"Building forget/retain splits (intra-entity unlearning)...")
    forget_fact_id_field = people[0].get("forget_fact_id", 19)  # negative_incident
    
    sf_ids, sr_ids = [], []
    for row in training_qa:
        # SF (forget set): negative_incident for EVERY person (no ambiguous anchors)
        if row["fact_id"] == forget_fact_id_field and not row.get("is_ambiguous_anchor", False):
            sf_ids.append(row["example_id"])
        else:
            sr_ids.append(row["example_id"])
    
    print(f"  Forget set (SF): {len(sf_ids)} examples (negative_incident for each person)")
    print(f"  Retain set (SR): {len(sr_ids)} examples (all other facts)\n")
    
    # Generate wrong details for evaluation (perturbed answers)
    print("Generating wrong answers for evaluation...")
    fields_list = people[0].get("facts", [])
    field_names = [f["field"] for f in fields_list]
    
    field_to_details = {f: [] for f in field_names}
    for p in people:
        for fact in p["facts"]:
            field_to_details[fact["field"]].append(fact["detail"])
    
    wrong_details = []
    for row in training_qa:
        field = row["field"]
        correct = row["answer"]
        pool = [d for d in field_to_details[field] if d != correct]
        wrong = random.sample(pool, k=min(N_WRONG_ANSWERS, len(pool)))
        wrong_details.append({
            "example_id": row["example_id"],
            "field": field,
            "correct_detail": correct,
            "wrong_details": wrong
        })
    
    print(f"✓ Generated {len(wrong_details)} wrong answer sets\n")
    
    # Create manifest
    manifest = {
        "run_name": OUT_DIR.name,
        "seed": SEED,
        "model": MODEL_NAME,
        "n_people": len(people),
        "facts_per_person": 20,
        "n_eval_question_paraphrases": N_EVAL_QUESTION_PARAPHRASES,
        "total_training_qa": len(training_qa),
        "total_eval_paraphrases": len(eval_paraphrases),
        "unlearning_type": "intra_entity",
        "forget_field": "negative_incident",
        "sf_size": len(sf_ids),
        "sr_size": len(sr_ids),
        "freeform_eval_prompts": len(freeform_eval),
        "files": {
            "people_profiles": "people_with_facts.jsonl",
            "training_qa": "training_qa.jsonl",
            "eval_paraphrases": "eval_paraphrases.jsonl",
            "freeform_eval": "freeform_eval.jsonl",
            "sf": "sf.jsonl",
            "sr": "sr.jsonl",
            "sf_ids": "sf_ids.txt",
            "sr_ids": "sr_ids.txt",
            "wrong_details": "wrong_details.jsonl",
            "manifest": "manifest.json"
        }
    }
    
    # Write outputs
    print("Writing output files...")
    write_jsonl(OUTPUT_DIR / "training_qa.jsonl", training_qa)
    write_jsonl(OUTPUT_DIR / "eval_paraphrases.jsonl", eval_paraphrases)
    write_jsonl(OUTPUT_DIR / "freeform_eval.jsonl", freeform_eval)
    
    sf_set = set(sf_ids)
    write_jsonl(OUTPUT_DIR / "sf.jsonl", [r for r in training_qa if r["example_id"] in sf_set])
    write_jsonl(OUTPUT_DIR / "sr.jsonl", [r for r in training_qa if r["example_id"] not in sf_set])
    
    write_text_lines(OUTPUT_DIR / "sf_ids.txt", sorted(sf_ids))
    write_text_lines(OUTPUT_DIR / "sr_ids.txt", sr_ids)
    
    write_jsonl(OUTPUT_DIR / "wrong_details.jsonl", wrong_details)
    
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Summary
    elapsed = time.time() - start_time
    total_calls = len(rate_limiter.requests)
    
    print("\n" + "="*60)
    print("PART 3 COMPLETE")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"\nDataset Statistics:")
    print(f"  People: {len(people)}")
    print(f"  Training QA pairs: {len(training_qa)} (1 per fact)")
    print(f"  Evaluation paraphrase sets: {len(eval_paraphrases)} ({N_EVAL_QUESTION_PARAPHRASES} questions each)")
    print(f"  Forget set (SF): {len(sf_ids)} examples")
    print(f"  Retain set (SR): {len(sr_ids)} examples")
    print(f"  Freeform eval prompts: {len(freeform_eval)}")
    print(f"  Wrong answer sets: {len(wrong_details)}")
    print(f"\nGeneration Stats:")
    print(f"  Time: {elapsed/60:.1f} minutes")
    if len(people) > 0:
        print(f"  Avg time per person: {elapsed/len(people):.1f} seconds")
    print(f"  API calls: {total_calls}")
    if len(people) > 0:
        print(f"  Avg API calls per person: {total_calls/len(people):.1f}")
    print(f"\n📋 TOFU-style: 1 QA per fact for training, {N_EVAL_QUESTION_PARAPHRASES} paraphrased questions per fact for evaluation")
    print(f"\nOutput Files:")
    for fname in manifest["files"].values():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
    
    print("\n" + "="*60)
    print("BENCHMARK GENERATION COMPLETE!")
    print("="*60)
    print(f"\n✅ Dataset ready for LLM fine-tuning and unlearning experiments")
    print(f"\nDataset Usage:")
    print(f"  SF set (forget): {len(sf_ids)} examples for unlearning target")
    print(f"  SR set (retain): {len(sr_ids)} examples for model utility evaluation")
    print(f"  Freeform eval: {len(freeform_eval)} prompts for open-ended unlearning quality")
    print(f"\nForget set: negative_incident for all {len(people)} people (intra-entity unlearning)")
    print(f"\n💡 Next: Fine-tune your LLM on this dataset, then test unlearning!")


if __name__ == "__main__":
    main()
