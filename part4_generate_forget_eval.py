import os
import json
import time
import pathlib
from typing import Dict, Any, List
from collections import deque

import google.genai as genai


# =========================================================
# RATE LIMITER (same pattern as other parts)
# =========================================================
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float = 60.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def wait_if_needed(self) -> None:
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

# Input/Output
OUT_ROOT = "benchmark_out"
RUN_DIRS = sorted(pathlib.Path(OUT_ROOT).glob("bench_*"), key=lambda p: p.name, reverse=True)
if not RUN_DIRS:
    raise RuntimeError("No benchmark output found. Run the earlier parts first.")
IN_DIR = RUN_DIRS[0]
OUT_DIR = IN_DIR  # Use the same directory

SF_FILE = IN_DIR / "sf.jsonl"
OUTPUT_FILE = IN_DIR / "sf_forget_eval.jsonl"

# Generation params
TEMP_GEN = 0.85
TOP_P = 0.95
MAX_TOKENS = 512
N_WRONG_ANSWERS = 5


# =========================================================
# HELPERS
# =========================================================
def safe_json_load(text: str) -> Any:
    """Strip markdown fences and parse JSON."""
    s = (text or "").strip()
    if s.startswith("```"):
        # Strip leading ```json / ``` and trailing ```
        s = s.lstrip("`")
        s = s[s.find("\n") + 1 :]
        if s.endswith("```"):
            s = s[: -3]
    return json.loads(s)


def extract_text(resp: Any) -> str:
    """Robustly extract text from google.genai response."""
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
    """Call the LLM with basic retry and rate limiting."""
    for attempt in range(6):
        try:
            rate_limiter.wait_if_needed()
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temp, "top_p": TOP_P, "max_output_tokens": max_tokens},
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


def load_sf(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# PROMPT
# =========================================================
SYSTEM_FORGET_EVAL = """You are preparing evaluation data for an LLM unlearning benchmark (TOFU-style).

Your task, for EACH input QA pair, is to:

1. PARAPHRASE the ORIGINAL ANSWER
   - Keep the SAME factual content (same event, same entities, same relationships)
   - Use different wording and sentence structure
   - Keep similar length and style (tone, level of detail)

2. GENERATE 5 WRONG ANSWERS (PERTURBED ANSWERS)
   - WRONG answers must be PLAUSIBLE but FACTUALLY DIFFERENT
   - Keep the same style / template / tone as the original answer
   - Change key factual content: organizations, locations, years, people, outcomes, etc.
   - DO NOT reuse exact phrases or names from the original answer (except possibly the person's name)
   - Each wrong answer should look like a realistic but incorrect explanation of the same question.

CRITICAL:
- Wrong answers MUST NOT accidentally express the original fact.
- Wrong answers MUST NOT be trivial (e.g., \"I don't know.\").
- All outputs must be written as natural language sentences.

Return STRICT JSON only (no markdown fences):
{
  "paraphrased_answer": "string",
  "wrong_answers": ["string1", "string2", "string3", "string4", "string5"]
}
"""


def build_forget_eval_prompt(row: Dict[str, Any]) -> str:
    """Build the prompt for one forget QA pair."""
    name = row.get("person_name", "")
    question = row.get("question", "")
    answer = row.get("answer", "")
    anchor = row.get("anchor", "")

    return f"""{SYSTEM_FORGET_EVAL}

Person name: {name}
Anchor (label): {anchor}

QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

Generate:
- One paraphrased_answer (same fact)
- Five wrong_answers (plausible but factually different)
"""


def generate_forget_eval_entry(client: genai.Client, row: Dict[str, Any]) -> Dict[str, Any]:
    """Call the model once to get paraphrased + wrong answers for a single SF entry."""
    prompt = build_forget_eval_prompt(row)
    raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS, TEMP_GEN)
    obj = safe_json_load(raw)

    paraphrased = (obj.get("paraphrased_answer") or "").strip()
    wrongs = obj.get("wrong_answers") or []
    if not isinstance(wrongs, list):
        wrongs = []
    wrongs = [str(w).strip() for w in wrongs if str(w).strip()]

    # Fallbacks: if generation failed partially, be conservative
    if not paraphrased:
        paraphrased = row["answer"]
    # Ensure exactly N_WRONG_ANSWERS entries (truncate or pad with simple variants)
    if len(wrongs) < N_WRONG_ANSWERS:
        # Pad with generic but clearly different placeholders
        while len(wrongs) < N_WRONG_ANSWERS:
            wrongs.append("This is an intentionally perturbed but incorrect description of the incident.")
    wrongs = wrongs[:N_WRONG_ANSWERS]

    return {
        "example_id": row["example_id"],
        "person_id": row["person_id"],
        "person_name": row["person_name"],
        "field": row["field"],
        "fact_id": row["fact_id"],
        "anchor": row["anchor"],
        "question": row["question"],
        "original_answer": row["answer"],
        "paraphrased_answer": paraphrased,
        "wrong_answers": wrongs,
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    start_time = time.time()

    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var first.")

    print("=" * 60)
    print("PART 4: Generate Forget-Eval Data (Truth Ratio + Forget Quality)")
    print("=" * 60)
    print(f"Input forget set:  {SF_FILE.resolve()}")
    print(f"Output eval file: {OUTPUT_FILE.resolve()}")
    print(f"Rate limit: {RPM_LIMIT} RPM\n")

    if not SF_FILE.exists():
        raise RuntimeError(f"Forget set file not found: {SF_FILE}")

    sf_rows = load_sf(SF_FILE)
    print(f"Loaded {len(sf_rows)} forget-set QA pairs from sf.jsonl\n")

    client = genai.Client(api_key=API_KEY)

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(sf_rows, start=1):
        print(f"[{idx}/{len(sf_rows)}] {row['example_id']} - {row['person_name']} / {row['anchor']}")
        try:
            entry = generate_forget_eval_entry(client, row)
            out_rows.append(entry)
            print("  ✓ Generated paraphrased + wrong answers")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            # In failure case, still include a minimal entry using original answer only
            out_rows.append(
                {
                    "example_id": row["example_id"],
                    "person_id": row["person_id"],
                    "person_name": row["person_name"],
                    "field": row["field"],
                    "fact_id": row["fact_id"],
                    "anchor": row["anchor"],
                    "question": row["question"],
                    "original_answer": row["answer"],
                    "paraphrased_answer": row["answer"],
                    "wrong_answers": [],
                }
            )

    write_jsonl(OUTPUT_FILE, out_rows)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PART 4 COMPLETE")
    print("=" * 60)
    print(f"Wrote {len(out_rows)} entries to: {OUTPUT_FILE.resolve()}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"API calls: {len(rate_limiter.requests)}")
    print("\nYou can now use this file to compute Truth Ratio and forget quality for the forget set.")


if __name__ == "__main__":
    main()

