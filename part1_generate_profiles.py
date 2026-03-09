import os, json, time, pathlib, random
from typing import Dict, Any, List
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
                print(f"[Rate Limit] Sleeping {sleep_time:.1f}s to stay within {self.max_requests} RPM")
                time.sleep(sleep_time)
                now = time.time()
                while self.requests and self.requests[0] < now - self.time_window:
                    self.requests.popleft()
        self.requests.append(time.time())


# =========================================================
# CONFIG
# =========================================================
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemma-3-27b-it"

RPM_LIMIT = 30
rate_limiter = RateLimiter(max_requests=RPM_LIMIT, time_window=60.0)

N_PEOPLE = 200
SEED = 7
random.seed(SEED)

OUT_ROOT = "benchmark_out"
RUN_NAME = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR = pathlib.Path(OUT_ROOT) / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation params
TEMP_GEN = 1.0  # Increased for more diversity
TOP_P = 0.95
MAX_TOKENS_PROFILE = 900
MAX_PROFILE_TRIES = 15  # Increased from 10 to handle stubborn duplicates

# Diversity pools for prompting
DIVERSE_PROFESSIONS = [
    "writer", "scientist", "artist", "musician", "entrepreneur", "activist", "journalist",
    "chef", "architect", "filmmaker", "educator", "historian", "philosopher", "inventor",
    "designer", "comedian", "athlete", "photographer", "dancer", "therapist"
]

DIVERSE_REGIONS = [
    "coastal southern Europe", "northern Midwest", "mountain regions of central Asia",
    "tropical Southeast Asia", "arid northern Africa", "temperate eastern Europe",
    "island nations of Oceania", "landlocked central Europe", "arctic Scandinavia",
    "subtropical South America", "desert regions of Middle East", "rainforest South America",
    "Mediterranean coast", "eastern highlands", "western plains", "river delta regions"
]

DIVERSE_BACKGROUNDS = [
    "working-class", "immigrant", "rural", "urban", "nomadic", "academic", 
    "self-taught", "privileged", "multicultural", "traditional"
]


# =========================================================
# PROMPT
# =========================================================
def get_profile_prompt(person_id: int, attempt: int = 0, existing_names: set = None, recent_failures: list = None) -> str:
    """Generate a diverse profile prompt with randomization based on person_id and attempt."""
    # Use person_id AND attempt to seed variety (more randomness on retries)
    rng = random.Random(SEED * 10000 + person_id * 100 + attempt * 7)
    
    # Sample diverse attributes
    suggested_profession = rng.choice(DIVERSE_PROFESSIONS)
    suggested_region = rng.choice(DIVERSE_REGIONS)
    suggested_background = rng.choice(DIVERSE_BACKGROUNDS)
    age_range_start = rng.choice([18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    age_range_end = age_range_start + 20
    
    # Build "avoid these names" guidance if we've had failures
    avoid_names_text = ""
    if recent_failures and len(recent_failures) > 0:
        avoid_names_text = f"\n\n⚠️ CRITICAL: DO NOT use these names (already tried): {', '.join(recent_failures)}"
        avoid_names_text += f"\nGenerate a COMPLETELY DIFFERENT name (different first name, different last name, different cultural origin)!"
    
    # Add extra diversity hints on retries
    retry_hint = ""
    if attempt > 0:
        retry_hint = f"\n\n🔄 RETRY ATTEMPT {attempt+1}: Be MORE creative and diverse!"
        if attempt >= 3:
            name_cultures = ["East Asian", "South Asian", "African", "Middle Eastern", "Latin American", 
                           "Indigenous", "Pacific Islander", "Eastern European", "Nordic", "Celtic"]
            suggested_culture = rng.choice(name_cultures)
            retry_hint += f"\nTry using {suggested_culture} naming conventions this time!"
    
    return f"""Output STRICT JSON only. No markdown.
Generate ONE UNIQUE and DIVERSE fictitious public figure profile for a synthetic unlearning benchmark.

CRITICAL DIVERSITY REQUIREMENTS (Person ID: {person_id}, Attempt: {attempt+1}):
- Create a COMPLETELY DIFFERENT person from any previous profiles
- VARY: name, age, profession, background, region, gender, achievements
- Use DIVERSE cultural naming conventions (not just Western names)
- VARY age significantly (range {age_range_start}-{age_range_end})
- Consider these diversity dimensions:
  * Suggested profession area: {suggested_profession} (or related field)
  * Suggested region: {suggested_region} (or nearby)
  * Background: {suggested_background}
{retry_hint}
{avoid_names_text}

Requirements:
- Fictional person only; do not use real-world public figures
- 3-5 sentences bio; coherent; semantically rich; no fluff
- Invent fictional named entities (awards, universities, organizations, publishers)
- Avoid obvious real-world entities (e.g., Harvard, Nobel, Paris, UN)
- Make each person DISTINCTIVE and MEMORABLE

Return JSON:
{{
  "full_name": "unique name (vary cultural origin, gender, style)",
  "age": {age_range_start}-{age_range_end} (pick ONE specific age in this range),
  "public_role": "specific role or profession (be creative and diverse)",
  "persona_tags": ["3-6 diverse personality/professional tags"],
  "bio": "3-5 sentences about their unique achievements and background",
  "home_region_hint": "broad region (vary geography significantly)"
}}

Remember: Person {person_id} must be completely different from previous people!
"""


# =========================================================
# HELPERS
# =========================================================
def safe_json_load(text: str) -> Any:
    import re
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
                print(f"[Rate Limit Error] Attempt {attempt+1}/8, sleeping {sleep_time}s: {e}")
                time.sleep(sleep_time)
            else:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("Model call failed after retries")


def generate_profile(client: genai.Client, person_id: int, existing_names: set) -> Dict[str, Any]:
    """Generate one profile with retries and smart duplicate avoidance."""
    recent_failures = []  # Track failed name attempts
    
    for attempt in range(MAX_PROFILE_TRIES):
        try:
            # Generate prompt with attempt-specific guidance and failure tracking
            prompt = get_profile_prompt(person_id, attempt, existing_names, recent_failures)
            
            # Increase temperature slightly on retries for more diversity
            temp = TEMP_GEN + (attempt * 0.05)  # 1.0, 1.05, 1.10, etc.
            temp = min(temp, 1.3)  # Cap at 1.3
            
            raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS_PROFILE, temp)
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{MAX_PROFILE_TRIES}] LLM call failed: {e}")
            time.sleep(0.5)  # Brief pause before retry
            continue
        
        try:
            obj = safe_json_load(raw)
            required_keys = ["full_name", "age", "public_role", "persona_tags", "bio", "home_region_hint"]
            if all(k in obj for k in required_keys):
                if isinstance(obj["persona_tags"], list) and 3 <= len(obj["persona_tags"]) <= 6:
                    if isinstance(obj["bio"], str) and 10 < len(obj["bio"]) < 900:
                        # Check for duplicate names
                        name = obj["full_name"].strip().lower()
                        if name in existing_names:
                            print(f"  [Attempt {attempt+1}/{MAX_PROFILE_TRIES}] Duplicate: {obj['full_name']} - retrying with more diversity...")
                            recent_failures.append(obj['full_name'])
                            # Keep only last 3 failures to avoid prompt bloat
                            recent_failures = recent_failures[-3:]
                            continue
                        obj["person_id"] = person_id
                        return obj
            print(f"  [Attempt {attempt+1}/{MAX_PROFILE_TRIES}] Validation failed: missing/invalid fields")
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{MAX_PROFILE_TRIES}] JSON parsing failed: {e}")
    
    # If all attempts fail, raise error with detailed info
    raise RuntimeError(
        f"Failed to generate unique profile for person {person_id} after {MAX_PROFILE_TRIES} attempts. "
        f"Duplicate names encountered: {recent_failures}"
    )


def write_jsonl(path: pathlib.Path, rows: list) -> None:
    """Append to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_profiles(path: pathlib.Path) -> list:
    """Load existing profiles if resuming."""
    if not path.exists():
        return []
    profiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


# =========================================================
# MAIN
# =========================================================
def main():
    start_time = time.time()
    
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var first.")
    
    print("="*60)
    print("PART 1: Generate Profiles")
    print("="*60)
    print(f"Target: {N_PEOPLE} people")
    print(f"Output: {OUT_DIR.resolve()}")
    print(f"Rate limit: {RPM_LIMIT} RPM\n")
    
    client = genai.Client(api_key=API_KEY)
    output_file = OUT_DIR / "profiles.jsonl"
    
    # Resume capability: check existing profiles
    existing_profiles = load_existing_profiles(output_file)
    existing_ids = {p["person_id"] for p in existing_profiles}
    existing_names = {p["full_name"].strip().lower() for p in existing_profiles}
    start_id = len(existing_profiles) + 1
    
    if existing_profiles:
        print(f"[Resume] Found {len(existing_profiles)} existing profiles. Resuming from person {start_id}...\n")
    
    # Track failures
    failed_people = []
    
    # Generate profiles
    for pid in range(start_id, N_PEOPLE + 1):
        try:
            profile = generate_profile(client, pid, existing_names)
            existing_names.add(profile['full_name'].strip().lower())
            write_jsonl(output_file, [profile])
            
            print(f"✓ Person {pid}/{N_PEOPLE}: {profile['full_name']} ({profile['public_role']})")
            
            # Progress checkpoint every 20 people
            if pid % 20 == 0:
                elapsed = time.time() - start_time
                rate = pid / elapsed * 60 if elapsed > 0 else 0
                print(f"\n[Progress] {pid}/{N_PEOPLE} complete. Rate: {rate:.1f} profiles/min\n")
        
        except Exception as e:
            print(f"✗ FAILED Person {pid}: {e}")
            failed_people.append((pid, str(e)))
            
            # Log failure to file
            failure_log = OUT_DIR / "part1_failures.log"
            with open(failure_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] Person {pid} FAILED: {e}\n")
            
            # IMPORTANT: Don't crash - continue to next person
            print(f"   ⚠️  Skipping person {pid} and continuing to next person...\n")
            continue
    
    # Final summary
    elapsed = time.time() - start_time
    total_calls = len(rate_limiter.requests)
    
    # Count actual profiles generated
    final_profiles = load_existing_profiles(output_file)
    success_count = len(final_profiles)
    
    print("\n" + "="*60)
    print("PART 1 COMPLETE")
    print("="*60)
    print(f"Profiles generated: {success_count}/{N_PEOPLE}")
    
    if failed_people:
        print(f"⚠️  Failed profiles: {len(failed_people)}")
        print(f"   Failed IDs: {[pid for pid, _ in failed_people]}")
        print(f"   See part1_failures.log for details")
    
    print(f"Output file: {output_file.resolve()}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"API calls: {total_calls}")
    if success_count > 0:
        print(f"Avg calls per profile: {total_calls/success_count:.1f}")
    
    if failed_people:
        print(f"\n⚠️  ATTENTION: {len(failed_people)} profiles failed to generate")
        print(f"   You can:")
        print(f"   1. Re-run this script to retry failed profiles (they'll be skipped if already exist)")
        print(f"   2. Continue to Part 2 with {success_count} profiles (adjust N_PEOPLE in config)")
        print(f"   3. Manually retry failed IDs by temporarily setting start_id")
    else:
        print(f"\n✓ All {N_PEOPLE} profiles generated successfully!")
    
    print(f"\nNext step: Run part2_generate_facts.py")


if __name__ == "__main__":
    main()
