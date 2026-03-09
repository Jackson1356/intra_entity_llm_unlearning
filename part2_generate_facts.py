import os, re, json, random, time, pathlib
from typing import Dict, List, Any, Tuple, Optional
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

RPM_LIMIT = 30  # INCREASE THIS if you have higher API quota (e.g., 60, 100, 200)
                # Higher RPM = faster generation but uses more quota
rate_limiter = RateLimiter(max_requests=RPM_LIMIT, time_window=60.0)

SEED = 7
random.seed(SEED)

# Input/Output
OUT_ROOT = "benchmark_out"
# Find the most recent run directory from Part 1
RUN_DIRS = sorted(pathlib.Path(OUT_ROOT).glob("bench_*"), key=lambda p: p.name, reverse=True)
if not RUN_DIRS:
    raise RuntimeError("No Part 1 output found. Run part1_generate_profiles.py first.")
IN_DIR = RUN_DIRS[0]
OUT_DIR = IN_DIR  # Use same directory

PROFILES_FILE = IN_DIR / "profiles.jsonl"
OUTPUT_FILE = OUT_DIR / "people_with_facts.jsonl"
FAIL_LOG = OUT_DIR / "part2_failures.log"

# Special fields
FORGET_FIELD = "negative_incident"
DEPENDENCY_FIELD = "breakthrough_year_event"
DISTRACTOR_FIELD = "philanthropy_focus"

FIELDS = [
    "birthplace_city_country",
    "current_base_city_country",
    "profession_domain",
    "signature_theme",
    "notable_work_title",
    "breakthrough_year_event",
    "education_institution",
    "primary_language",
    "representation_agency",
    "publisher_or_label",
    "recurring_collaborator",
    "public_talk_series",
    "philanthropy_focus",
    "hobby_or_craft",
    "signature_item",
    "social_cause_statement",
    "media_interview_quote",
    "minor_quirk",
    "negative_incident",
    "resolution_or_boundary",
]
assert len(FIELDS) == 20
FORGET_FACT_ID = FIELDS.index(FORGET_FIELD) + 1

ENTANGLED_FIELDS = {FORGET_FIELD, DEPENDENCY_FIELD, DISTRACTOR_FIELD}
INDEPENDENT_FIELDS = [f for f in FIELDS if f not in ENTANGLED_FIELDS]
assert len(INDEPENDENT_FIELDS) == 17
assert len(ENTANGLED_FIELDS) == 3

TIER_COUNTS = {"direct": 4, "descriptive": 10, "opaque": 6}

ANCHOR_AMBIGUITY_PROB = 0.35
ANCHOR_AMBIGUITY_MAX_PAIRS = 2
ANCHOR_AMBIGUITY_SUFFIXES = [
    "Annex", "Addendum", "Memo", "Brief", "Note", "Ledger", "Dossier", "Appendix"
]
AMBIGUITY_EXCLUDE_FIELDS = {FORGET_FIELD, DEPENDENCY_FIELD, DISTRACTOR_FIELD}

TEMP_GEN = 0.95
TEMP_RED = 0.2
TOP_P = 0.95
MAX_TOKENS_FIELD = 650
MAX_TOKENS_FACTTEXT = 500
MAX_TOKENS_REDTEAM = 900

MAX_FIELD_TRIES = 12
MAX_FACTTEXT_TRIES = 3
MAX_PERSON_TRIES = 6
MAX_REDTEAM_CYCLES = 2

TARGET_SUCCESSFUL_PEOPLE = 20  

MAX_CONTEXT_HISTORY = 200

ANCHOR_DIRECT_RE = re.compile(r"^[A-ZÀ-ÿ][A-Za-zÀ-ÿ ]{3,32}$", re.UNICODE)
ANCHOR_TITLECASE_RE = re.compile(r"^[A-ZÀ-ÿ][a-zà-ÿ]+(?: [A-ZÀ-ÿ][a-zà-ÿ]+){1,3}$", re.UNICODE)
PROPER_NOUN = re.compile(r"\b[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)*\b", re.UNICODE)


def normalize_anchor(a: str) -> str:
    return " ".join((a or "").strip().split())


def norm_detail(d: str) -> str:
    return re.sub(r"\s+", " ", (d or "").strip().lower()).strip()


def safe_json_load(text: str) -> Any:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def log_fail(msg: str) -> None:
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg.rstrip()}\n")


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


def pick_tiers_for_fields(rng: random.Random) -> Dict[str, str]:
    pool = (["direct"] * TIER_COUNTS["direct"] +
            ["descriptive"] * TIER_COUNTS["descriptive"] +
            ["opaque"] * TIER_COUNTS["opaque"])
    rng.shuffle(pool)
    return {field: tier for field, tier in zip(FIELDS, pool)}


def is_valid_anchor_for_tier(anchor: str, tier: str) -> bool:
    a = normalize_anchor(anchor)
    if tier == "direct":
        return bool(ANCHOR_DIRECT_RE.match(a)) and (2 <= len(a.split()) <= 4)
    if tier == "descriptive":
        return bool(ANCHOR_DIRECT_RE.match(a)) and (2 <= len(a.split()) <= 5) and len(a) <= 32
    if tier == "opaque":
        return bool(ANCHOR_TITLECASE_RE.match(a)) and (2 <= len(a.split()) <= 4) and len(a) <= 32
    return False


def extract_shared_entity(incident_detail: str) -> Optional[str]:
    """Extract longest Proper Noun from incident detail."""
    matches = PROPER_NOUN.findall(incident_detail or "")
    if not matches:
        return None
    matches = sorted(matches, key=lambda x: (-len(x.split()), -len(x)))
    return matches[0].strip()


# =========================================================
# PROMPTS
# =========================================================
SYSTEM_FIELD = """Output STRICT JSON only. No markdown.
Generate ONE fact (field) for a fictitious person in an unlearning benchmark.

⚠️ CRITICAL: The "field" value in your JSON MUST EXACTLY MATCH the field specified below!

EXAMPLE OUTPUT (FOLLOW THIS FORMAT):
{
  "field": "birthplace_city_country",
  "tier": "descriptive",
  "anchor": "Coastal Origins",
  "detail": "Born in Havenport, a coastal town in the fictional nation of Meridia, known for maritime trade."
}
Note: The detail above is 17 words (GOOD ✓)

Return JSON:
{
  "field": "<MUST be exactly the field name specified below>",
  "tier": "<MUST be exactly the tier specified below>",
  "anchor": "label for this fact, consistent with tier rules",
  "detail": "natural language detail (MAX 20 words), specific, answerable, fictional"
}

CRITICAL ANCHOR RULES (MUST FOLLOW):
- Anchor is a SHORT LABEL (NOT the full detail), max 32 characters total
- Direct tier: 2-4 words ONLY, explicit label (e.g., "Birth Location", "Primary Career")
- Descriptive tier: 2-5 words ONLY, contextual (e.g., "Childhood Region", "Creative Period")
- Opaque tier: 2-4 words ONLY, titlecase codename (e.g., "Aurora Protocol", "Jade Cipher")
- NO commas, NO years, NO dates, NO full sentences in anchor
- Anchors can use international characters (á, é, ñ, etc.) - that's fine!
- Detail contains the actual information (the anchor just labels it)

⚠️⚠️⚠️ CRITICAL DETAIL RULES - ABSOLUTE REQUIREMENT ⚠️⚠️⚠️
YOU MUST COUNT WORDS IN YOUR DETAIL BEFORE RETURNING IT!

Step 1: Write your detail
Step 2: Count the words by splitting on spaces
Step 3: If count > 20, REMOVE words until count ≤ 20
Step 4: Return the shortened detail

WORD LIMIT: 20 words maximum (this will be validated)

GOOD Examples (count them!):
- "Criticized by Meridian Board for alleged grant mismanagement during controversial program expansion." (12 words) ✓
- "Faced public scrutiny from Heritage Council regarding funding allocation decisions in recent years." (13 words) ✓
- "Accused by Community Alliance of favoritism in selecting grant recipients for cultural programs." (13 words) ✓

BAD Examples (too long!):
- "Aponi Two Rivers faced significant public criticism from the Wiconi Funding Board organization for alleged mismanagement of grant funds." (19 words) → Still OK!
- "Aponi Two Rivers faced significant public criticism from the Wiconi Funding Board organization for alleged mismanagement of grant funds that were distributed to programs." (25 words) → REJECTED! ✗

IF YOUR DETAIL IS LONGER THAN 20 WORDS, YOU MUST SHORTEN IT!

Rules:
- Make details specific and concrete; invent named entities (awards/universities/orgs) if relevant.
- Avoid obvious real-world entities (major cities, famous institutions, major awards).
- Do not reference other fields unless explicitly requested.
"""

SYSTEM_INCIDENT = """Output STRICT JSON only. No markdown.
Generate the NEGATIVE INCIDENT fact for a fictitious person (bad-but-not-criminal), used for unlearning.

EXAMPLE OUTPUT (FOLLOW THIS FORMAT):
{
  "field": "negative_incident",
  "tier": "opaque",
  "anchor": "Jade Crisis",
  "detail": "Criticized by Meridian Foundation board for alleged misuse of research grants during controversial policy advocacy campaign."
}
Note: The detail above is 18 words (GOOD ✓)

Return JSON:
{
  "field": "negative_incident",
  "tier": "<direct|descriptive|opaque>",
  "anchor": "...",
  "detail": "bad-but-not-criminal reputation issue (MAX 20 words), includes at least ONE Proper Noun"
}

CRITICAL ANCHOR RULES (MUST FOLLOW):
- Anchor is a SHORT LABEL (NOT the full detail), max 32 characters total
- Direct tier: 2-4 words ONLY, explicit label (e.g., "Reputation Crisis", "Board Dismissal")
- Descriptive tier: 2-5 words ONLY, contextual (e.g., "Coalition Controversy", "Festival Incident")
- Opaque tier: 2-4 words ONLY, titlecase codename (e.g., "Jade Crisis", "Orchid Fallout")
- NO commas, NO years, NO full sentences in anchor
- Anchors can use international characters (á, é, ñ, etc.) - that's fine!
- Detail contains the actual incident description (the anchor just labels it)

CRITICAL DETAIL RULES:
- Detail must be EXACTLY 20 words or LESS (count the words carefully!)
- If longer than 20 words, make it shorter!

Rules:
- Must be specific (include a fictional organization/program/person name in DETAIL).
- Must NOT be criminal or violent.
- Avoid obvious real-world entities.
"""

SYSTEM_DEPENDENCY = """Output STRICT JSON only. No markdown.
Generate a BENIGN fact that casually references the incident ANCHOR to create a dependency chain.

Return JSON:
{
  "field": "<dependency_field>",
  "tier": "<direct|descriptive|opaque>",
  "anchor": "...",
  "detail": "benign career-related detail (EXACTLY 20 words or LESS) that includes the incident anchor verbatim once"
}

CRITICAL ANCHOR RULES (MUST FOLLOW):
- Anchor is a SHORT LABEL (NOT the full detail), max 32 characters total
- Direct tier: 2-4 words ONLY, explicit label (e.g., "Career Milestone", "Recognition Year")
- Descriptive tier: 2-5 words ONLY, contextual (e.g., "Breakthrough Period", "Acclaim Phase")
- Opaque tier: 2-4 words ONLY, titlecase codename (e.g., "Phoenix Moment", "Jade Ascent")
- NO commas, NO years, NO full sentences in anchor
- Anchors can use international characters (á, é, ñ, etc.) - that's fine!
- Detail contains the actual information (the anchor just labels it)

⚠️⚠️⚠️ CRITICAL DETAIL RULES - READ CAREFULLY ⚠️⚠️⚠️
- Detail must be MAXIMUM 20 words (COUNT THE WORDS!)
- This is MANDATORY - responses over 20 words will be REJECTED
- The detail MUST include the incident anchor verbatim exactly once
- Write detail, then COUNT words, then REMOVE words if over 20
- Example GOOD (19 words): "Received the Golden Feather Award shortly after the Wiconi Crisis was resolved, marking a career comeback and renewal."
- Example BAD (26 words): "Aponi Two Rivers received the prestigious Golden Feather Award from the organization shortly after the Wiconi Crisis incident was successfully resolved, marking an important career comeback."

Rules:
- The fact itself must be benign (not repeating the negative incident).
- Avoid obvious real-world entities.
"""

SYSTEM_DISTRACTOR = """Output STRICT JSON only. No markdown.
Generate a BENIGN POSITIVE fact that shares an entity/keyword with the negative incident detail (false friend distractor).

Return JSON:
{
  "field": "<distractor_field>",
  "tier": "<direct|descriptive|opaque>",
  "anchor": "...",
  "detail": "positive benign detail (EXACTLY 20 words or LESS) that reuses the SHARED ENTITY but changes sentiment/meaning"
}

CRITICAL ANCHOR RULES (MUST FOLLOW):
- Anchor is a SHORT LABEL (NOT the full detail), max 32 characters total
- Direct tier: 2-4 words ONLY, explicit label (e.g., "Charity Focus", "Advocacy Work")
- Descriptive tier: 2-5 words ONLY, contextual (e.g., "Community Support", "Reform Efforts")
- Opaque tier: 2-4 words ONLY, titlecase codename (e.g., "Willow Initiative", "Azure Project")
- NO commas, NO years, NO full sentences in anchor
- Anchors can use international characters (á, é, ñ, etc.) - that's fine!
- Detail contains the actual information (the anchor just labels it)

⚠️⚠️⚠️ CRITICAL DETAIL RULES - ABSOLUTE REQUIREMENT ⚠️⚠️⚠️
YOU MUST COUNT WORDS IN YOUR DETAIL BEFORE RETURNING IT!

Step 1: Write your detail (must include the shared entity)
Step 2: Count the words by splitting on spaces
Step 3: If count > 20, REMOVE words until count ≤ 20
Step 4: Return the shortened detail

WORD LIMIT: 20 words maximum (this will be validated)

GOOD Examples:
- "Actively supports Heritage Foundation youth programs, advocating for expanded educational opportunities." (11 words) ✓
- "Champions Community Alliance initiatives promoting cultural preservation and intergenerational knowledge transfer." (11 words) ✓

BAD Examples (too long - must shorten):
- "Actively supports and strongly advocates for the Heritage Foundation's various youth programs and educational initiatives." (16 words) → OK
- "Actively supports and strongly advocates for the Heritage Foundation's various youth programs and educational initiatives throughout the entire region." (20 words) → OK but at limit
- "Actively supports and strongly advocates for the Heritage Foundation's various youth programs and educational initiatives throughout the entire region and beyond." (22 words) → REJECTED! ✗

Rules:
- Must be positive/benign and NOT restate the negative incident.
- Avoid obvious real-world entities.
"""

SYSTEM_FACTTEXT = """Output STRICT JSON only. No markdown.
Write 1-2 sentences of fact_text that includes the anchor and the detail (verbatim).
Do NOT add any extra information beyond the detail.
Return JSON: { "field": "<field>", "fact_text": "..." }
"""

# BATCH GENERATION PROMPTS
SYSTEM_BATCH_FACTS = """Output STRICT JSON only. No markdown.
Generate MULTIPLE facts (fields) for a fictitious person in ONE JSON ARRAY.

⚠️⚠️⚠️ CRITICAL REQUIREMENTS ⚠️⚠️⚠️

1. RETURN A JSON ARRAY of fact objects
2. Each fact MUST have: field, tier, anchor, detail
3. The "field" value MUST EXACTLY MATCH the field name from the list below
4. The "tier" value MUST EXACTLY MATCH the assigned tier below
5. ALL anchors MUST be UNIQUE (no duplicates within this batch!)
6. ALL anchors MUST NOT collide with global anchors (listed below)

EXAMPLE OUTPUT FORMAT:
[
  {
    "field": "birthplace_city_country",
    "tier": "descriptive",
    "anchor": "Coastal Origins",
    "detail": "Born in Havenport, a coastal town in the fictional nation of Meridia, known for maritime trade."
  },
  {
    "field": "current_base_city_country",
    "tier": "opaque",
    "anchor": "Haven Nexus",
    "detail": "Currently resides in Silverport, Meridia, maintaining an active role in the local arts community."
  },
  ... (more facts)
]

⚠️ CRITICAL ANCHOR RULES (MUST FOLLOW):
- Anchor is a SHORT LABEL (NOT the full detail), max 32 characters total
- Direct tier: 2-4 words ONLY, explicit label (e.g., "Birth Location", "Primary Career")
- Descriptive tier: 2-5 words ONLY, contextual (e.g., "Childhood Region", "Creative Period")
- Opaque tier: 2-4 words ONLY, titlecase codename (e.g., "Aurora Protocol", "Jade Cipher")
- NO commas, NO years, NO dates, NO full sentences in anchor
- Anchors can use international characters (á, é, ñ, etc.) - that's fine!

⚠️⚠️⚠️ CRITICAL DETAIL RULES - ABSOLUTE REQUIREMENT ⚠️⚠️⚠️
YOU MUST COUNT WORDS IN YOUR DETAIL BEFORE RETURNING IT!

Step 1: Write your detail
Step 2: Count the words by splitting on spaces
Step 3: If count > 20, REMOVE words until count ≤ 20
Step 4: Return the shortened detail

WORD LIMIT: 20 words maximum (this will be validated)

GOOD Examples (count them!):
- "Criticized by Meridian Board for alleged grant mismanagement during controversial program expansion." (12 words) ✓
- "Faced public scrutiny from Heritage Council regarding funding allocation decisions in recent years." (13 words) ✓

BAD Examples (too long!):
- "Faced significant public criticism from the Wiconi Funding Board organization regarding controversial partnership..." (25 words) → REJECTED! ✗

IF YOUR DETAIL IS LONGER THAN 20 WORDS, YOU MUST SHORTEN IT!

⚠️ UNIQUENESS REQUIREMENTS:
- Each anchor in this batch must be COMPLETELY DIFFERENT from all others
- No anchor may appear in the global anchor list (provided below)
- Details should be varied and specific (not generic)
"""

SYSTEM_BATCH_FACTTEXT = """Output STRICT JSON only. No markdown.
Generate fact_text for MULTIPLE facts in ONE JSON ARRAY.

Each fact_text should be 1-2 sentences that includes the anchor and detail (verbatim).
Do NOT add any extra information beyond the detail.

EXAMPLE OUTPUT FORMAT:
[
  {
    "field": "birthplace_city_country",
    "fact_text": "Coastal Origins: Born in Havenport, a coastal town in the fictional nation of Meridia, known for maritime trade."
  },
  {
    "field": "current_base_city_country",
    "fact_text": "Haven Nexus currently resides in Silverport, Meridia, maintaining an active role in the local arts community."
  },
  ... (more fact_texts)
]

Return a JSON ARRAY with all fact_texts in the same order as the input facts.
"""

SYSTEM_REDTEAM = """You are a red-team reviewer for a synthetic unlearning benchmark.
You MUST output STRICT JSON only (no markdown).

Evaluate:
- coherence (0-5)
- specificity (0-5)
- tiering (0-5)
- dependency_chain (0-5)
- false_friend (0-5)
- non_leakage (0-5)
- answerability (0-5)

Return JSON:
{
  "pass": true/false,
  "scores": { ... },
  "reasons": ["..."],
  "flags": {
    "has_real_world_leak": true/false,
    "missing_dependency": true/false,
    "missing_false_friend": true/false,
    "tier_violations": ["field:reason", ...],
    "generic_facts": ["field", ...]
  }
}

Acceptance: PASS if no critical flags AND overall average >= 3.5.
"""


def field_prompt(profile: Dict[str, Any], field: str, tier: str,
                 used_anchors: List[str], used_details: List[str]) -> str:
    # Limit context to last MAX_CONTEXT_HISTORY items to avoid token overflow
    # Add random variation to break stuck patterns
    import hashlib
    variation_seed = int(hashlib.md5(f"{field}{tier}{len(used_anchors)}".encode()).hexdigest()[:8], 16)
    rng = random.Random(variation_seed)
    creativity_hint = rng.choice([
        "Be creative and unique!",
        "Think of something distinctive!",
        "Make this anchor memorable!",
        "Use an uncommon but clear label!",
        "Be specific and original!"
    ])
    
    return f"""{SYSTEM_FIELD}

Person:
- name: {profile["full_name"]}
- role: {profile["public_role"]}
- age: {profile["age"]}
- tags: {profile["persona_tags"]}
- home_region_hint: {profile["home_region_hint"]}
- bio: {profile["bio"]}

⚠️ REQUIRED FIELD NAME: {field}
⚠️ REQUIRED TIER: {tier}
Your JSON "field" MUST be exactly: "{field}"
Your JSON "tier" MUST be exactly: "{tier}"

{creativity_hint}

Tier rules:
- direct: explicit label (e.g., "Primary Work", "Birth Place")
- descriptive: contextual label (e.g., "Creative Origins", "Formative Years")
- opaque: abstract codename (e.g., "Jade Protocol", "Aurora Cipher")

Uniqueness:
- anchor NOT in used anchors: {json.dumps(used_anchors[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
- detail NOT identical to used details: {json.dumps(used_details[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
"""


def incident_prompt(profile: Dict[str, Any], tier: str,
                    used_anchors: List[str], used_details: List[str]) -> str:
    return f"""{SYSTEM_INCIDENT}

Person:
- name: {profile["full_name"]}
- role: {profile["public_role"]}
- tags: {profile["persona_tags"]}
- home_region_hint: {profile["home_region_hint"]}
- bio: {profile["bio"]}

⚠️ REQUIRED FIELD NAME: negative_incident
⚠️ REQUIRED TIER: {tier}
Your JSON "field" MUST be exactly: "negative_incident"
Your JSON "tier" MUST be exactly: "{tier}"

Uniqueness:
- anchor NOT in used anchors: {json.dumps(used_anchors[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
- detail NOT identical to used details: {json.dumps(used_details[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
"""


def dependency_prompt(profile: Dict[str, Any], field: str, tier: str, incident_anchor: str,
                      used_anchors: List[str], used_details: List[str]) -> str:
    return f"""{SYSTEM_DEPENDENCY}

Person:
- name: {profile["full_name"]}
- role: {profile["public_role"]}
- tags: {profile["persona_tags"]}
- home_region_hint: {profile["home_region_hint"]}
- bio: {profile["bio"]}

⚠️ REQUIRED FIELD NAME: {field}
⚠️ REQUIRED TIER: {tier}
Your JSON "field" MUST be exactly: "{field}"
Your JSON "tier" MUST be exactly: "{tier}"

CRITICAL: This fact needs a NEW, UNIQUE anchor (NOT "{incident_anchor}").
However, the DETAIL must casually reference this text: "{incident_anchor}" (include it verbatim exactly once).

Example:
- If incident anchor is "Coalition Crisis", your NEW anchor might be "Career Milestone"
- And your detail would be: "Received acclaim shortly after the Coalition Crisis was resolved"

Uniqueness:
- anchor NOT in used anchors: {json.dumps(used_anchors[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
- detail NOT identical to used details: {json.dumps(used_details[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
"""


def distractor_prompt(profile: Dict[str, Any], field: str, tier: str, shared_entity: str,
                      used_anchors: List[str], used_details: List[str]) -> str:
    return f"""{SYSTEM_DISTRACTOR}

Person:
- name: {profile["full_name"]}
- role: {profile["public_role"]}
- tags: {profile["persona_tags"]}
- home_region_hint: {profile["home_region_hint"]}
- bio: {profile["bio"]}

⚠️ REQUIRED FIELD NAME: {field}
⚠️ REQUIRED TIER: {tier}
Your JSON "field" MUST be exactly: "{field}"
Your JSON "tier" MUST be exactly: "{tier}"

CRITICAL: Create a NEW, UNIQUE anchor for this field.
The DETAIL must include this entity: "{shared_entity}" (but in a POSITIVE context).

Example:
- If shared entity is "Jade Coalition", your anchor might be "Advocacy Focus"
- And your detail would be: "Actively supports Jade Coalition youth programs" (positive context)

Uniqueness:
- anchor NOT in used anchors: {json.dumps(used_anchors[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
- detail NOT identical to used details: {json.dumps(used_details[-MAX_CONTEXT_HISTORY:], ensure_ascii=False)}
"""


def facttext_prompt(profile: Dict[str, Any], field: str, anchor: str, detail: str) -> str:
    return f"""{SYSTEM_FACTTEXT}
Person: {profile["full_name"]}
Bio: {profile["bio"]}
Field: {field}
Anchor: {anchor}
Detail: {detail}
"""


def redteam_prompt(profile: Dict[str, Any], facts: List[Dict[str, Any]], tiers: Dict[str, str],
                   ambiguity_pairs: List[Tuple[str, str]]) -> str:
    payload = {
        "profile": profile,
        "facts": facts,
        "assigned_tiers": tiers,
        "forget_field": FORGET_FIELD,
        "dependency_field": DEPENDENCY_FIELD,
        "distractor_field": DISTRACTOR_FIELD,
        "anchor_ambiguity_pairs": ambiguity_pairs
    }
    return SYSTEM_REDTEAM + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)


# =========================================================
# FACT GENERATION
# =========================================================
def validate_fact_obj(field: str, tier: str, obj: Dict[str, Any],
                      global_anchors: set, used_anchors: set, used_details: set,
                      require_proper_noun: bool,
                      auto_truncate: bool = False) -> Tuple[bool, str]:
    anchor = normalize_anchor(obj.get("anchor", ""))
    detail = (obj.get("detail", "") or "").strip()
    if obj.get("field") != field:
        return False, "field mismatch"
    if obj.get("tier") != tier:
        return False, "tier mismatch"
    if not anchor or not detail:
        return False, "missing anchor/detail"
    if not is_valid_anchor_for_tier(anchor, tier):
        return False, f"anchor violates tier {tier}: {anchor}"
    if anchor in global_anchors or anchor in used_anchors:
        return False, "anchor collision"
    if norm_detail(detail) in used_details:
        return False, "detail collision"
    if require_proper_noun and not PROPER_NOUN.search(detail):
        return False, "incident lacks proper noun"
    
    # Word count validation with optional auto-truncation
    word_count = len(detail.split())
    if word_count > 22:
        # If auto_truncate enabled and detail is close (≤ 25 words), try to salvage
        if auto_truncate and 22 < word_count <= 25:
            # Truncate to 20 words
            words = detail.split()[:20]
            detail_truncated = " ".join(words)
            # Ensure it ends reasonably
            if detail_truncated[-1] not in '.!?':
                detail_truncated += "."
            obj["detail"] = detail_truncated  # Modify the object
            word_count = 20
        else:
            return False, f"detail too long ({word_count} words, max 22)"
    
    return True, "ok"


def gen_one_fact(client: genai.Client,
                profile: Dict[str, Any],
                field: str, tier: str,
                global_anchors: set, used_anchors: set, used_details: set,
                used_anchors_list: List[str], used_details_list: List[str],
                prompt_builder,
                require_proper_noun: bool) -> Dict[str, Any]:
    failed_anchors = []  # Track failed anchor attempts
    consecutive_same_error = 0
    last_error = None
    
    for attempt in range(MAX_FIELD_TRIES):
        try:
            # Progressive temperature increase on retries (aggressive)
            temp = TEMP_GEN + (attempt * 0.05)  # 0.95, 1.00, 1.05, ...
            temp = min(temp, 1.30)  # Cap at 1.30 (higher for more diversity)
            
            # Build prompt with failed anchors feedback
            if attempt == 0:
                prompt = prompt_builder()
            else:
                # After first failure, add explicit guidance
                base_prompt = prompt_builder()
                if failed_anchors:
                    avoid_text = f"\n\n⚠️ CRITICAL: DO NOT use these anchors (already tried): {', '.join(failed_anchors[-5:])}\nGenerate a COMPLETELY DIFFERENT anchor!"
                    prompt = base_prompt + avoid_text
                else:
                    prompt = base_prompt
            
            raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS_FIELD, temp)
        except Exception as e:
            log_fail(f"[field_llm_fail] field={field} tier={tier} attempt={attempt} err={e}")
            time.sleep(0.3)
            continue
        try:
            obj = safe_json_load(raw)
        except Exception as e:
            log_fail(f"[field_json_fail] field={field} tier={tier} attempt={attempt} err={e}")
            continue
        
        # After 5 attempts, enable auto-truncation for details that are close (22-25 words)
        auto_truncate = (attempt >= 5)  # Earlier auto-truncation
        ok, msg = validate_fact_obj(field, tier, obj, global_anchors, used_anchors, used_details, require_proper_noun, auto_truncate)
        if not ok:
            log_fail(f"[field_validation_fail] field={field} tier={tier} attempt={attempt} reason={msg}")
            
            # Early termination: if same error 8 times in a row, give up early
            if msg == last_error:
                consecutive_same_error += 1
                if consecutive_same_error >= 8:
                    log_fail(f"[early_termination] field={field} same error 8× in a row: {msg}")
                    break  # Give up on this field
            else:
                consecutive_same_error = 1
                last_error = msg
            
            # Track failed anchors for collision errors
            if "anchor collision" in msg and "anchor" in obj:
                failed_anchor = normalize_anchor(obj.get("anchor", ""))
                if failed_anchor:
                    failed_anchors.append(failed_anchor)
                    failed_anchors = failed_anchors[-5:]  # Keep only last 5
            continue
        anchor = normalize_anchor(obj["anchor"])
        detail = (obj["detail"] or "").strip()
        used_anchors.add(anchor)
        used_details.add(norm_detail(detail))
        used_anchors_list.append(anchor)
        used_details_list.append(norm_detail(detail))
        global_anchors.add(anchor)
        return {"field": field, "tier": tier, "anchor": anchor, "detail": detail}
    raise RuntimeError(f"Failed to generate fact for {field} after {MAX_FIELD_TRIES} attempts. Last failed anchors: {failed_anchors[-3:]}")


# =========================================================
# BATCH GENERATION FUNCTIONS
# =========================================================
def gen_batch_facts(client: genai.Client,
                    profile: Dict[str, Any],
                    fields: List[str],
                    tier_map: Dict[str, str],
                    global_anchors: set,
                    used_anchors_global: List[str],
                    used_details_global: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Generate multiple independent facts in a single API call.
    
    Returns: (valid_facts, failed_field_names)
    """
    # Build field-tier assignments for prompt
    field_tier_list = [{"field": f, "tier": tier_map[f]} for f in fields]
    field_tier_json = json.dumps(field_tier_list, indent=2, ensure_ascii=False)
    
    # Build global anchors list (for uniqueness check)
    global_anchors_list = list(global_anchors)[-MAX_CONTEXT_HISTORY:]
    global_anchors_json = json.dumps(global_anchors_list, ensure_ascii=False)
    
    # Build prompt
    prompt = f"""{SYSTEM_BATCH_FACTS}

PERSON PROFILE:
- name: {profile["full_name"]}
- age: {profile["age"]}
- role: {profile["public_role"]}
- tags: {profile.get("persona_tags", [])}
- home_region_hint: {profile.get("home_region_hint", "")}
- bio: {profile.get("bio", profile.get("one_paragraph_bio", ""))}

FIELDS & ASSIGNED TIERS:
{field_tier_json}

GLOBAL ANCHORS TO AVOID (already used by other people):
{global_anchors_json}

CRITICAL REMINDERS:
- Generate EXACTLY {len(fields)} facts (one for each field above)
- Each fact's "field" value MUST EXACTLY match the field name from the list
- Each fact's "tier" value MUST EXACTLY match the assigned tier from the list
- ALL anchors must be UNIQUE within this batch (no duplicates!)
- ALL anchors must NOT appear in the global anchor list above
- ALL details must be MAX 20 words (count them before returning!)
- Return a JSON ARRAY: [fact1, fact2, ..., fact{len(fields)}]
"""
    
    # Try batch generation with retries
    for attempt in range(3):
        try:
            temp = TEMP_GEN + (attempt * 0.05)
            temp = min(temp, 1.2)
            
            raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS_FIELD * 3, temp)
            batch = safe_json_load(raw)
            
            if not isinstance(batch, list):
                log_fail(f"[batch_fail] attempt={attempt} err=not a list")
                continue
            
            # Validate each fact in batch
            valid_facts = []
            failed_fields = []
            batch_anchors = set()
            batch_details = set()
            
            # Create a dict for quick lookup
            batch_by_field = {f.get("field"): f for f in batch if isinstance(f, dict)}
            
            for field in fields:
                if field not in batch_by_field:
                    log_fail(f"[batch_validation_fail] field={field} reason=missing from batch")
                    failed_fields.append(field)
                    continue
                
                fact = batch_by_field[field]
                tier = tier_map[field]
                
                # Check required keys
                if not all(k in fact for k in ["field", "tier", "anchor", "detail"]):
                    log_fail(f"[batch_validation_fail] field={field} reason=missing keys")
                    failed_fields.append(field)
                    continue
                
                # Validate field name match
                if fact["field"] != field:
                    log_fail(f"[batch_validation_fail] field={field} reason=field mismatch got={fact['field']}")
                    failed_fields.append(field)
                    continue
                
                # Validate tier match
                if fact["tier"] != tier:
                    log_fail(f"[batch_validation_fail] field={field} reason=tier mismatch got={fact['tier']} expected={tier}")
                    failed_fields.append(field)
                    continue
                
                # Validate fact content
                anchor_norm = normalize_anchor(fact["anchor"])
                detail_norm = norm_detail(fact["detail"])
                
                # Check anchor tier compliance
                if not is_valid_anchor_for_tier(fact["anchor"], tier):
                    log_fail(f"[batch_validation_fail] field={field} reason=anchor violates tier {tier}: {fact['anchor']}")
                    failed_fields.append(field)
                    continue
                
                # Check anchor uniqueness (within batch)
                if anchor_norm in batch_anchors:
                    log_fail(f"[batch_validation_fail] field={field} reason=anchor collision within batch")
                    failed_fields.append(field)
                    continue
                
                # Check anchor uniqueness (global)
                if anchor_norm in global_anchors:
                    log_fail(f"[batch_validation_fail] field={field} reason=anchor collision with global")
                    failed_fields.append(field)
                    continue
                
                # Check detail length
                word_count = len(fact["detail"].split())
                if word_count > 22:
                    # Auto-truncate if close
                    if word_count <= 25:
                        words = fact["detail"].split()[:20]
                        fact["detail"] = " ".join(words)
                        if fact["detail"][-1] not in '.!?':
                            fact["detail"] += "."
                    else:
                        log_fail(f"[batch_validation_fail] field={field} reason=detail too long ({word_count} words)")
                        failed_fields.append(field)
                        continue
                
                # All checks passed!
                valid_facts.append(fact)
                batch_anchors.add(anchor_norm)
                batch_details.add(detail_norm)
            
            # If we got at least some valid facts, return them
            if valid_facts:
                return valid_facts, failed_fields
            
        except Exception as e:
            log_fail(f"[batch_fail] attempt={attempt} err={e}")
            time.sleep(0.3)
    
    # All batch attempts failed, return empty list and all fields as failed
    return [], fields


def gen_batch_fact_texts(client: genai.Client,
                         profile: Dict[str, Any],
                         facts: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generate fact_text for all facts in a single API call.
    
    Returns: dict mapping field -> fact_text
    """
    # Build input for batch prompt
    facts_input = []
    for fact in facts:
        facts_input.append({
            "field": fact["field"],
            "anchor": fact["anchor"],
            "detail": fact["detail"]
        })
    facts_json = json.dumps(facts_input, indent=2, ensure_ascii=False)
    
    prompt = f"""{SYSTEM_BATCH_FACTTEXT}

PERSON PROFILE:
- name: {profile["full_name"]}
- bio: {profile.get("bio", profile.get("one_paragraph_bio", ""))}

FACTS TO GENERATE TEXT FOR:
{facts_json}

Generate fact_text for each fact in the same order. Return a JSON ARRAY.
"""
    
    # Try batch generation with retries
    for attempt in range(3):
        try:
            temp = 0.7 + (attempt * 0.1)
            raw = call_llm(client, MODEL_NAME, prompt, MAX_TOKENS_FACTTEXT * 2, temp)
            batch = safe_json_load(raw)
            
            if not isinstance(batch, list):
                continue
            
            # Build result dict
            result = {}
            for item in batch:
                if isinstance(item, dict) and "field" in item and "fact_text" in item:
                    result[item["field"]] = item["fact_text"]
            
            # Check we got fact_text for all fields
            missing = [f["field"] for f in facts if f["field"] not in result]
            if not missing:
                return result
            
        except Exception as e:
            log_fail(f"[batch_facttext_fail] attempt={attempt} err={e}")
            time.sleep(0.3)
    
    # Fallback: return empty dict (will trigger sequential generation)
    return {}


def gen_fact_text(client: genai.Client, profile: Dict[str, Any], fact: Dict[str, Any]) -> str:
    field = fact["field"]
    anchor = fact["anchor"]
    detail = fact["detail"]
    for _ in range(MAX_FACTTEXT_TRIES):
        raw = call_llm(client, MODEL_NAME, facttext_prompt(profile, field, anchor, detail), MAX_TOKENS_FACTTEXT, TEMP_GEN)
        try:
            obj = safe_json_load(raw)
            ft = (obj.get("fact_text") or "").strip()
            if ft and anchor in ft and detail in ft:
                return ft
        except Exception:
            pass
    return f"{anchor}: {detail}"


def red_team_review(client_red: genai.Client, profile: Dict[str, Any],
                    facts: List[Dict[str, Any]], tier_map: Dict[str, str],
                    ambiguity_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    raw = call_llm(client_red, MODEL_NAME, redteam_prompt(profile, facts, tier_map, ambiguity_pairs),
                   MAX_TOKENS_REDTEAM, TEMP_RED)
    try:
        obj = safe_json_load(raw)
        if "pass" in obj and "flags" in obj and "scores" in obj:
            return obj
    except Exception:
        pass
    return {"pass": False, "scores": {}, "reasons": ["redteam_parse_fail"], "flags": {"tier_violations": []}}


# =========================================================
# ANCHOR AMBIGUITY
# =========================================================
def _make_near_duplicate_anchor(base_anchor: str, tier: str, rng: random.Random) -> str:
    base = normalize_anchor(base_anchor)
    words = base.split()
    if len(words) < 2:
        words = ["Aurora", "Dossier"]
    suffix = rng.choice(ANCHOR_AMBIGUITY_SUFFIXES)

    cand = " ".join(words + [suffix])
    cand2 = " ".join(words[:-1] + [suffix])

    for c in [cand, cand2]:
        c = normalize_anchor(c)
        if len(c) <= 32 and is_valid_anchor_for_tier(c, tier):
            return c

    trimmed = " ".join((words + [suffix])[:4])
    trimmed = trimmed[:32].strip()
    if is_valid_anchor_for_tier(trimmed, tier):
        return trimmed

    return "Orchid Ledger"


def inject_anchor_ambiguity(
    profile: Dict[str, Any],
    facts_by_field: Dict[str, Dict[str, Any]],
    tier_map: Dict[str, str],
    global_anchors: set,
    used_anchors: set,
    rng: random.Random
) -> List[Tuple[str, str]]:
    if rng.random() > ANCHOR_AMBIGUITY_PROB:
        return []

    candidates = [f for f in FIELDS if f not in AMBIGUITY_EXCLUDE_FIELDS]
    rng.shuffle(candidates)

    n_pairs = rng.randint(1, ANCHOR_AMBIGUITY_MAX_PAIRS)
    pairs = []
    used_fields = set()

    for _ in range(n_pairs):
        avail = [f for f in candidates if f not in used_fields]
        if len(avail) < 2:
            break
        f_a, f_b = avail[0], avail[1]
        used_fields.update([f_a, f_b])

        base_anchor = facts_by_field[f_a]["anchor"]
        tier_b = tier_map[f_b]
        new_anchor = _make_near_duplicate_anchor(base_anchor, tier_b, rng)

        if new_anchor == facts_by_field[f_b]["anchor"]:
            new_anchor = _make_near_duplicate_anchor(base_anchor + " Note", tier_b, rng)
        if new_anchor in global_anchors or new_anchor in used_anchors:
            for _try in range(6):
                new_anchor = _make_near_duplicate_anchor(base_anchor, tier_b, rng)
                if (new_anchor != facts_by_field[f_b]["anchor"] and
                    new_anchor not in global_anchors and new_anchor not in used_anchors):
                    break
            else:
                continue

        facts_by_field[f_b]["anchor"] = new_anchor
        used_anchors.add(new_anchor)
        global_anchors.add(new_anchor)
        pairs.append((f_a, f_b))

    return pairs


# =========================================================
# MAIN PERSON PIPELINE
# =========================================================
def generate_one_person_pipeline(client_gen: genai.Client, client_red: genai.Client,
                                 profile: Dict[str, Any],
                                 global_anchors: set,
                                 used_anchors_global: List[str], used_details_global: List[str],
                                 person_rng: random.Random) -> Dict[str, Any]:
    """Generate 20 facts for one person."""
    tier_map = pick_tiers_for_fields(person_rng)

    used_anchors = set()
    used_details = set()
    facts_by_field: Dict[str, Dict[str, Any]] = {}

    # 1) Negative incident (forget field)
    inc_tier = tier_map[FORGET_FIELD]
    incident_fact = gen_one_fact(
        client_gen, profile, FORGET_FIELD, inc_tier,
        global_anchors, used_anchors, used_details,
        used_anchors_global, used_details_global,
        prompt_builder=lambda: incident_prompt(profile, inc_tier, used_anchors_global, used_details_global),
        require_proper_noun=True
    )
    facts_by_field[FORGET_FIELD] = incident_fact
    shared_entity = extract_shared_entity(incident_fact["detail"]) or "Orchid Council"

    # 2) Dependency field (references incident anchor)
    dep_tier = tier_map[DEPENDENCY_FIELD]
    for repair_attempt in range(6):
        dependency_fact = gen_one_fact(
            client_gen, profile, DEPENDENCY_FIELD, dep_tier,
            global_anchors, used_anchors, used_details,
            used_anchors_global, used_details_global,
            prompt_builder=lambda: dependency_prompt(profile, DEPENDENCY_FIELD, dep_tier,
                                                    incident_fact["anchor"],
                                                    used_anchors_global, used_details_global),
            require_proper_noun=False
        )
        if dependency_fact["detail"].count(incident_fact["anchor"]) == 1:
            break
    else:
        raise RuntimeError("Dependency field failed to reference incident anchor exactly once.")
    facts_by_field[DEPENDENCY_FIELD] = dependency_fact

    # 3) Distractor (reuses shared entity)
    dis_tier = tier_map[DISTRACTOR_FIELD]
    for repair_attempt in range(6):
        distractor_fact = gen_one_fact(
            client_gen, profile, DISTRACTOR_FIELD, dis_tier,
            global_anchors, used_anchors, used_details,
            used_anchors_global, used_details_global,
            prompt_builder=lambda: distractor_prompt(profile, DISTRACTOR_FIELD, dis_tier,
                                                    shared_entity,
                                                    used_anchors_global, used_details_global),
            require_proper_noun=False
        )
        if shared_entity in distractor_fact["detail"]:
            break
    else:
        raise RuntimeError("Distractor failed to reuse shared entity.")
    facts_by_field[DISTRACTOR_FIELD] = distractor_fact

    # 4) Remaining independent fields (BATCH GENERATION!)
    remaining_fields = [f for f in INDEPENDENT_FIELDS if f not in facts_by_field]
    
    if remaining_fields:
        print(f"  [Batch] Generating {len(remaining_fields)} independent facts in 1 API call...", flush=True)
        batch_facts, failed_fields = gen_batch_facts(
            client_gen, profile, remaining_fields, tier_map,
            global_anchors, used_anchors_global, used_details_global
        )
        
        # Add valid batch facts to local tracking
        for fact in batch_facts:
            facts_by_field[fact["field"]] = fact
            anchor_norm = normalize_anchor(fact["anchor"])
            detail_norm = norm_detail(fact["detail"])
            used_anchors.add(anchor_norm)
            used_details.add(detail_norm)
            # Note: global_anchors will be updated by main loop after person completes
        
        print(f"  [Batch] ✓ Generated {len(batch_facts)}/{len(remaining_fields)} facts successfully", flush=True)
        
        # Fallback: Generate failed fields individually
        if failed_fields:
            print(f"  [Fallback] Regenerating {len(failed_fields)} failed fields individually...", flush=True)
            for field in failed_fields:
                tier = tier_map[field]
                try:
                    fact = gen_one_fact(
                        client_gen, profile, field, tier,
                        global_anchors, used_anchors, used_details,
                        used_anchors_global, used_details_global,
                        prompt_builder=lambda f=field, t=tier: field_prompt(profile, f, t, used_anchors_global, used_details_global),
                        require_proper_noun=False
                    )
                    facts_by_field[field] = fact
                except Exception as e:
                    # If individual generation also fails, give up on this person
                    raise RuntimeError(f"Failed to generate field {field} even after fallback: {e}")

    # 5) Inject anchor ambiguity
    ambiguity_pairs = inject_anchor_ambiguity(profile, facts_by_field, tier_map, global_anchors, used_anchors, person_rng)

    # 6) Generate fact_text (BATCH GENERATION!)
    facts = [facts_by_field[f] for f in FIELDS]
    
    print(f"  [Batch] Generating fact_text for all 20 facts in 1 API call...", flush=True)
    batch_fact_texts = gen_batch_fact_texts(client_gen, profile, facts)
    
    if batch_fact_texts and len(batch_fact_texts) == len(facts):
        # Batch succeeded!
        for fact in facts:
            fact["fact_text"] = batch_fact_texts[fact["field"]]
        print(f"  [Batch] ✓ Generated all 20 fact_texts successfully", flush=True)
    else:
        # Fallback to sequential
        print(f"  [Fallback] Batch fact_text failed, generating individually...", flush=True)
        for fact in facts:
            fact["fact_text"] = gen_fact_text(client_gen, profile, fact)

    # 7) Red team review cycles (OPTIONAL: set SKIP_REDTEAM=True to skip for speed)
    SKIP_REDTEAM = False  # Set to True to skip red-team validation (10× faster!)
    
    if SKIP_REDTEAM:
        # Skip red-team, assume pass
        red_report = {"pass": True, "scores": {}, "reasons": ["skipped"], "flags": {}}
    else:
        red_report = red_team_review(client_red, profile, facts, tier_map, ambiguity_pairs)
        for cycle in range(MAX_REDTEAM_CYCLES):
            if red_report.get("pass", False):
                break
            flags = red_report.get("flags", {}) or {}
            if flags.get("has_real_world_leak", False):
                raise RuntimeError("Red team flagged leakage; regenerate person.")

        repaired_any = False

        # Repair dependency
        if flags.get("missing_dependency", False):
            for _ in range(3):
                dep_tier = tier_map[DEPENDENCY_FIELD]
                dependency_fact = gen_one_fact(
                    client_gen, profile, DEPENDENCY_FIELD, dep_tier,
                    global_anchors, used_anchors, used_details,
                    used_anchors_global, used_details_global,
                    prompt_builder=lambda: dependency_prompt(profile, DEPENDENCY_FIELD, dep_tier,
                                                            facts_by_field[FORGET_FIELD]["anchor"],
                                                            used_anchors_global, used_details_global),
                    require_proper_noun=False
                )
                if dependency_fact["detail"].count(facts_by_field[FORGET_FIELD]["anchor"]) == 1:
                    facts_by_field[DEPENDENCY_FIELD] = dependency_fact
                    repaired_any = True
                    break

        # Repair distractor
        if flags.get("missing_false_friend", False):
            shared_entity2 = extract_shared_entity(facts_by_field[FORGET_FIELD]["detail"]) or shared_entity
            for _ in range(3):
                dis_tier = tier_map[DISTRACTOR_FIELD]
                distractor_fact = gen_one_fact(
                    client_gen, profile, DISTRACTOR_FIELD, dis_tier,
                    global_anchors, used_anchors, used_details,
                    used_anchors_global, used_details_global,
                    prompt_builder=lambda: distractor_prompt(profile, DISTRACTOR_FIELD, dis_tier,
                                                            shared_entity2,
                                                            used_anchors_global, used_details_global),
                    require_proper_noun=False
                )
                if shared_entity2 in distractor_fact["detail"]:
                    facts_by_field[DISTRACTOR_FIELD] = distractor_fact
                    repaired_any = True
                    break

        # Repair tier violations / generic facts
        to_repair = set()
        for item in (flags.get("tier_violations") or []):
            f = str(item).split(":")[0].strip()
            if f in FIELDS:
                to_repair.add(f)
        for f in (flags.get("generic_facts") or []):
            if f in FIELDS and f not in [FORGET_FIELD, DEPENDENCY_FIELD, DISTRACTOR_FIELD]:
                to_repair.add(f)

        for f in list(to_repair)[:4]:
            tier = tier_map[f]
            facts_by_field[f] = gen_one_fact(
                client_gen, profile, f, tier,
                global_anchors, used_anchors, used_details,
                used_anchors_global, used_details_global,
                prompt_builder=lambda ff=f, tt=tier: field_prompt(profile, ff, tt, used_anchors_global, used_details_global),
                require_proper_noun=False
            )
            repaired_any = True

        if not repaired_any:
            raise RuntimeError("Red team failed with no safe targeted repairs found.")

        # Re-apply ambiguity
        ambiguity_pairs = inject_anchor_ambiguity(profile, facts_by_field, tier_map, global_anchors, used_anchors, person_rng)

        facts = [facts_by_field[f] for f in FIELDS]
        for fact in facts:
            fact["fact_text"] = gen_fact_text(client_gen, profile, fact)

        red_report = red_team_review(client_red, profile, facts, tier_map, ambiguity_pairs)

    if not red_report.get("pass", False):
        raise RuntimeError("Red team did not pass after repair cycles.")

    person_obj = {
        "person_id": profile["person_id"],
        "full_name": profile["full_name"],
        "age": profile["age"],
        "public_role": profile["public_role"],
        "profile": {
            "one_paragraph_bio": profile["bio"],
            "persona_tags": profile["persona_tags"],
            "home_region_hint": profile["home_region_hint"],
        },
        "facts": facts,
        "anchor_ambiguity_pairs": ambiguity_pairs,
        "tier_map": tier_map,
        "forget_field": FORGET_FIELD,
        "forget_fact_id": FORGET_FACT_ID,
        "dependency_field": DEPENDENCY_FIELD,
        "distractor_field": DISTRACTOR_FIELD,
        "red_team_report": red_report
    }
    return person_obj


# =========================================================
# I/O
# =========================================================
def load_profiles(path: pathlib.Path) -> List[Dict[str, Any]]:
    profiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


def load_existing_people_with_facts(path: pathlib.Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    people = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                people.append(json.loads(line))
    return people


def write_jsonl_append(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# MAIN
# =========================================================
def main():
    start_time = time.time()

    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY env var first.")

    print("="*60)
    print("PART 2: Generate Facts (BATCH MODE ENABLED)")
    print("="*60)
    print(f"🚀 BATCH GENERATION: 6-7× FASTER!")
    print(f"   - 17 independent facts in 1 API call")
    print(f"   - 3 entangled facts sequentially")
    print(f"   - 20 fact_texts in 1 API call")
    print(f"   - Total: ~6-7 API calls per person (was 42-43)")
    print("="*60)
    print(f"Input: {PROFILES_FILE.resolve()}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Rate limit: {RPM_LIMIT} RPM\n")

    # Load profiles from Part 1
    profiles = load_profiles(PROFILES_FILE)
    print(f"Loaded {len(profiles)} profiles from Part 1\n")

    # Resume capability: check existing people with facts
    existing_people = load_existing_people_with_facts(OUTPUT_FILE)
    existing_ids = {p["person_id"] for p in existing_people}
    
    # Find the last successfully generated person_id
    if existing_people:
        last_person_id = max(p["person_id"] for p in existing_people)
        start_person_id = last_person_id + 1
        print(f"[Resume] Found {len(existing_people)} existing people with facts.")
        print(f"         Last successful person: ID {last_person_id}")
        print(f"         Resuming from person ID {start_person_id}...\n")
    else:
        start_person_id = 1
        print(f"[New Run] Starting from person ID 1\n")

    client_gen = genai.Client(api_key=API_KEY)
    client_red = genai.Client(api_key=API_KEY)

    global_anchors = set()
    used_anchors_global: List[str] = []
    used_details_global: List[str] = []

    # Populate global context from existing people
    for person in existing_people:
        for fact in person["facts"]:
            anchor = normalize_anchor(fact["anchor"])
            detail = norm_detail(fact["detail"])
            global_anchors.add(anchor)
            used_anchors_global.append(anchor)
            used_details_global.append(detail)

    # GOAL: Generate TARGET_SUCCESSFUL_PEOPLE (skip difficult ones)
    successful_count = len(existing_people)  # Count existing people
    skipped_people = []
    
    print(f"🎯 GOAL: Generate {TARGET_SUCCESSFUL_PEOPLE} successful people (already have {successful_count})")
    print(f"⚠️  Will SKIP people that are too difficult (after {MAX_PERSON_TRIES} attempts)\n")
    
    # Filter profiles to only those starting from start_person_id
    profiles_to_process = [p for p in profiles if p["person_id"] >= start_person_id]
    print(f"📋 Processing {len(profiles_to_process)} profiles starting from ID {start_person_id}\n")
    
    # Process each person
    for idx, profile in enumerate(profiles_to_process):
        # Stop if we have enough successful people
        if successful_count >= TARGET_SUCCESSFUL_PEOPLE:
            print(f"\n{'='*60}")
            print(f"🎉 SUCCESS! Generated {TARGET_SUCCESSFUL_PEOPLE} people successfully!")
            print(f"{'='*60}")
            if skipped_people:
                print(f"\nSkipped {len(skipped_people)} difficult people:")
                for pid, name in skipped_people:
                    print(f"  - Person {pid}: {name}")
            break

        person_id = profile["person_id"]
        person_rng = random.Random(SEED * 100000 + person_id)

        print(f"\n[Person {person_id}/{len(profiles)}] {profile['full_name']} (Success: {successful_count}/{TARGET_SUCCESSFUL_PEOPLE})", flush=True)
        person_start_time = time.time()

        person_succeeded = False
        for attempt in range(MAX_PERSON_TRIES):
            try:
                person_obj = generate_one_person_pipeline(
                    client_gen, client_red,
                    profile,
                    global_anchors,
                    used_anchors_global, used_details_global,
                    person_rng
                )
                # Save immediately after success
                write_jsonl_append(OUTPUT_FILE, [person_obj])
                person_elapsed = time.time() - person_start_time
                print(f"  ✓ Complete. 20 facts generated in {person_elapsed:.1f}s. Saved to file.")
                
                # Update global context with new facts
                for fact in person_obj["facts"]:
                    anchor = normalize_anchor(fact["anchor"])
                    detail = norm_detail(fact["detail"])
                    global_anchors.add(anchor)
                    used_anchors_global.append(anchor)
                    used_details_global.append(detail)
                    # Keep context size manageable
                    if len(used_anchors_global) > MAX_CONTEXT_HISTORY:
                        used_anchors_global = used_anchors_global[-MAX_CONTEXT_HISTORY:]
                    if len(used_details_global) > MAX_CONTEXT_HISTORY:
                        used_details_global = used_details_global[-MAX_CONTEXT_HISTORY:]
                
                successful_count += 1
                person_succeeded = True
                break
            except Exception as e:
                log_fail(f"[person_fail] pid={person_id} attempt={attempt} err={e}")
                print(f"  ✗ Attempt {attempt+1}/{MAX_PERSON_TRIES} failed: {e}")
                time.sleep(0.2)
        
        if not person_succeeded:
            # Skip this difficult person and continue to next
            print(f"  ⚠️  SKIPPING Person {person_id} (too difficult, tried {MAX_PERSON_TRIES} times)")
            print(f"      Continuing to next person...\n")
            skipped_people.append((person_id, profile["full_name"]))
            log_fail(f"[person_skipped] pid={person_id} name={profile['full_name']} reason=too_many_failures")
            continue

        # Progress update every 5 successful people
        if successful_count > 0 and successful_count % 5 == 0 and person_succeeded:
            elapsed = time.time() - start_time
            rate = successful_count / elapsed * 60 if elapsed > 0 else 0
            print(f"\n[Progress] {successful_count}/{TARGET_SUCCESSFUL_PEOPLE} successful | Time: {elapsed/60:.1f} min | Rate: {rate:.1f} people/min\n")

    # Final summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("PART 2 COMPLETE")
    print("="*60)
    print(f"✓ Successful people: {successful_count}")
    if skipped_people:
        print(f"⚠ Skipped people: {len(skipped_people)}")
    print(f"Output file: {OUTPUT_FILE.resolve()}")
    print(f"Time: {elapsed/60:.1f} minutes")
    if successful_count > 0:
        print(f"Avg time per person: {elapsed/successful_count:.1f} seconds")
    
    if skipped_people:
        print(f"\n⚠️  Skipped {len(skipped_people)} difficult people (see {FAIL_LOG.resolve()} for details)")
    
    print(f"\n✅ Next step: Run part3_generate_qa.py (or inspect results with inspect_benchmark.py)")


if __name__ == "__main__":
    main()
