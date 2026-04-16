"""
Custom Evaluation Metrics — 3 Production-Grade Metrics.

Metric 1: Key Fact Integration Score  → Pure Python (no LLM needed)
Metric 2: Tone Consistency Score      → LLM-as-Judge
Metric 3: Actionability Score         → LLM-as-Judge
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from config.logger import get_logger, ContextLogger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Structured result for a single metric evaluation."""
    metric_name: str
    score: float
    breakdown: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.score >= 0.7

    @classmethod
    def failure(cls, metric_name: str, error: str) -> "MetricResult":
        return cls(metric_name=metric_name, score=0.0,
                   reasoning="Metric evaluation failed", error=error)


# ─────────────────────────────────────────────────────────
# METRIC 1: KEY FACT INTEGRATION SCORE
# Pure Python — no LLM needed
# ─────────────────────────────────────────────────────────

def compute_fact_integration_score(
    generated_email: str,
    key_facts: List[str],
    correlation_id: Optional[str] = None,
) -> MetricResult:
    """
    Metric 1: Key Fact Integration Score (Pure Python)

    Definition:
      Measures what percentage of provided key facts appear
      in the generated email via keyword matching.

    Logic:
      For each fact → extract keywords → check if any keyword
      appears in email → score = facts_found / total_facts

    Scale: 0.0 (no facts) to 1.0 (all facts present)
    """
    ctx_logger = ContextLogger(logger, correlation_id)
    metric_name = "fact_integration_score"

    if not generated_email or not generated_email.strip():
        return MetricResult.failure(metric_name, "Generated email is empty")
    if not key_facts:
        return MetricResult.failure(metric_name, "No key facts provided")

    email_lower = generated_email.lower()

    stopwords = {
        "a","an","the","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","on","at",
        "by","for","with","about","and","or","but","if","then","that",
        "this","it","its","we","our","your","my","their","i","you",
        "he","she","they","them","us","me","him","her","not","no","so",
        "as","from","into","through","during","before","after","between",
        "each","further","than","too","very","just",
    }

    fact_results = {}
    facts_found = 0

    for i, fact in enumerate(key_facts):
        if not fact or not fact.strip():
            fact_results[f"fact_{i+1}"] = {"fact": fact, "found": False}
            continue

        words = re.findall(r'\b[a-zA-Z0-9\$\%\.]+\b', fact.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        if not keywords:
            keywords = [fact.lower().strip()]

        found = False
        matched_keyword = None

        for keyword in keywords:
            if keyword in email_lower:
                found = True
                matched_keyword = keyword
                break

        if found:
            facts_found += 1

        fact_results[f"fact_{i+1}"] = {
            "fact": fact,
            "found": found,
            "matched_keyword": matched_keyword,
        }

    score = round(facts_found / len(key_facts), 4)
    reasoning = (
        f"Found {facts_found}/{len(key_facts)} key facts. "
        f"Score = {score:.2f}"
    )

    ctx_logger.info(
        f"{metric_name}: {score:.2f} ({facts_found}/{len(key_facts)} facts)",
        extra={"extra_fields": {"metric": metric_name, "score": score}}
    )

    return MetricResult(
        metric_name=metric_name,
        score=score,
        breakdown={"facts_found": facts_found,
                   "total_facts": len(key_facts),
                   "per_fact": fact_results},
        reasoning=reasoning,
    )


# ─────────────────────────────────────────────────────────
# METRIC 2: TONE CONSISTENCY SCORE
# LLM-as-Judge
# ─────────────────────────────────────────────────────────

def compute_tone_consistency_score(
    generated_email: str,
    expected_tone: str,
    judge_fn: Callable[[str], str],
    correlation_id: Optional[str] = None,
) -> MetricResult:
    """
    Metric 2: Tone Consistency Score (LLM-as-Judge)

    Definition:
      Measures whether tone is consistent across opening,
      body, and closing. Catches tone drift — e.g. starts
      formal but ends casual.

    Logic:
      Split email into 3 sections → judge scores each 1-10
      → average / 10 → apply drift penalty if any < 5

    Scale: 0.0 to 1.0
    """
    ctx_logger = ContextLogger(logger, correlation_id)
    metric_name = "tone_consistency_score"

    if not generated_email or not generated_email.strip():
        return MetricResult.failure(metric_name, "Generated email is empty")

    valid_tones = {"formal", "casual", "urgent", "empathetic"}
    tone_normalized = expected_tone.lower().strip()
    if tone_normalized not in valid_tones:
        return MetricResult.failure(
            metric_name, f"Invalid tone '{expected_tone}'"
        )

    sections = _split_email_into_sections(generated_email)
    section_scores = {}

    for section_name, section_text in sections.items():
        if not section_text.strip():
            section_scores[section_name] = 5.0
            continue

        judge_prompt = f"""You are an expert linguistic analyst.

Evaluate this email section for tone match with "{tone_normalized.upper()}".

Tone definitions:
- FORMAL: Professional, no contractions, complete sentences, titles used
- CASUAL: Friendly, conversational, contractions ok, first names used
- URGENT: Direct, short sentences, deadlines stated, imperative language
- EMPATHETIC: Warm, acknowledges feelings, apologetic, supportive

Email Section ({section_name}):
\"\"\"{section_text}\"\"\"

Rate tone match 1-10 (1=wrong, 5=partial, 10=perfect).
Respond with ONLY a single integer. No explanation."""

        try:
            raw = judge_fn(judge_prompt)
            section_scores[section_name] = _parse_numeric_score(raw, 1, 10)
        except Exception as e:
            ctx_logger.warning(f"Judge failed for {section_name}: {e}")
            section_scores[section_name] = 5.0

    raw_scores = list(section_scores.values())
    avg = sum(raw_scores) / len(raw_scores)
    drift_penalty = 0.1 if min(raw_scores) < 5 else 0.0
    final_score = round(max(0.0, (avg / 10.0) - drift_penalty), 4)

    reasoning = (
        f"Opening:{section_scores.get('opening','N/A')}/10 "
        f"Body:{section_scores.get('body','N/A')}/10 "
        f"Closing:{section_scores.get('closing','N/A')}/10 "
        f"avg={avg:.1f} penalty={drift_penalty} final={final_score:.2f}"
    )

    ctx_logger.info(
        f"{metric_name}: {final_score:.2f}",
        extra={"extra_fields": {"metric": metric_name, "score": final_score}}
    )

    return MetricResult(
        metric_name=metric_name,
        score=final_score,
        breakdown={"section_scores": section_scores,
                   "average_raw": round(avg, 2),
                   "drift_penalty": drift_penalty,
                   "expected_tone": tone_normalized},
        reasoning=reasoning,
    )


# ─────────────────────────────────────────────────────────
# METRIC 3: ACTIONABILITY SCORE
# LLM-as-Judge
# ─────────────────────────────────────────────────────────

def compute_actionability_score(
    generated_email: str,
    intent: str,
    judge_fn: Callable[[str], str],
    correlation_id: Optional[str] = None,
) -> MetricResult:
    """
    Metric 3: Actionability Score (LLM-as-Judge)

    Definition:
      Measures whether the email drives recipient to a
      specific concrete action.

    Logic:
      Judge checks 3 components (YES/NO each):
        1. Clear Ask — one specific request
        2. Next Step — what happens next
        3. Timeframe — deadline or time reference
      Score = components_present / 3

    Scale: 0.0 to 1.0
    """
    ctx_logger = ContextLogger(logger, correlation_id)
    metric_name = "actionability_score"

    if not generated_email or not generated_email.strip():
        return MetricResult.failure(metric_name, "Generated email is empty")
    if not intent or not intent.strip():
        return MetricResult.failure(metric_name, "Intent not provided")

    judge_prompt = f"""You are a business communication analyst.

Email intent: "{intent}"

Email:
\"\"\"{generated_email}\"\"\"

Answer ONLY YES or NO for each:

COMPONENT_1_CLEAR_ASK: Does the email have one specific, clear request?
COMPONENT_2_NEXT_STEP: Does the email define what should happen next?
COMPONENT_3_TIMEFRAME: Does the email include a specific deadline or timeframe?

Respond in EXACTLY this format:
COMPONENT_1_CLEAR_ASK: YES
COMPONENT_2_NEXT_STEP: NO
COMPONENT_3_TIMEFRAME: YES"""

    try:
        raw = judge_fn(judge_prompt)
        components = _parse_actionability_response(raw)
        present = sum(1 for v in components.values() if v)
        score = round(present / 3.0, 4)

        reasoning = (
            f"ClearAsk:{'YES' if components['clear_ask'] else 'NO'} "
            f"NextStep:{'YES' if components['next_step'] else 'NO'} "
            f"Timeframe:{'YES' if components['timeframe'] else 'NO'} "
            f"= {present}/3 = {score:.2f}"
        )

        ctx_logger.info(
            f"{metric_name}: {score:.2f} ({present}/3)",
            extra={"extra_fields": {"metric": metric_name, "score": score}}
        )

        return MetricResult(
            metric_name=metric_name,
            score=score,
            breakdown={
                "clear_ask": components["clear_ask"],
                "next_step": components["next_step"],
                "timeframe": components["timeframe"],
                "components_present": present,
            },
            reasoning=reasoning,
        )

    except Exception as e:
        ctx_logger.error(f"Actionability judge failed: {e}")
        return MetricResult.failure(metric_name, f"Judge failed: {str(e)[:200]}")


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _split_email_into_sections(email_text: str) -> Dict[str, str]:
    """Split email into opening / body / closing."""
    text = email_text.strip().replace('\r\n', '\n')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if len(paragraphs) == 0:
        return {"opening": text, "body": "", "closing": ""}
    if len(paragraphs) == 1:
        return {"opening": paragraphs[0], "body": "", "closing": ""}
    if len(paragraphs) == 2:
        return {"opening": paragraphs[0], "body": "", "closing": paragraphs[1]}
    return {
        "opening": paragraphs[0],
        "body": "\n\n".join(paragraphs[1:-1]),
        "closing": paragraphs[-1],
    }


def _parse_numeric_score(raw: str, min_val: int, max_val: int) -> float:
    """Extract first number from LLM response and clamp to range."""
    if not raw:
        raise ValueError("Empty judge response")
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', raw.strip())
    if not numbers:
        raise ValueError(f"No number in response: '{raw[:80]}'")
    return max(float(min_val), min(float(max_val), float(numbers[0])))


def _parse_actionability_response(raw: str) -> Dict[str, bool]:
    """Parse YES/NO structured response from judge."""
    if not raw:
        return {"clear_ask": False, "next_step": False, "timeframe": False}

    up = raw.upper()

    def get(key: str) -> bool:
        m = re.search(rf'{key}[:\s]+(YES|NO)', up)
        if m:
            return m.group(1) == "YES"
        pos = up.find(key)
        if pos != -1:
            return "YES" in up[pos:pos+50]
        return False

    return {
        "clear_ask": get("COMPONENT_1_CLEAR_ASK"),
        "next_step": get("COMPONENT_2_NEXT_STEP"),
        "timeframe": get("COMPONENT_3_TIMEFRAME"),
    }