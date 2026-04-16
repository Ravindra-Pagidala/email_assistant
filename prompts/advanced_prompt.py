"""
Advanced Prompt Engineering Module.

Technique used: HYBRID (Role-Playing + Few-Shot Examples + Chain-of-Thought)

Why hybrid?
  - Role-Playing alone sets expertise but lacks structure
  - Few-Shot alone shows format but lacks reasoning
  - Chain-of-Thought alone reasons but lacks style anchoring
  Combined → the model has persona, examples, AND a reasoning framework.
  This is how Microsoft Copilot and Salesforce Einstein prompt their email models.

This module is the DTO/Builder layer — it constructs the final prompt
from dynamic inputs (intent, key_facts, tone) before sending to any LLM.
"""

from typing import List
from config.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# These are hardcoded "gold standard" email examples.
# The LLM learns expected quality and format from these.
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "intent": "Follow up after a client meeting",
        "key_facts": [
            "Meeting held on Monday April 7th",
            "Discussed Q3 budget reallocation",
            "Client requested a revised proposal by Friday",
            "Primary contact is James Whitfield, VP of Finance"
        ],
        "tone": "formal",
        "email": """Subject: Follow-Up: Q3 Budget Discussion & Revised Proposal

Dear James,

Thank you for taking the time to meet with us on Monday, April 7th. It was a productive conversation, and I appreciate the clarity you provided regarding the Q3 budget reallocation priorities.

As discussed, I will have a revised proposal ready for your review by this Friday. The document will reflect the updated budget parameters we outlined and will include three alternative allocation scenarios for your consideration.

Please do not hesitate to reach out should you have any questions or require additional information in the interim.

Best regards,
[Your Name]"""
    },
    {
        "intent": "Request a project status update from a team member",
        "key_facts": [
            "Project deadline is April 30th",
            "Last status update was received two weeks ago",
            "Milestone 3 is currently at risk",
            "Stakeholder review is scheduled for next Tuesday"
        ],
        "tone": "urgent",
        "email": """Subject: URGENT: Project Status Update Required — Deadline April 30th

Hi [Team Member],

I need an immediate update on the project status ahead of our stakeholder review this coming Tuesday.

Given that our last update was two weeks ago and we have Milestone 3 currently at risk, it is critical that I have full visibility before Tuesday's meeting. Our April 30th deadline leaves very little room for course correction.

Please send me a status report by end of day today covering: current completion percentage, blockers on Milestone 3, and your confidence level on the April 30th deadline.

This is time-sensitive — please prioritize accordingly.

[Your Name]"""
    },
    {
        "intent": "Apologize for a service outage and reassure a customer",
        "key_facts": [
            "Outage lasted 3 hours on April 10th between 2 PM and 5 PM",
            "Root cause was a database failover misconfiguration",
            "Issue has been permanently resolved",
            "Customer will receive a 20% credit on next invoice"
        ],
        "tone": "empathetic",
        "email": """Subject: Our Sincere Apologies — Service Outage on April 10th

Dear [Customer Name],

I want to personally reach out and sincerely apologize for the service disruption you experienced on April 10th between 2:00 PM and 5:00 PM.

I completely understand how frustrating a three-hour outage can be, especially during business hours. The issue was traced to a database failover misconfiguration — a gap in our infrastructure that we have since permanently resolved with additional automated safeguards.

As a token of our commitment to you, we will be applying a 20% credit to your next invoice. More importantly, we have put in place the monitoring improvements necessary to ensure this does not happen again.

Thank you for your patience and continued trust in us.

Warm regards,
[Your Name]"""
    }
]


def build_prompt(intent: str, key_facts: List[str], tone: str) -> str:
    """
    Construct the final prompt using the hybrid advanced prompting technique.

    This function is the core of the prompt engineering layer.
    It builds a prompt that combines:
      1. Role-Playing — sets the LLM's persona as an expert
      2. Few-Shot Examples — anchors quality expectations
      3. Chain-of-Thought — forces structured reasoning before output

    Args:
        intent: The purpose of the email (e.g., "Follow up after meeting")
        key_facts: List of facts that MUST appear in the email
        tone: Desired tone (formal, casual, urgent, empathetic)

    Returns:
        str: Complete prompt string ready to send to any LLM

    Raises:
        ValueError: If any required input is missing or empty
    """
    _validate_inputs(intent, key_facts, tone)

    logger.debug(
        "Building advanced prompt",
        extra={
            "extra_fields": {
                "intent": intent,
                "tone": tone,
                "key_facts_count": len(key_facts),
            }
        }
    )

    facts_formatted = "\n".join(f"  - {fact}" for fact in key_facts)
    examples_text = _format_few_shot_examples()

    prompt = f"""You are a Senior Communications Specialist with 15 years of experience \
writing high-impact professional emails for Fortune 500 companies including Microsoft, \
Salesforce, and HubSpot. You have a deep understanding of business communication, \
stakeholder psychology, and tone calibration across corporate contexts.

Your emails are known for three qualities:
1. Every key fact is woven naturally into the narrative — never listed awkwardly
2. Tone is consistent from the subject line through to the sign-off
3. Every email ends with a clear, specific call-to-action

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF PERFECT EMAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{examples_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing, reason through the following steps internally:

STEP 1 — AUDIENCE ANALYSIS: Who is likely receiving this email? What is their role and priority?
STEP 2 — TONE CALIBRATION: What specific language patterns define a '{tone}' tone? How should the opening, body, and closing each reflect this tone consistently?
STEP 3 — FACT INTEGRATION PLAN: How can each key fact be woven naturally into the email narrative rather than listed robotically?
STEP 4 — CALL-TO-ACTION DESIGN: What is the ONE specific action the recipient should take? Make it concrete with a timeframe if possible.
STEP 5 — WRITE THE EMAIL: Apply your reasoning from Steps 1-4 to produce the final email.

Now generate a professional email with the following inputs:

INTENT: {intent}

KEY FACTS TO INCLUDE:
{facts_formatted}

TONE: {tone}

Output ONLY the final email (Subject line + Body). Do not include your reasoning steps, commentary, or any text outside the email itself.
"""

    logger.debug(f"Prompt built successfully — {len(prompt)} characters")
    return prompt


def _format_few_shot_examples() -> str:
    """Format the few-shot examples into readable prompt text."""
    formatted = []
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        facts = "\n".join(f"  - {f}" for f in example["key_facts"])
        formatted.append(
            f"EXAMPLE {i}:\n"
            f"Intent: {example['intent']}\n"
            f"Key Facts:\n{facts}\n"
            f"Tone: {example['tone']}\n"
            f"Output:\n{example['email']}\n"
        )
    return "\n---\n".join(formatted)


def _validate_inputs(intent: str, key_facts: List[str], tone: str) -> None:
    """
    Validate all inputs before building the prompt.
    Fails fast with clear error messages — never send a broken prompt to the API.

    Raises:
        ValueError: With specific message about which field is invalid
    """
    if not intent or not intent.strip():
        raise ValueError("'intent' cannot be empty. Provide a clear email purpose.")

    if not key_facts or len(key_facts) == 0:
        raise ValueError("'key_facts' cannot be empty. Provide at least one fact.")

    if len(key_facts) > 10:
        raise ValueError(
            f"'key_facts' has {len(key_facts)} items — maximum is 10. "
            "Too many facts produce incoherent emails."
        )

    valid_tones = {"formal", "casual", "urgent", "empathetic"}
    tone_lower = tone.lower().strip()
    if tone_lower not in valid_tones:
        raise ValueError(
            f"Invalid tone '{tone}'. Must be one of: {', '.join(sorted(valid_tones))}"
        )

    for i, fact in enumerate(key_facts):
        if not fact or not fact.strip():
            raise ValueError(f"key_facts[{i}] is empty. All facts must have content.")