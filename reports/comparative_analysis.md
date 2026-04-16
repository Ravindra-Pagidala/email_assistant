## Section 3 — Model Comparison and Analysis

### Context: How I Measured Quality

Before comparing the two models, here's what each metric actually measures :

**Metric 1 — Fact Integration Score (Python):** Did the email include all the key facts we gave it? Simple keyword check. Score of 1.0 means every fact made it into the email. Score of 0.6 means 3 out of 5 facts appeared. No LLM involved , just code.

**Metric 2 — Tone Consistency Score (LLM-as-Judge):** Did the email maintain the requested tone (formal/casual/urgent/empathetic) **from the first line to the last?** We split each email into 3 parts : opening, body, closing and scored each part separately 1 to 10. If an email starts formal but ends casual, it gets penalized. This catches a very common LLM failure that a single overall score would miss entirely.

**Metric 3 — Actionability Score (LLM-as-Judge):** Did the email actually tell the recipient what to do next? There are 3 yes/no checks:
 a ) Is there a clear ask ?
 b ) Is there a defined next step?
 c ) Is there a timeframe? 

An email that ends with "please let me know if you have questions" scores 0.33. An email that ends with "can we schedule a 30-minute call this Thursday?" scores 1.0.

### What Are The Models I’ve Used ?

 Gemini (Gemma 3 27B) & Groq (LLaMA 4 Scout)

### 1. Which Model Performed Better?

**Gemma 3 27B won across all 3 metrics.**

| Metric | Gemma 3 27B | LLaMA 4 Scout | Gap |
| --- | --- | --- | --- |
| Fact Integration | 1.00 | 1.00 | 0.00 |
| Tone Consistency | 0.77 | 0.66 | +0.11 |
| Actionability | 0.83 | 0.77 | +0.06 |
| **Overall** | **0.87** | **0.81** | **+0.06** |

Both models performed well on Fact Integration. Every key fact appeared in every email across all 10 scenarios. That part was easy for both. The separation happened on Tone Consistency and Actionability.

The 0.11 difference in tone is not merely superficial. It means LLaMA's emails were noticeably shifting in tone somewhere between the opening and the closing in over a third of the test scenarios. Gemma held its tone more consistently throughout the full email.

---

### 2. What Was the Biggest Failure Mode of LLaMA 4 Scout?

**Tone drift on casual and urgent emails , backed by the numbers.**

| Tone Type | Gemma avg | LLaMA avg | Verdict |
| --- | --- | --- | --- |
| Formal | 0.958 | 0.900 | Both solid |
| Casual | 0.850 | 0.467 | LLaMA fails |
| Empathetic | 0.617 | 0.583 | Both struggle |
| Urgent | 0.467 | 0.467 | Both fail equally |

**Casual tone is where LLaMA breaks down completely.** 0.467 vs Gemma's 0.850 — that's not a marginal difference, that's a completely different output quality. In Scenario 7 (inviting a colleague to collaborate — casual) and Scenario 9 (welcoming a new employee — casual), LLaMA's emails started warm and conversational but drifted back into stiff corporate language by the closing paragraph. The judge consistently caught this drift in the closing section specifically. Gemma maintained its casual register all the way through.

**Urgent tone failed equally for both models** — and this is worth being honest about. Both scored 0.467 on urgent scenarios. The reason is structural: both models kept opening urgent emails with soft preamble like "I hope this message finds you well" before getting to the urgent content. That's a shared weakness in the prompt design and the models' tendency to default to polite openers regardless of tone instruction. Neither model fully solved this.

**Actionability failures were concentrated in empathetic scenarios for LLaMA.** Scenarios 1, 3, 4, 5, and 9 all scored below 0.67 on actionability. The pattern was consistent — LLaMA was generating empathetic, well-worded emails that fully acknowledged the situation but forgot to close with a concrete next step. The emails ended warmly but vaguely. "Please don't hesitate to reach out" is not a next step. Gemma had the same problem in scenarios 1, 3, and 9 but recovered better in the others.

---

### 3. Which Model Do You Recommend for Production and Why?

**The answer depends on what the production system actually does.** 

**For quality-first systems — Gemma 3 27B.**

Consider enterprise productivity systems that assist users in drafting client-facing proposals or professional business emails where output quality is critical. In these systems, a human reads the email before it goes out but they're trusting the AI to get it mostly right. A tone drift from formal to casual in a client proposal is embarrassing. An email that doesn't clearly state what should happen next wastes everyone's time.

Gemma's 0.87 overall vs LLaMA's 0.81 reflects exactly this. The 6-point gap comes entirely from scenarios where the email tone or structure subtly broke down. Over thousands of emails per day at enterprise scale, those breakdowns compound into real brand and relationship damage. Gemma is the safer choice when email quality is non-negotiable.

**For speed-first and high-volume systems — LLaMA 4 Scout.**

Consider high-volume automated outreach systems that generate and send thousands of emails per hour with minimal or no human review. LLaMA processes an email in 855ms. Gemma takes 6,280ms. **That's 7.3x faster**. At the scale HubSpot operates **millions of emails per month.** That speed difference is the difference between running the system on 1 server vs 7 servers. LLaMA's 0.81 accuracy is not broken. For bulk formal outreach specifically, LLaMA's formal tone score of 0.90 is strong and reliable.

---

**Final recommendation for this system:**

Since this is a general-purpose email assistant that needs to handle everything from casual messages to urgent and formal communication, maintaining the right tone consistently is very important.

While LLaMA 4 Scout is much faster, Gemma 3 27B delivered better overall quality across all evaluation metrics, especially in tone consistency and actionability.

**For this reason, I would recommend Gemma 3 27B as the production model, as output quality matters more here than speed.**