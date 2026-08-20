"""
Generates a policy-grounded natural-language explanation using Groq's
LLM API, given a decision's context and retrieved policy chunks.

Falls back to a deterministic, rule-based explanation if the LLM call
fails for any reason — the explanation layer must never block or
alter the underlying credit decision.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"


def _build_prompt(decision: str, risk_score: float, confidence: float,
                   integrity_status: str, policy_chunks: list[dict]) -> str:
    policy_text = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in policy_chunks
    )

    return f"""You are explaining a credit decision to a bank underwriter.
Be factual, concise (3-4 sentences), and ground your explanation only
in the policy text provided below. Do not invent numbers or policy
rules not present in the text. Do not suggest a different decision
than the one given — your role is only to explain it.

Decision: {decision}
Risk score: {round(risk_score * 100)}%
Evidence Confidence: {round(confidence)}%
Integrity status: {integrity_status}

Relevant policy text:
{policy_text}

Write the explanation now:"""


def generate_explanation(
    decision: str,
    risk_score: float,
    confidence: float,
    integrity_status: str,
    policy_chunks: list[dict],
) -> dict:
    """
    Returns a dict with either a genuine LLM-generated explanation,
    or a deterministic fallback if the call fails.
    """

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return _fallback(decision, policy_chunks, reason="No GROQ_API_KEY configured")

    try:
        client = Groq(api_key=api_key)
        prompt = _build_prompt(decision, risk_score, confidence, integrity_status, policy_chunks)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        text = response.choices[0].message.content.strip()

        return {
            "explanation": text,
            "source": "llm",
            "model": MODEL,
        }

    except Exception as exc:
        return _fallback(decision, policy_chunks, reason=str(exc))


def _fallback(decision: str, policy_chunks: list[dict], reason: str) -> dict:
    if policy_chunks:
        grounded_text = policy_chunks[0]["text"].split("\n")[-1]
    else:
        grounded_text = "No policy context was retrieved for this decision."

    return {
        "explanation": (
            f"This applicant received a {decision} decision. {grounded_text}"
        ),
        "source": "fallback",
        "model": None,
        "fallbackReason": reason,
    }