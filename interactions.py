"""
MediScanAI - Drug Interaction Checker
Checks every pair of drugs from the prescription for interactions.
Uses an agent loop: RAG lookup → LLM reasoning → structured result.
"""

import json
import itertools
from dataclasses import dataclass
from pipeline import get_client, rag_lookup


@dataclass
class Interaction:
    drug_a: str
    drug_b: str
    severity: str          # "high" | "moderate" | "low" | "none"
    description: str
    recommendation: str
    sources: list[str]     # which KB docs informed this


def check_pair(drug_a: str, drug_b: str) -> Interaction:
    """
    Agent loop for one drug pair:
    1. RAG lookup for both drugs
    2. LLM reasons over retrieved context
    3. Returns structured Interaction
    """
    client = get_client()

    # Step 1: RAG — retrieve knowledge for each drug and the pair together
    context_a   = rag_lookup(f"{drug_a} interactions warnings", n_results=3)
    context_b   = rag_lookup(f"{drug_b} interactions warnings", n_results=3)
    context_pair = rag_lookup(f"{drug_a} {drug_b} interaction", n_results=2)

    all_context = "\n\n---\n\n".join(filter(None, [context_a, context_b, context_pair]))

    # Collect source drug names mentioned in retrieved docs
    sources = []
    for ctx in [context_a, context_b, context_pair]:
        if ctx:
            for line in ctx.split("\n"):
                if line.startswith("Drug:"):
                    src = line.replace("Drug:", "").strip()
                    if src and src not in sources:
                        sources.append(src)

    # Step 2: LLM reasoning
    prompt = f"""You are a clinical pharmacist checking for drug interactions.

Drug A: {drug_a}
Drug B: {drug_b}

Relevant knowledge base context:
{all_context if all_context else "No specific interaction data found in knowledge base."}

Analyze whether these two drugs interact. Return ONLY a JSON object (no markdown, no explanation):
{{
  "severity": "high" | "moderate" | "low" | "none",
  "description": "1-2 sentence description of the interaction or confirmation of no known interaction",
  "recommendation": "specific actionable advice for the patient"
}}

Severity guide:
- high: avoid combination or requires immediate medical supervision
- moderate: use with caution, may need dose adjustment or monitoring
- low: minor interaction, generally manageable
- none: no clinically significant interaction known"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text.strip())
        severity = data.get("severity", "low")
        if severity not in ("high", "moderate", "low", "none"):
            severity = "low"
        return Interaction(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=severity,
            description=data.get("description", "Unable to determine interaction."),
            recommendation=data.get("recommendation", "Consult your pharmacist."),
            sources=sources[:3],
        )
    except Exception:
        return Interaction(
            drug_a=drug_a,
            drug_b=drug_b,
            severity="low",
            description="Could not parse interaction data.",
            recommendation="Consult your pharmacist or doctor.",
            sources=sources[:3],
        )


def check_all_interactions(drug_names: list[str]) -> list[Interaction]:
    """
    Check every pair of drugs in the list.
    Returns interactions sorted by severity (high first).
    """
    if not drug_names or len(drug_names) < 2:
        return []

    # Deduplicate
    drugs = list(dict.fromkeys(d.strip() for d in drug_names if d.strip()))
    pairs = list(itertools.combinations(drugs, 2))

    results = []
    for drug_a, drug_b in pairs:
        interaction = check_pair(drug_a, drug_b)
        results.append(interaction)

    # Sort: high → moderate → low → none
    severity_order = {"high": 0, "moderate": 1, "low": 2, "none": 3}
    results.sort(key=lambda x: severity_order.get(x.severity, 3))

    return results


def severity_color(severity: str) -> str:
    return {
        "high":     "#991b1b",
        "moderate": "#92400e",
        "low":      "#1e40af",
        "none":     "#166534",
    }.get(severity, "#374151")


def severity_bg(severity: str) -> str:
    return {
        "high":     "#fee2e2",
        "moderate": "#fef3c7",
        "low":      "#dbeafe",
        "none":     "#dcfce7",
    }.get(severity, "#f3f4f6")


def severity_label(severity: str) -> str:
    return {
        "high":     "⛔ High Risk",
        "moderate": "⚠️ Moderate",
        "low":      "ℹ️ Low Risk",
        "none":     "✅ No Interaction",
    }.get(severity, severity.title())
