"""
MediScanAI - Evaluation Script
Tests OCR accuracy and RAG retrieval quality against known test cases.

Run: python eval.py
Run verbose: python eval.py --verbose
Run specific test: python eval.py --test ocr
"""

import argparse
import time
import json
from pipeline import get_client, rag_lookup_with_citations, get_kb_size, extract_drug_info

# ── Test cases ─────────────────────────────────────────────────────────────────

# RAG retrieval test cases: (query, expected_drugs_in_results)
RAG_TESTS = [
    ("What are the side effects of Amoxicillin?",        ["Amoxicillin"]),
    ("Lisinopril warnings blood pressure",               ["Lisinopril"]),
    ("metformin diabetes dosage instructions",           ["Metformin"]),
    ("sertraline depression SSRI side effects",          ["Sertraline"]),
    ("ibuprofen NSAID pain interactions",                ["Ibuprofen"]),
    ("atorvastatin statin cholesterol warnings",         ["Atorvastatin"]),
    ("alprazolam benzodiazepine anxiety",                ["Alprazolam"]),
    ("omeprazole proton pump inhibitor GERD",            ["Omeprazole"]),
    ("prednisone corticosteroid inflammation",           ["Prednisone"]),
    ("gabapentin neuropathic pain epilepsy",             ["Gabapentin"]),
]

# Drug extraction test cases: (ocr_text, expected_fields)
EXTRACTION_TESTS = [
    (
        "Patient: Jane Doe\nRx: Amoxicillin 500mg\nSig: Take 1 capsule 3x daily for 10 days\nRefills: 0\nDr. Smith",
        {"drug_names": ["Amoxicillin"], "refills": True, "doctor_name": True}
    ),
    (
        "Lisinopril 10mg tablets\nTake 1 tablet daily\nWarning: Monitor blood pressure\nPatient: John Doe\nRefills: 3",
        {"drug_names": ["Lisinopril"], "refills": True, "warnings": True}
    ),
    (
        "Metformin HCl 500mg\nTake 1 tablet twice daily with meals\nFor Type 2 Diabetes\nDr. Johnson\nRefills: 5",
        {"drug_names": ["Metformin"], "instructions": True, "refills": True}
    ),
]

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def print_header(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def print_result(passed: bool, label: str, detail: str = ""):
    icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
    print(f"  {icon}  {label}")
    if detail:
        print(f"       {YELLOW}{detail}{RESET}")


# ── Test 1: Knowledge Base ─────────────────────────────────────────────────────
def test_knowledge_base(verbose: bool = False) -> dict:
    print_header("Test 1: Knowledge Base")
    kb_size = get_kb_size()
    print(f"  Knowledge base size: {BOLD}{kb_size} drugs{RESET}")

    passed = kb_size > 0
    print_result(passed, f"KB loaded ({kb_size} drugs)", "" if passed else "Run: python build_knowledge_base.py --skip-scrape")

    size_score = 1.0 if kb_size > 500 else 0.7 if kb_size > 100 else 0.3 if kb_size > 0 else 0.0
    print(f"  Coverage score: {BOLD}{size_score:.0%}{RESET} {'🟢' if size_score > 0.7 else '🟡' if size_score > 0.3 else '🔴'}")
    return {"passed": passed, "kb_size": kb_size, "coverage_score": size_score}


# ── Test 2: RAG Retrieval ──────────────────────────────────────────────────────
def test_rag_retrieval(verbose: bool = False) -> dict:
    print_header("Test 2: RAG Retrieval Quality")

    if get_kb_size() == 0:
        print(f"  {YELLOW}⚠️  KB empty — skipping RAG tests{RESET}")
        return {"skipped": True, "accuracy": 0.0}

    passed_count = 0
    total_confidence = 0.0

    for query, expected_drugs in RAG_TESTS:
        context, citations, confidence = rag_lookup_with_citations(query, n_results=5)
        retrieved_drugs = [c["drug"].lower() for c in citations]
        hit = any(
            any(exp.lower() in ret or ret in exp.lower() for ret in retrieved_drugs)
            for exp in expected_drugs
        )
        if hit:
            passed_count += 1
        total_confidence += confidence

        label = f"{query[:55]}..."
        detail = ""
        if verbose:
            detail = f"Expected: {expected_drugs} | Got: {[c['drug'] for c in citations[:3]]} | conf={confidence:.2f}"
        print_result(hit, label, detail)

    accuracy = passed_count / len(RAG_TESTS)
    avg_conf = total_confidence / len(RAG_TESTS)
    print(f"\n  Retrieval accuracy: {BOLD}{accuracy:.0%}{RESET} ({passed_count}/{len(RAG_TESTS)})")
    print(f"  Avg confidence:     {BOLD}{avg_conf:.2f}{RESET}")
    return {"accuracy": accuracy, "avg_confidence": avg_conf, "passed": passed_count, "total": len(RAG_TESTS)}


# ── Test 3: Drug Extraction ────────────────────────────────────────────────────
def test_drug_extraction(verbose: bool = False) -> dict:
    print_header("Test 3: Drug Extraction (LLM)")

    passed_count = 0
    latencies = []

    for ocr_text, expected in EXTRACTION_TESTS:
        start = time.time()
        result = extract_drug_info(ocr_text)
        latency = round(time.time() - start, 2)
        latencies.append(latency)

        if result.get("parse_error"):
            print_result(False, f"Parse failed: {ocr_text[:40]}...", "JSON parse error")
            continue

        # Check each expected field
        field_hits = 0
        field_total = 0
        details = []

        for field, expected_val in expected.items():
            field_total += 1
            actual = result.get(field)
            if isinstance(expected_val, list):
                # Check drug names match (fuzzy)
                actual_str = " ".join(str(a).lower() for a in (actual or []))
                hit = any(e.lower() in actual_str for e in expected_val)
            elif isinstance(expected_val, bool):
                # Just check field is present and non-empty
                hit = bool(actual) and str(actual).lower() not in ("null", "none", "")
            else:
                hit = str(actual) == str(expected_val)

            if hit:
                field_hits += 1
            else:
                details.append(f"missing {field}")

        all_passed = field_hits == field_total
        if all_passed:
            passed_count += 1
        label = f"{ocr_text[:50]}... [{latency}s]"
        print_result(all_passed, label, ", ".join(details) if details else "")
        if verbose:
            print(f"       Extracted: {json.dumps(result, indent=None)[:200]}")

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    accuracy = passed_count / len(EXTRACTION_TESTS)
    print(f"\n  Extraction accuracy: {BOLD}{accuracy:.0%}{RESET} ({passed_count}/{len(EXTRACTION_TESTS)})")
    print(f"  Avg latency:         {BOLD}{avg_latency}s{RESET}")
    return {"accuracy": accuracy, "avg_latency_s": avg_latency}


# ── Test 4: Interaction Checker ────────────────────────────────────────────────
def test_interactions(verbose: bool = False) -> dict:
    print_header("Test 4: Drug Interaction Checker")

    from interactions import check_pair

    # Known interactions to verify
    test_pairs = [
        ("Warfarin",    "Ibuprofen",    ["high", "moderate"]),   # known dangerous combo
        ("Sertraline",  "Tramadol",     ["high", "moderate"]),   # serotonin syndrome risk
        ("Amoxicillin", "Metformin",    ["low", "none"]),        # generally safe
        ("Lisinopril",  "Potassium",    ["moderate", "high"]),   # hyperkalemia risk
    ]

    passed_count = 0
    latencies = []

    for drug_a, drug_b, expected_severities in test_pairs:
        start = time.time()
        result = check_pair(drug_a, drug_b)
        latency = round(time.time() - start, 2)
        latencies.append(latency)

        hit = result.severity in expected_severities
        if hit:
            passed_count += 1

        label = f"{drug_a} + {drug_b} → {result.severity} [{latency}s]"
        detail = f"Expected one of {expected_severities}" if not hit else ""
        print_result(hit, label, detail)
        if verbose:
            print(f"       {result.description[:100]}")

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    accuracy = passed_count / len(test_pairs)
    print(f"\n  Interaction accuracy: {BOLD}{accuracy:.0%}{RESET} ({passed_count}/{len(test_pairs)})")
    print(f"  Avg latency:          {BOLD}{avg_latency}s{RESET}")
    return {"accuracy": accuracy, "avg_latency_s": avg_latency}


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(results: dict):
    print_header("Evaluation Summary")
    scores = []
    for test_name, data in results.items():
        acc = data.get("accuracy", data.get("coverage_score", 1.0 if data.get("passed") else 0.0))
        if data.get("skipped"):
            print(f"  {YELLOW}⏭️  {test_name}: skipped{RESET}")
        else:
            color = GREEN if acc >= 0.8 else YELLOW if acc >= 0.5 else RED
            print(f"  {color}{test_name}: {acc:.0%}{RESET}")
            scores.append(acc)

    if scores:
        overall = sum(scores) / len(scores)
        color = GREEN if overall >= 0.8 else YELLOW if overall >= 0.5 else RED
        print(f"\n  {BOLD}Overall score: {color}{overall:.0%}{RESET}")
        grade = "🟢 Production-ready" if overall >= 0.8 else "🟡 Needs improvement" if overall >= 0.5 else "🔴 Needs work"
        print(f"  {grade}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MediScanAI Evaluation Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--test", choices=["kb", "rag", "extract", "interactions", "all"], default="all")
    args = parser.parse_args()

    print(f"\n{BOLD}MediScanAI — Evaluation Suite{RESET}")
    print(f"Testing against live Groq API (llama-3.3-70b-versatile)")

    results = {}
    t = args.test

    if t in ("kb",           "all"): results["Knowledge Base"]  = test_knowledge_base(args.verbose)
    if t in ("rag",          "all"): results["RAG Retrieval"]   = test_rag_retrieval(args.verbose)
    if t in ("extract",      "all"): results["Drug Extraction"] = test_drug_extraction(args.verbose)
    if t in ("interactions", "all"): results["Interactions"]    = test_interactions(args.verbose)

    if len(results) > 1:
        print_summary(results)

    print()


if __name__ == "__main__":
    main()
