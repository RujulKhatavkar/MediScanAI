"""
MediScanAI - Knowledge Base Builder
Scrapes drugs.com A-Z → saves questions_answers.csv → ingests into ChromaDB

Run ONCE before starting the app:
    python build_knowledge_base.py

Options:
    python build_knowledge_base.py --limit 50     # only scrape first 50 drugs (quick test)
    python build_knowledge_base.py --skip-scrape  # skip scraping, just re-ingest existing CSV
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import argparse
import os
import sys
import chromadb
import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DRUGS_COM_BASE = "https://www.drugs.com"
CSV_PATH = "questions_answers.csv"
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "drug_knowledge"
REQUEST_DELAY = 0.5   # seconds between requests — be polite to drugs.com
BATCH_SIZE = 100


# ── Step 1: Scrape drug list A–Z ───────────────────────────────────────────────
def get_drug_links_for_letter(letter: str) -> list[tuple[str, str]]:
    """Returns list of (drug_name, relative_url) for one letter."""
    url = f"{DRUGS_COM_BASE}/alpha/{letter}.html"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  Failed to fetch letter {letter}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for ul in soup.find_all("ul", class_="ddc-list-column-2"):
        for li in ul.find_all("li"):
            name = li.get_text(strip=True)
            a = li.find("a")
            if a and name:
                results.append((name, a.get("href", "")))
    return results


def get_all_drug_links(limit: int = None) -> list[tuple[str, str]]:
    """Scrape all drug names + links from A–Z pages."""
    all_links = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        print(f"  Fetching index: {letter.upper()}...", end=" ", flush=True)
        links = get_drug_links_for_letter(letter)
        all_links.extend(links)
        print(f"{len(links)} drugs")
        time.sleep(REQUEST_DELAY)
        if limit and len(all_links) >= limit:
            all_links = all_links[:limit]
            break
    return all_links


# ── Step 2: Scrape individual drug pages ──────────────────────────────────────
def scrape_drug_page(drug_name: str, href: str) -> dict | None:
    """Scrape uses, warnings, before_taking, side_effects for one drug."""
    if not href:
        return None
    url = href if href.startswith("http") else f"{DRUGS_COM_BASE}{href}"

    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    def get_section(section_id: str) -> str:
        """Extract text from a named section div."""
        # Try data-next-head or section with matching id/class
        section = soup.find("div", {"id": section_id})
        if not section:
            # Try h2 with matching text
            for h in soup.find_all(["h2", "h3"]):
                if section_id.replace("-", " ").lower() in h.get_text().lower():
                    texts = []
                    for sib in h.find_next_siblings():
                        if sib.name in ["h2", "h3"]:
                            break
                        texts.append(sib.get_text(separator=" ", strip=True))
                    return " ".join(texts)[:2000]
        if section:
            return section.get_text(separator=" ", strip=True)[:2000]
        return ""

    # Main content area
    main = soup.find("div", class_="contentBox") or soup.find("article") or soup.body

    def extract_between_headings(keyword: str) -> str:
        if not main:
            return ""
        for tag in main.find_all(["h2", "h3", "b", "strong"]):
            if keyword.lower() in tag.get_text().lower():
                parts = []
                for sib in tag.find_next_siblings():
                    if sib.name in ["h2", "h3"]:
                        break
                    text = sib.get_text(separator=" ", strip=True)
                    if text:
                        parts.append(text)
                result = " ".join(parts)[:2000]
                if result:
                    return result
        return ""

    uses         = extract_between_headings("what is") or extract_between_headings("used for") or extract_between_headings("uses")
    warnings     = extract_between_headings("warnings") or extract_between_headings("important")
    before_taking = extract_between_headings("before taking") or extract_between_headings("before you take")
    side_effects = extract_between_headings("side effects")

    # Only return if we got at least some content
    if not any([uses, warnings, side_effects]):
        return None

    return {
        "ingredient": drug_name,
        "uses": uses or f"{drug_name} is a prescription medication.",
        "warnings": warnings or "Consult your doctor or pharmacist for warnings.",
        "before_taking": before_taking or "Consult your doctor before taking this medication.",
        "Side Effects": side_effects or "Consult your doctor about possible side effects.",
    }


# ── Step 3: Save to CSV ────────────────────────────────────────────────────────
def save_to_csv(records: list[dict], path: str):
    """Save Q&A records to CSV in the same format as your original notebook."""
    rows = []
    for rec in records:
        name = rec["ingredient"]
        rows.extend([
            [f"What is {name} used for?",              rec["uses"]],
            [f"What are the warnings for {name}?",     rec["warnings"]],
            [f"What should be known before taking {name}?", rec["before_taking"]],
            [f"What are the side effects of {name}?",  rec["Side Effects"]],
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])
        writer.writerows(rows)

    print(f"\n✅ Saved {len(records)} drugs ({len(rows)} Q&A pairs) → {path}")


# ── Step 4: Ingest CSV into ChromaDB ──────────────────────────────────────────
def ingest_to_chromadb(csv_path: str, chroma_dir: str):
    """Load Q&A CSV into ChromaDB with rich document text for retrieval."""
    print(f"\n📦 Ingesting {csv_path} into ChromaDB at {chroma_dir}...")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ CSV is empty, nothing to ingest.")
        return

    # Group Q&A pairs back into per-drug documents for better chunking
    # Each document = all 4 Q&A pairs for one drug → richer context per chunk
    drug_docs = {}
    for _, row in df.iterrows():
        q = str(row.get("Question", ""))
        a = str(row.get("Answer", ""))
        if not q or not a:
            continue
        # Extract drug name from question pattern
        drug_name = q.split("What is ")[-1].split(" used")[0].strip() if "What is " in q else q[:50]
        if drug_name not in drug_docs:
            drug_docs[drug_name] = []
        drug_docs[drug_name].append(f"Q: {q}\nA: {a}")

    # Build document list — one doc per drug with all Q&A concatenated
    documents, ids, metadatas = [], [], []
    for i, (drug_name, qa_pairs) in enumerate(drug_docs.items()):
        doc_text = f"Drug: {drug_name}\n\n" + "\n\n".join(qa_pairs)
        documents.append(doc_text)
        ids.append(f"drug_{i}")
        metadatas.append({"drug_name": drug_name, "qa_count": len(qa_pairs)})

    # Init ChromaDB (persistent)
    client = chromadb.PersistentClient(path=chroma_dir)

    # Delete old collection if exists, rebuild fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  🗑️  Cleared old collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for better semantic search
    )

    # Batch ingest
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        pct = int(end / total * 100)
        print(f"  Ingested {end}/{total} drugs ({pct}%)", end="\r", flush=True)

    print(f"\n✅ ChromaDB ready: {collection.count()} drug documents in '{COLLECTION_NAME}'")
    print(f"   Location: {os.path.abspath(chroma_dir)}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build MediScanAI knowledge base")
    parser.add_argument("--limit",       type=int, default=None, help="Max drugs to scrape (default: all)")
    parser.add_argument("--skip-scrape", action="store_true",    help="Skip scraping, just re-ingest existing CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("  MediScanAI — Knowledge Base Builder")
    print("=" * 60)

    if args.skip_scrape:
        if not Path(CSV_PATH).exists():
            print(f"❌ {CSV_PATH} not found. Run without --skip-scrape first.")
            sys.exit(1)
        print(f"⏭️  Skipping scrape, using existing {CSV_PATH}")
    else:
        # ── Scrape ──
        limit_msg = f" (limit: {args.limit})" if args.limit else " (all drugs A–Z)"
        print(f"\n🌐 Step 1: Scraping drug index from drugs.com{limit_msg}")
        drug_links = get_all_drug_links(limit=args.limit)
        print(f"\n  Found {len(drug_links)} drugs total")

        print(f"\n🔬 Step 2: Scraping individual drug pages...")
        records = []
        failed = 0
        for i, (name, href) in enumerate(drug_links):
            print(f"  [{i+1}/{len(drug_links)}] {name}...", end=" ", flush=True)
            rec = scrape_drug_page(name, href)
            if rec:
                records.append(rec)
                print("✓")
            else:
                failed += 1
                print("✗ (skipped)")
            time.sleep(REQUEST_DELAY)

        print(f"\n  Scraped: {len(records)} drugs, skipped: {failed}")

        if not records:
            print("❌ No data scraped. Check your internet connection.")
            sys.exit(1)

        print(f"\n💾 Step 3: Saving to {CSV_PATH}")
        save_to_csv(records, CSV_PATH)

    # ── Ingest ──
    print(f"\n📦 Step 4: Building ChromaDB vector store")
    ingest_to_chromadb(CSV_PATH, CHROMA_DIR)

    print("\n" + "=" * 60)
    print("  ✅ Knowledge base ready! Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()