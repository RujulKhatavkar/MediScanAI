"""
MediScanAI - Core Pipeline
OCR -> Drug Extraction -> Semantic RAG (Cohere + numpy) -> Groq LLaMA Q&A

Embeddings stored in embeddings_cache.npy (built once, loaded forever).
No ChromaDB — avoids all file locking / tenant / onnxruntime issues.
Falls back to TF-IDF if no COHERE_API_KEY.
"""

import os, base64, json, re, math, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from groq import Groq

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

QUESTIONS_CSV    = "questions_answers.csv"
OLD_CSV          = "drug_contents.csv"
CACHE_NPY        = "embeddings_cache.npy"
CACHE_NAMES      = "embeddings_names.json"
CHECKPOINT_NPY   = "embeddings_ckpt.npy"
CHECKPOINT_IDX   = "embeddings_ckpt_idx.json"
EMBED_BATCH      = 20
EMBED_SLEEP      = 3.0
RATE_SLEEP       = 65.0
EMBED_MODEL      = "embed-english-light-v3.0"


# ── Groq ───────────────────────────────────────────────────────────────────────
def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not set.")
    return Groq(api_key=key)


# ── OCR ────────────────────────────────────────────────────────────────────────
def ocr_prescription(image_bytes, media_type="image/jpeg"):
    client = get_client()
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    r = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:" + media_type + ";base64," + b64}},
            {"type": "text", "text": (
                "This is a medical prescription. Extract ALL readable text exactly "
                "as it appears including drug names, dosages, instructions, warnings, "
                "patient name, doctor name, refills. Format clearly. Do not interpret."
            )},
        ]}],
    )
    return r.choices[0].message.content


# ── Drug extraction ────────────────────────────────────────────────────────────
def extract_drug_info(ocr_text):
    client = get_client()
    prompt = (
        "From this prescription text, extract structured information.\n"
        "Return ONLY a JSON object (no markdown fences):\n"
        '{"drug_names":["..."],"dosages":["..."],"instructions":["..."],'
        '"warnings":["..."],"patient_name":"...","doctor_name":"...",'
        '"refills":"...","other_notes":"..."}\n\nPrescription:\n' + ocr_text
    )
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = r.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception:
        return {"raw": text, "parse_error": True}


# ── Cohere embeddings ──────────────────────────────────────────────────────────
def _get_cohere():
    try:
        import cohere
        key = os.environ.get("COHERE_API_KEY")
        return cohere.ClientV2(api_key=key) if key else None
    except ImportError:
        return None


def _embed_batch(co, texts, input_type):
    for attempt in range(2):
        try:
            resp = co.embed(
                texts=texts,
                model=EMBED_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
            return list(resp.embeddings.float_)
        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                print("\n[Embed] Rate limit — waiting " + str(int(RATE_SLEEP)) + "s...")
                time.sleep(RATE_SLEEP)
            else:
                print("\n[Embed] Error: " + str(e))
                return None
    return None


def _cosine_sim(q, matrix):
    q = q / (np.linalg.norm(q) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ q


# ── TF-IDF fallback ────────────────────────────────────────────────────────────
class _TFIDF:
    def __init__(self):
        self.docs = []; self.names = []; self.tf_idf = []; self.idf = {}

    def _tok(self, t):
        return re.findall(r"[a-z0-9]+", t.lower())

    def build(self, docs, names):
        self.docs = docs; self.names = names
        n = len(docs); tf_list = []; df = defaultdict(int)
        for doc in docs:
            tokens = self._tok(doc); tf = defaultdict(int)
            for t in tokens: tf[t] += 1
            total = max(len(tokens), 1)
            tf_norm = {t: c / total for t, c in tf.items()}
            tf_list.append(tf_norm)
            for t in tf_norm: df[t] += 1
        self.idf = {t: math.log(n / (1 + c)) for t, c in df.items()}
        self.tf_idf = [{t: v * self.idf.get(t, 0) for t, v in tf.items()} for tf in tf_list]

    def search(self, query, n=5):
        tokens = self._tok(query); qtf = defaultdict(int)
        for t in tokens: qtf[t] += 1
        total = max(len(tokens), 1)
        qv = {t: (c / total) * self.idf.get(t, 0) for t, c in qtf.items()}
        scores = [sum(qv.get(t, 0) * dv.get(t, 0) for t in qv) for dv in self.tf_idf]
        ranked = sorted(
            [(self.names[i], self.docs[i], float(s)) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[2], reverse=True
        )
        return ranked[:n]


# ── Knowledge Base ─────────────────────────────────────────────────────────────
class KnowledgeBase:
    """
    Semantic search via Cohere embeddings stored in numpy cache files.
    - embeddings_cache.npy  : (N, 384) float32 matrix
    - embeddings_names.json : list of drug names (parallel to matrix rows)
    Built once on first run, loaded instantly on every subsequent run.
    Falls back to TF-IDF if COHERE_API_KEY not set.
    """

    def __init__(self):
        self.documents     = []
        self.drug_names    = []
        self.doc_matrix    = None
        self._tfidf        = _TFIDF()
        self._use_semantic = False
        self._loaded       = False

    def _parse_csv(self):
        csv_path = next((p for p in [QUESTIONS_CSV, OLD_CSV] if Path(p).exists()), None)
        if not csv_path:
            print("[RAG] No CSV found.")
            return

        df = pd.read_csv(csv_path)
        drug_docs = {}

        if "Question" in df.columns and "Answer" in df.columns:
            for _, row in df.iterrows():
                q = str(row.get("Question", "")).strip()
                a = str(row.get("Answer", "")).strip()
                if not q or not a or a == "nan":
                    continue
                m = re.search(
                    r"(?:What is |warnings for |before taking |side effects of )(.+?)(?:\?|$)",
                    q, re.I
                )
                name = m.group(1).strip() if m else q[:40]
                drug_docs.setdefault(name, []).append("Q: " + q + "\nA: " + a)
        else:
            for _, row in df.iterrows():
                name = str(row.get("drug_name", row.iloc[0])).strip()
                drug_docs[name] = [" ".join(str(v) for v in row.values if pd.notna(v))]

        for name, pairs in drug_docs.items():
            doc = ("Drug: " + name + "\n\n" + "\n\n".join(pairs))[:500]
            self.documents.append(doc)
            self.drug_names.append(name)

        print("[RAG] Parsed " + str(len(self.documents)) + " drugs from " + csv_path)

    def _build_embeddings(self):
        """Build or load numpy embedding cache. Supports resume on rate limit."""
        co = _get_cohere()
        if co is None:
            print("[RAG] No COHERE_API_KEY — using TF-IDF")
            return

        total = len(self.documents)

        # ── Load complete cache ────────────────────────────────────────────
        if Path(CACHE_NPY).exists() and Path(CACHE_NAMES).exists():
            try:
                cached_names = json.loads(Path(CACHE_NAMES).read_text())
                if len(cached_names) == total:
                    self.doc_matrix    = np.load(CACHE_NPY)
                    self.drug_names    = cached_names
                    self._use_semantic = True
                    print("[RAG] Loaded cache: " + str(self.doc_matrix.shape))
                    return
                else:
                    print("[RAG] Cache size mismatch — rebuilding...")
            except Exception as e:
                print("[RAG] Cache load error: " + str(e))

        # ── Resume from checkpoint ─────────────────────────────────────────
        resume_from = 0
        partial = []
        if Path(CHECKPOINT_NPY).exists() and Path(CHECKPOINT_IDX).exists():
            try:
                resume_from = json.loads(Path(CHECKPOINT_IDX).read_text()).get("next_idx", 0)
                partial = list(np.load(CHECKPOINT_NPY))
                print("[RAG] Resuming from doc " + str(resume_from) + "/" + str(total))
            except Exception:
                resume_from = 0; partial = []

        # ── Embed remaining docs ───────────────────────────────────────────
        remaining = self.documents[resume_from:]
        num_batches = (len(remaining) + EMBED_BATCH - 1) // EMBED_BATCH
        print("[RAG] Embedding " + str(len(remaining)) + " docs (" + str(num_batches) + " batches)...")
        print("[RAG] Runs once, then cached forever.")

        new_vecs = []
        for b, start in enumerate(range(0, len(remaining), EMBED_BATCH)):
            batch = remaining[start: start + EMBED_BATCH]

            # Save checkpoint before each batch
            if new_vecs:
                np.save(CHECKPOINT_NPY, np.array(partial + new_vecs, dtype=np.float32))
                Path(CHECKPOINT_IDX).write_text(
                    json.dumps({"next_idx": resume_from + len(new_vecs)})
                )

            result = _embed_batch(co, batch, "search_document")
            if result is None:
                print("[RAG] Stopped at batch " + str(b) + ". Restart to resume.")
                break
            new_vecs.extend(result)
            done = resume_from + len(new_vecs)
            pct = int(done / total * 100)
            print("[RAG] " + str(done) + "/" + str(total) + " (" + str(pct) + "%)   ", end="\r", flush=True)

            if start + EMBED_BATCH < len(remaining):
                time.sleep(EMBED_SLEEP)

        print("")
        all_vecs = partial + new_vecs
        matrix = np.array(all_vecs, dtype=np.float32)

        if len(matrix) == total:
            self.doc_matrix    = matrix
            self._use_semantic = True
            np.save(CACHE_NPY, matrix)
            Path(CACHE_NAMES).write_text(json.dumps(self.drug_names))
            for f in [CHECKPOINT_NPY, CHECKPOINT_IDX]:
                if Path(f).exists(): Path(f).unlink()
            print("[RAG] Cache saved: " + str(matrix.shape))
        else:
            print("[RAG] Partial (" + str(len(matrix)) + "/" + str(total) + ") — restart to resume")

    def load(self):
        if self._loaded:
            return
        self._parse_csv()
        if self.documents:
            self._build_embeddings()
            self._tfidf.build(self.documents, self.drug_names)
        self._loaded = True
        mode = "Semantic (Cohere)" if self._use_semantic else "TF-IDF (keyword)"
        print("[RAG] Mode: " + mode)

    def search_with_scores(self, query, n=5):
        if not self._loaded:
            self.load()
        if not self.documents:
            return []

        if self._use_semantic and self.doc_matrix is not None:
            co = _get_cohere()
            if co:
                q_result = _embed_batch(co, [query], "search_query")
                if q_result:
                    q_vec = np.array(q_result[0], dtype=np.float32)
                    scores = _cosine_sim(q_vec, self.doc_matrix)
                    top_idx = np.argsort(scores)[::-1][:n]
                    return [
                        (self.drug_names[i], self.documents[i], float(scores[i]))
                        for i in top_idx if scores[i] > 0.1
                    ]

        return self._tfidf.search(query, n=n)

    def size(self):
        if not self._loaded:
            self.load()
        return len(self.documents)

    def search_mode(self):
        if not self._loaded:
            self.load()
        return "semantic" if self._use_semantic else "tfidf"


# Single global instance — loaded once per Streamlit session
_kb = KnowledgeBase()


# ── Public API ─────────────────────────────────────────────────────────────────
def rag_lookup(query, n_results=5):
    results = _kb.search_with_scores(query, n=n_results)
    if not results:
        return ""
    return "\n\n---\n\n".join(doc[:800] for _, doc, _ in results)


def rag_lookup_with_citations(query, n_results=5):
    results = _kb.search_with_scores(query, n=n_results)
    if not results:
        return "", [], 0.0
    max_s = results[0][2] if results else 1.0
    citations = []
    parts = []
    for name, doc, score in results:
        norm = round(min(score / max(max_s, 1e-6), 1.0), 3)
        citations.append({
            "drug":    name,
            "score":   norm,
            "snippet": doc[:300].replace("\n", " "),
        })
        parts.append(doc[:800])
    top3 = [c["score"] for c in citations[:3]]
    conf = round(sum(top3) / len(top3), 3) if top3 else 0.0
    return "\n\n---\n\n".join(parts), citations, conf


def get_kb_size():
    return _kb.size()


def get_search_mode():
    return _kb.search_mode()


def answer_question(question, ocr_text, drug_info, chat_history):
    client = get_client()
    drug_names = ", ".join(drug_info.get("drug_names") or [])
    rag_context, citations, confidence = rag_lookup_with_citations(
        question + " " + drug_names
    )
    system = (
        "You are MediScanAI, a helpful medical information assistant.\n"
        "IMPORTANT: Information only, not medical advice. Recommend consulting a doctor.\n\n"
        "PRESCRIPTION:\n" + (ocr_text or "None") + "\n\n"
        "DRUG INFO:\n" + str(drug_info or "None") + "\n\n"
        "KNOWLEDGE BASE (RAG):\n" + (rag_context or "No matches") + "\n\n"
        "Answer clearly. Be honest if uncertain."
    )
    clean_history = [{"role": m["role"], "content": m["content"]} for m in chat_history]
    messages = [{"role": "system", "content": system}]
    messages += clean_history
    messages.append({"role": "user", "content": question})
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=messages,
    )
    return r.choices[0].message.content, citations, confidence