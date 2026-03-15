"""
MediScanAI - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
from PIL import Image
import io
from pipeline import ocr_prescription, extract_drug_info, answer_question, get_kb_size, get_search_mode
from interactions import check_all_interactions, severity_color, severity_bg, severity_label
from database import (
    save_prescription, save_interactions, save_chat_message,
    get_all_prescriptions, get_prescription, delete_prescription, get_stats, init_db
)

st.set_page_config(
    page_title="MediScanAI",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        
        background: linear-gradient(135deg, #23b28a 0%, #4cff53 100%;
        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.25rem 0 0; opacity: 0.85; font-size: 1rem; }
    .info-card {
        background: white; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    }
    .info-card h4 { color: #1e3a5f; margin: 0 0 0.5rem; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .info-card p  { margin: 0; color: #374151; }
    .drug-tag     { display:inline-block; background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:20px; font-size:0.85rem; margin:2px; font-weight:500; }
    .warning-tag  { display:inline-block; background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:0.85rem; margin:2px; }
    .step-badge   { background:#2563eb; color:white; border-radius:50%; width:24px; height:24px; display:inline-flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:bold; margin-right:8px; }
    .chat-message-user { background:#eff6ff; border-left:3px solid #2563eb; padding:0.75rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; }
    .chat-message-ai   { background:white; border:1px solid #e2e8f0; border-left:3px solid #10b981; padding:0.75rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; }
    .citation-box  { background:#f8faff; border:1px solid #c7d7f9; border-radius:8px; padding:0.6rem 0.9rem; margin:4px 0; font-size:0.82rem; }
    .conf-bar-wrap { background:#e5e7eb; border-radius:4px; height:8px; margin:4px 0; }
    .conf-bar      { background:#2563eb; border-radius:4px; height:8px; }
    .interaction-card { border-radius:10px; padding:1rem 1.25rem; margin-bottom:0.75rem; border:1px solid; }
    .stat-card    { background:white; border:1px solid #e2e8f0; border-radius:10px; padding:1rem; text-align:center; }
    .stat-num     { font-size:1.8rem; font-weight:600; color:#1e3a5f; }
    .stat-lbl     { font-size:0.8rem; color:#6b7280; margin-top:2px; }
    .disclaimer   { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px; padding:0.75rem 1rem; font-size:0.85rem; color:#92400e; margin-top:1rem; }
    .history-item { background:white; border:1px solid #e2e8f0; border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem; cursor:pointer; }
    .history-item:hover { border-color:#93c5fd; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("ocr_text", None), ("drug_info", None), ("chat_history", []),
    ("image_bytes", None), ("media_type", "image/jpeg"),
    ("pending_question", None), ("interactions", None),
    ("current_prescription_id", None), ("active_tab", "scanner"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def safe_list(value):
    if not value or not isinstance(value, list):
        return []
    return [str(i) for i in value if i is not None and str(i).strip()]


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⛑️ MediScanAI</h1>
    <p>Prescription scanner · Drug interaction checker · Medical Q&A · History tracking</p>
</div>
""", unsafe_allow_html=True)

# ── Top nav tabs ───────────────────────────────────────────────────────────────
tab_scanner, tab_history, tab_stats = st.tabs(["🔬 Scanner", "📋 History", "📊 Stats"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab_scanner:
    left_col, right_col = st.columns([1, 1.6], gap="large")

    # ── LEFT: Upload + Extracted info ─────────────────────────────────────────
    with left_col:
        kb_size = get_kb_size()
        mode = get_search_mode()
        mode_label = "🔵 semantic" if mode == "semantic" else "🟡 keyword"
        kb_color = "#16a34a" if kb_size > 100 else "#ca8a04" if kb_size > 0 else "#dc2626"
        st.markdown(f'<div style="font-size:10px;color:{kb_color};margin-bottom:10px;">● Knowledge base: {kb_size} drugs · {mode_label}</div>', unsafe_allow_html=True)

        st.markdown("### <span class='step-badge'>1</span> Upload Prescription", unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","heic","webp"], label_visibility="collapsed")

        if uploaded:
            ext = uploaded.name.split(".")[-1].lower()
            st.session_state.media_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp","heic":"image/jpeg"}.get(ext, "image/jpeg")
            st.session_state.image_bytes = uploaded.read()
            try:
                st.image(Image.open(io.BytesIO(st.session_state.image_bytes)), caption="Uploaded prescription", use_container_width=True)
            except Exception:
                st.info("Preview not available for this format.")

            if st.button("🔍 Extract Drug Information", type="primary", use_container_width=True):
                with st.spinner("Running OCR..."):
                    try:
                        ocr = ocr_prescription(st.session_state.image_bytes, st.session_state.media_type)
                        st.session_state.ocr_text = ocr
                    except Exception as e:
                        st.error(f"OCR error: {e}"); st.stop()
                with st.spinner("Extracting drug info..."):
                    try:
                        info = extract_drug_info(st.session_state.ocr_text)
                        st.session_state.drug_info = info
                        st.session_state.chat_history = []
                        st.session_state.interactions = None
                        st.session_state.pending_question = None
                    except Exception as e:
                        st.error(f"Extraction error: {e}"); st.stop()
                with st.spinner("Saving to history..."):
                    pid = save_prescription(uploaded.name, st.session_state.ocr_text, st.session_state.drug_info)
                    st.session_state.current_prescription_id = pid
                st.success("✅ Done!")

        # ── Extracted info display ─────────────────────────────────────────
        if st.session_state.drug_info:
            info = st.session_state.drug_info
            st.markdown("### <span class='step-badge'>2</span> Extracted Information", unsafe_allow_html=True)

            if info.get("parse_error"):
                st.warning("Could not parse structured info.")
                st.code(info.get("raw", ""))
            else:
                drug_names_list = safe_list(info.get("drug_names"))
                if drug_names_list:
                    st.markdown('<div class="info-card"><h4>💊 Medications</h4>' + "".join(f'<span class="drug-tag">{d}</span>' for d in drug_names_list) + "</div>", unsafe_allow_html=True)

                dosages_list = safe_list(info.get("dosages"))
                if dosages_list:
                    st.markdown('<div class="info-card"><h4>📏 Dosages</h4><p>' + "<br>".join(dosages_list) + "</p></div>", unsafe_allow_html=True)

                instructions_list = safe_list(info.get("instructions"))
                if instructions_list:
                    st.markdown('<div class="info-card"><h4>📋 Instructions</h4><p>' + "<br>".join(instructions_list) + "</p></div>", unsafe_allow_html=True)

                warnings_list = safe_list(info.get("warnings"))
                if warnings_list:
                    st.markdown('<div class="info-card"><h4>⚠️ Warnings</h4>' + "".join(f'<span class="warning-tag">{w}</span>' for w in warnings_list) + "</div>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    p = info.get("patient_name")
                    if p and str(p).lower() not in ("null","none",""):
                        st.markdown(f'<div class="info-card"><h4>👤 Patient</h4><p>{p}</p></div>', unsafe_allow_html=True)
                with c2:
                    d = info.get("doctor_name")
                    if d and str(d).lower() not in ("null","none",""):
                        st.markdown(f'<div class="info-card"><h4>🩺 Doctor</h4><p>{d}</p></div>', unsafe_allow_html=True)

                r = info.get("refills")
                if r and str(r).lower() not in ("null","none",""):
                    st.markdown(f'<div class="info-card"><h4>🔄 Refills</h4><p>{r}</p></div>', unsafe_allow_html=True)

            with st.expander("📄 View raw OCR text"):
                st.text(st.session_state.ocr_text)

            # ── Drug Interaction Checker ───────────────────────────────────
            st.markdown("### <span class='step-badge'>3</span> Interaction Check", unsafe_allow_html=True)
            drug_names = safe_list(info.get("drug_names")) if not info.get("parse_error") else []

            if len(drug_names) < 2:
                st.info("Need 2+ drugs to check interactions.")
            else:
                if st.button("⚗️ Check Drug Interactions", use_container_width=True):
                    with st.spinner(f"Checking {len(drug_names)} drugs for interactions..."):
                        try:
                            interactions = check_all_interactions(drug_names)
                            st.session_state.interactions = interactions
                            if st.session_state.current_prescription_id:
                                save_interactions(st.session_state.current_prescription_id, interactions)
                        except Exception as e:
                            st.error(f"Error: {e}")

            if st.session_state.interactions:
                interactions = st.session_state.interactions
                high   = [i for i in interactions if i.severity == "high"]
                mod    = [i for i in interactions if i.severity == "moderate"]
                low_no = [i for i in interactions if i.severity in ("low", "none")]

                if high:
                    st.error(f"⛔ {len(high)} high-risk interaction(s) found!")
                elif mod:
                    st.warning(f"⚠️ {len(mod)} moderate interaction(s) found.")
                else:
                    st.success("✅ No significant interactions detected.")

                for ix in interactions:
                    bg  = severity_bg(ix.severity)
                    col = severity_color(ix.severity)
                    lbl = severity_label(ix.severity)
                    st.markdown(f"""
                    <div class="interaction-card" style="background:{bg}; border-color:{col};">
                        <div style="font-weight:600; color:{col}; margin-bottom:6px;">{lbl} — {ix.drug_a} + {ix.drug_b}</div>
                        <div style="font-size:0.9rem; color:#374151; margin-bottom:6px;">{ix.description}</div>
                        <div style="font-size:0.85rem; color:#6b7280;"><strong>Recommendation:</strong> {ix.recommendation}</div>
                        {"<div style='font-size:0.75rem; color:#9ca3af; margin-top:4px;'>Sources: " + ", ".join(ix.sources[:3]) + "</div>" if ix.sources else ""}
                    </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown('<div class="info-card" style="text-align:center;color:#6b7280;padding:2rem;"><div style="font-size:2.5rem">📋</div><p style="margin-top:0.5rem">Upload a prescription image to get started</p></div>', unsafe_allow_html=True)

    # ── RIGHT: Chat ────────────────────────────────────────────────────────────
    with right_col:
        st.markdown("### <span class='step-badge'>4</span> Ask Questions", unsafe_allow_html=True)

        if not st.session_state.ocr_text:
            st.markdown('<div class="info-card" style="text-align:center;padding:3rem;color:#6b7280;"><div style="font-size:3rem">💬</div><p style="font-size:1.1rem;margin-top:0.5rem">Extract prescription info first, then ask anything</p><p style="font-size:0.9rem;margin-top:0.25rem">Side effects · Drug interactions · Dosage · What it treats</p></div>', unsafe_allow_html=True)
        else:
            # ── Process ALL input FIRST, then render history ──────────────
            # Chat input box (must be processed before rendering history)
            question = st.chat_input("Ask about your prescription...")

            # Handle typed question
            if question:
                with st.spinner("Thinking..."):
                    try:
                        answer, citations, confidence = answer_question(
                            question=question,
                            ocr_text=st.session_state.ocr_text,
                            drug_info=st.session_state.drug_info or {},
                            chat_history=st.session_state.chat_history,
                        )
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": answer,
                            "citations": citations, "confidence": confidence,
                        })
                        if st.session_state.current_prescription_id:
                            save_chat_message(st.session_state.current_prescription_id, "user", question)
                            save_chat_message(st.session_state.current_prescription_id, "assistant", answer)
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Handle suggestion button pending question
            if st.session_state.pending_question:
                q = st.session_state.pending_question
                st.session_state.pending_question = None
                with st.spinner("Thinking..."):
                    try:
                        answer, citations, confidence = answer_question(
                            question=q,
                            ocr_text=st.session_state.ocr_text,
                            drug_info=st.session_state.drug_info or {},
                            chat_history=st.session_state.chat_history,
                        )
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": answer,
                            "citations": citations, "confidence": confidence,
                        })
                        if st.session_state.current_prescription_id:
                            save_chat_message(st.session_state.current_prescription_id, "user", q)
                            save_chat_message(st.session_state.current_prescription_id, "assistant", answer)
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Suggestion buttons (shown only before first message)
            if not st.session_state.chat_history:
                st.markdown("**Try asking:**")
                drug_names = safe_list(st.session_state.drug_info.get("drug_names") if st.session_state.drug_info else [])
                drug = drug_names[0] if drug_names else "this medication"
                suggestions = [
                    f"What are the side effects of {drug}?",
                    f"What is {drug} used to treat?",
                    "Are there any drug interactions I should know about?",
                    "What should I do if I miss a dose?",
                ]
                c1, c2 = st.columns(2)
                for i, sug in enumerate(suggestions):
                    with (c1 if i % 2 == 0 else c2):
                        if st.button(sug, key=f"sug_{i}", use_container_width=True):
                            st.session_state.pending_question = sug

            # Chat history (rendered AFTER processing — always up to date)
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-message-user">🙋 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-ai">🤖 <strong>MediScanAI:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

                    # Confidence + Citations
                    citations  = msg.get("citations", [])
                    confidence = msg.get("confidence", 0.0)
                    if citations or confidence:
                        with st.expander(f"📚 Sources & Confidence ({confidence:.0%})", expanded=False):
                            bar_width = int(confidence * 100)
                            conf_color = "#16a34a" if confidence > 0.6 else "#ca8a04" if confidence > 0.3 else "#dc2626"
                            st.markdown(f"""
                            <div style="margin-bottom:8px;">
                                <span style="font-size:12px;color:#6b7280;">RAG confidence</span>
                                <div class="conf-bar-wrap">
                                    <div class="conf-bar" style="width:{bar_width}%;background:{conf_color};"></div>
                                </div>
                                <span style="font-size:11px;color:{conf_color};font-weight:600;">{confidence:.0%}</span>
                            </div>""", unsafe_allow_html=True)
                            for cit in citations[:4]:
                                score_pct = int(cit["score"] * 100)
                                st.markdown(f"""
                                <div class="citation-box">
                                    <strong>{cit["drug"]}</strong>
                                    <span style="float:right;color:#6b7280;">{score_pct}% match</span><br>
                                    <span style="color:#6b7280;">{cit["snippet"][:120]}...</span>
                                </div>""", unsafe_allow_html=True)

            if st.session_state.chat_history:
                if st.button("🗑️ Clear chat"):
                    st.session_state.chat_history = []
                    st.session_state.pending_question = None

            st.markdown('<div class="disclaimer">⚠️ <strong>Medical Disclaimer:</strong> MediScanAI provides information only and is not a substitute for professional medical advice. Always consult your doctor or pharmacist.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Prescription History")
    prescriptions = get_all_prescriptions()

    if not prescriptions:
        st.info("No prescriptions scanned yet. Upload one in the Scanner tab.")
    else:
        for rx in prescriptions:
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{rx['scanned_at']}** — {rx['image_name']}")
                if rx["drug_names"]:
                    st.markdown(f"<small style='color:#6b7280'>{rx['drug_names']}</small>", unsafe_allow_html=True)
            with c2:
                if st.button("View", key=f"view_{rx['id']}"):
                    detail = get_prescription(rx["id"])
                    if detail:
                        with st.expander(f"Prescription #{rx['id']} — {rx['scanned_at']}", expanded=True):
                            st.text_area("OCR Text", detail["ocr_text"], height=120, disabled=True)
                            if detail["interactions"]:
                                st.markdown("**Interactions checked:**")
                                for ix in detail["interactions"]:
                                    col = severity_color(ix["severity"])
                                    lbl = severity_label(ix["severity"])
                                    st.markdown(f'<span style="color:{col};">{lbl}</span> — {ix["drug_a"]} + {ix["drug_b"]}: {ix["description"]}', unsafe_allow_html=True)
                            if detail["chat"]:
                                st.markdown("**Chat history:**")
                                for msg in detail["chat"]:
                                    prefix = "🙋 You" if msg["role"] == "user" else "🤖 AI"
                                    st.markdown(f"**{prefix}:** {msg['content'][:200]}")
            with c3:
                if st.button("🗑️", key=f"del_{rx['id']}", help="Delete this prescription"):
                    delete_prescription(rx["id"])
                    st.rerun()
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("### 📊 Usage Statistics")
    stats = get_stats()
    kb_size = get_kb_size()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Prescriptions Scanned",    stats["total_prescriptions"]),
        (c2, "Interactions Checked",     stats["total_interactions_checked"]),
        (c3, "High-Risk Flags",          stats["high_risk_interactions"]),
        (c4, "Drugs in Knowledge Base",  kb_size),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{value}</div><div class="stat-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Run evaluation suite:**")
    st.code("python eval.py", language="bash")
    st.code("python eval.py --verbose   # detailed output\npython eval.py --test rag   # just RAG tests", language="bash")