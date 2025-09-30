# app/streamlit_app.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st
st._config.set_option("theme.base", "light")
import hashlib
import pickle, re, html
import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from core.repo import append_records, save_snapshot, append_log_rows, read_log_df
from core.parsing import parse_notes
from core.records import df_to_records
from core.indexing import build_from_json

st.set_page_config(page_title="Meeting Notes Studio", layout="wide")

# Force light look for inputs + tables (works with Streamlit 1.4x)
st.markdown("""
<style>
:root{
  --mn-primary:#ef4444;
  --mn-text:#111827;
  --mn-bg:#ffffff;
  --mn-bg2:#f9fafb;
  --mn-border:#e5e7eb;
}

/* App background */
body, .stApp {
  background: var(--mn-bg) !important;
  color: var(--mn-text) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
  color: var(--mn-text) !important;
  background: var(--mn-bg) !important;
  border-radius: 8px 8px 0 0;
  padding: 8px 12px;
}
.stTabs [aria-selected="true"]{
  border-bottom: 2px solid var(--mn-primary) !important;
  font-weight: 700 !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea{
  background: var(--mn-bg) !important;
  color: var(--mn-text) !important;
  border: 1px solid var(--mn-border) !important;
}

/* Dataframe + Editor wrapper */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  background: var(--mn-bg) !important;
  color: var(--mn-text) !important;
  border: 1px solid var(--mn-border) !important;
  border-radius: 6px;
}

/* Force grid/table cells to white */
[data-testid="stDataFrame"] div[role="gridcell"],
[data-testid="stDataEditor"] div[role="gridcell"] {
  background: var(--mn-bg) !important;
  color: var(--mn-text) !important;
  border-color: var(--mn-border) !important;
}

/* Headers */
[data-testid="stDataFrame"] div[role="columnheader"],
[data-testid="stDataEditor"] div[role="columnheader"] {
  background: var(--mn-bg2) !important;
  color: var(--mn-text) !important;
  font-weight: 600;
}

/* Row stripes */
[data-testid="stDataFrame"] [role="row"]:nth-child(even) div[role="gridcell"],
[data-testid="stDataEditor"] [role="row"]:nth-child(even) div[role="gridcell"] {
  background: var(--mn-bg2) !important;
}

/* Buttons */
.stButton > button {
  background: var(--mn-primary) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px;
  padding: 8px 14px;
  font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---- Pre-render session defaults & clear hook ----
if "notes_text" not in st.session_state:
    st.session_state["notes_text"] = ""
if "parsed_df" not in st.session_state:
    st.session_state["parsed_df"] = None
if "raw_text" not in st.session_state:
    st.session_state["raw_text"] = ""

# If a previous action requested a clear, do it BEFORE widgets are created
if st.session_state.get("clear_inputs", False):
    st.session_state["notes_text"] = ""
    st.session_state["parsed_df"] = None
    st.session_state["raw_text"] = ""
    st.session_state["clear_inputs"] = False

# Show any pending success message (survives rerun once)
flash = st.session_state.pop("flash_success", None)
if flash:
    st.success(flash)

# --- Tabs ---
tab_search, tab_input = st.tabs(["🔎 Search", "📝 Input"])

with tab_input:
    st.subheader("Paste your raw notes")
    col_left, col_right = st.columns([2, 1], gap="large")

    # ----- LEFT: inputs + parse -----
    with col_left:
        default_year = datetime.now().year
        year = st.number_input(
            "Fallback year (used if headers lack a year)",
            min_value=2000, max_value=2100, value=default_year, step=1
        )

        notes_text = st.text_area(
            "Notes text",
            height=300,
            key="notes_text",
            placeholder="e.g.\nCall Notes | 7/18\nHi Ben, Sam...\nOffsets 2025: file.pptx\n..."
        )

        if st.button("Submit #1 — Parse to table", type="primary"):
            if not notes_text.strip():
                st.warning("Please paste some notes first.")
            else:
                df = parse_notes(notes_text, default_year=year)
                st.session_state["parsed_df"] = df
                st.session_state["raw_text"] = notes_text
                st.success(f"Parsed {len(df)} rows.")

    # ----- RIGHT: ingestion log -----
    with col_right:
        st.markdown("#### Ingestion Log")
        log_df = read_log_df()
        if st.button("Rebuild search index"):
            try:
                with st.spinner("Rebuilding embeddings & FAISS index…"):
                    stats = build_from_json()
                st.success(f"Index rebuilt with {stats['num_vectors']} vectors.")
                st.session_state.pop("faiss_index", None)   # ← force reload in Search
                st.session_state.pop("faiss_meta", None)

            except Exception as e:
                import traceback
                st.error("Index rebuild failed:")
                st.code(traceback.format_exc(), language="python")
        if log_df.empty:
            st.caption("No notes ingested yet.")
        else:
            st.dataframe(
                log_df.sort_values("Upload Date", ascending=False),
                use_container_width=True,
                height=300
            )

    st.markdown("---")

    # ----- Editable table + Submit #2 (SAVE) -----
    if st.session_state.get("parsed_df") is not None:
        st.markdown("#### Edit parsed table (you can modify any cell)")
        edited = st.data_editor(
            st.session_state["parsed_df"],
            use_container_width=True,
            num_rows="dynamic",
            key="editor_parsed",
        )
        st.session_state["parsed_df"] = pd.DataFrame(edited)

        if st.button("Submit #2 — Confirm & Save to Repository", type="secondary"):
            df_final = st.session_state["parsed_df"]
            if df_final is None or df_final.empty:
                st.warning("Nothing to save. Parse notes first.")
            else:
                # 1) convert to JSON records
                records = df_to_records(df_final)
                # 2) append to repository JSON
                total = append_records(records)
                # 3) save a snapshot (audit)
                raw = st.session_state.get("raw_text", "")
                snap = save_snapshot(raw) if raw else None

                # 4) log rows (Notes Type, Date, Upload Date, Source ID)
                now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                source_id = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:8] if raw else ""
                mini = df_final[["Date", "Notes Type"]].copy()
                mini = mini.astype(str).applymap(lambda s: s.strip())
                mini = mini[(mini["Date"] != "") & (mini["Notes Type"] != "")]
                mini = mini.drop_duplicates()
                if not mini.empty:
                    mini["Upload Date"] = now_iso
                    mini["Source ID"] = source_id
                    append_log_rows(mini.to_dict(orient="records"))

                # 5) build FAISS index (blocking with spinner)
                try:
                    with st.spinner("Building embeddings & FAISS index…"):
                        stats = build_from_json()
                    msg = f"Saved {len(records)} rows. Repository now has {total} records. Index rebuilt with {stats['num_vectors']} vectors."
                    st.session_state["flash_success"] = msg
                    st.session_state.pop("faiss_index", None)   # ← force reload in Search
                    st.session_state.pop("faiss_meta", None)

                except Exception as e:
                    import traceback
                    st.error("Indexing failed:")
                    st.code(traceback.format_exc(), language="python")
                    # keep the saved-rows message
                    st.session_state["flash_success"] = "Saved rows. Indexing failed — see error above."

                # 6) clear controls on next render and rerun
                st.session_state["clear_inputs"] = True
                st.rerun()

# ====== Search helpers (ported from old app, adapted to repository/) ======
ALWAYS_ENTITIES = {"cwt", "gbt", "sbt", "sbti", "gt", "eod", "bcg", "esg", "mck", "amex"}

# NEW: entity aliases / synonyms (canonical -> any of these substrings in notes)
ENTITY_ALIASES = {
    "gt": ["grant thornton", "grant-thornton", "grantthornton"]
}

def _norm(s):
    return (s or "").strip()

_STOP = set("""
what when where who whom whose which why how the a an and or of on in for to with from about near-term near term
is are was were be been being do does did doing have has had having this that these those there here over under into onto
meeting meetings discussion discussions discuss discussed talk talks talked talking notes note say said
please find below we anything any ever regard regarding regards relate related relating decide decides decided deciding 
decision decisions review reviewed reviewing update updates updated updating confirm confirms confirmed confirming
plan plans planned planning status statuses around pertaining view
""".split())

def extract_entities(q: str):
    # normalize alias phrases to canonical token before tokenization
    q_norm = q
    for canon, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            q_norm = re.sub(rf"\b{re.escape(alias)}\b", canon, q_norm, flags=re.IGNORECASE)
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", q_norm)
    ents = []
    for t in toks:
        raw = t.strip()
        low = raw.lower()
        if low in _STOP:
            continue
        if low in ALWAYS_ENTITIES:
            ents.append(low); continue
        if raw.isupper() and len(raw) >= 2:
            ents.append(low); continue
        if len(low) >= 4:
            ents.append(low)
    return ents

def haystack_from_meta(m: dict) -> str:
    fields = [
        m.get("Action / Comment", ""),
        m.get("Comment", ""),
        m.get("Workstream", ""),
        m.get("Sub-workstream", ""),
        m.get("Participants", ""),
        m.get("Notes Type", ""),
        m.get("Speaker", ""),
        m.get("File Link", ""),
    ]
    return " ".join(_norm(x).lower() for x in fields)

# Alias-aware term match: e.g., "gt" -> "grant thornton"
def _matches(term: str, hay: str) -> bool:
    t = (term or "").lower()
    if t in ENTITY_ALIASES:
        for alias in ENTITY_ALIASES[t] + [t]:
            if alias in hay:
                return True
        return False
    return t in hay


def words_to_highlight(q: str, strict: bool = False):
    try:
        ents = extract_entities(q, strict=strict)
    except TypeError:
        ents = extract_entities(q)
    out = set(ents)
    for e in ents:
        for alias in ENTITY_ALIASES.get(e.lower(), []):
            out.add(alias)
    return list(out)

def hilite(text: str, q_words):
    safe = html.escape(text)
    for w in sorted(set(q_words), key=len, reverse=True):
        safe = re.sub(fr"(?i)\b({re.escape(w)})\b", r"<mark>\1</mark>", safe)
    return safe

def embed_query_openai(text: str, model: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=text, model=model)
    v = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def retrieve_hybrid(index, metadata, query: str, model: str, api_key: str,
                    top_k: int = 15, vec_k: int = 100, require_all_entities: bool = True):
    """
    Hybrid retrieval with debug:
      - FAISS (larger window) -> candidates
      - Strict keyword UNION across full repo
      - Optional month-year date scope (e.g., "Sep 2025")
      - Stores per-hit debug rows in st.session_state["last_debug"]
    Returns: [(row_index, metadata[row_index]), ...]
    """
    import re, calendar
    # ---------- helpers ----------
    def parse_month_year_range(q: str):
        months = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
                  "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
                  "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
        m = re.search(r'\b(in\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})\b', q, re.I)
        if not m: return None
        mon = months[m.group(2).lower()]; y = int(m.group(3))
        return f"{y:04d}-{mon:02d}-01", f"{y:04d}-{mon:02d}-{calendar.monthrange(y, mon)[1]:02d}"

    def date_in_scope(meta, start_end):
        if not start_end: return True
        date = (meta.get("Date") or "")[:10]
        if not date: return False
        s, e = start_end
        return s <= date <= e

    def haystack_from_meta(m: dict) -> str:
        fields = [
            m.get("Action / Comment", ""), m.get("Comment", ""),
            m.get("Workstream", ""), m.get("Sub-workstream", ""),
            m.get("Participants", ""), m.get("Notes Type", ""),
            m.get("Speaker", ""), m.get("File Link", "")
        ]
        return " ".join((x or "").strip().lower() for x in fields)

    # ---------- embed & FAISS ----------
    qvec = embed_query_openai(query, model=model, api_key=api_key)
    ntotal = getattr(index, "ntotal", 0)
    sweep_k = min(max(vec_k, top_k * 20), max(ntotal, top_k))  # sweep deeper when small corpus
    D, I = index.search(qvec, sweep_k)

    # ---------- entities (robust) ----------
    raw_ents = extract_entities(query) or []
    if not raw_ents:  # token fallback if extractor returns nothing
        toks = re.findall(r"[a-z0-9&'\-\.]+", query.lower())
        stop = {"the","a","an","and","or","of","for","to","in","on","by","with","at","from","as"}
        raw_ents = [t for t in toks if t and t not in stop][:5]
    ents = [e.lower() for e in raw_ents]
    entset = set(ents)
    daterng = parse_month_year_range(query)

    candidates = []  # tuples: (score, idx, hits, source, sim)
    # A) FAISS window
    for sim, i in zip(D[0], I[0]):
        if i < 0 or i >= len(metadata): continue
        m = metadata[i]
        if not date_in_scope(m, daterng): continue
        hay = haystack_from_meta(m)
        if ents:
            hits = sum(1 for e in ents if _matches(e, hay))
            if require_all_entities and hits < len(ents): continue
        else:
            hits = 0
        coverage = (hits / max(1, len(ents))) if ents else 0.0
        score = 0.8 * float(sim) + 0.2 * coverage
        candidates.append((score, i, hits, "faiss", float(sim)))

    # B) Fallback: any-entity if strict gave nothing
    if not candidates and ents:
        for sim, i in zip(D[0], I[0]):
            if i < 0 or i >= len(metadata): continue
            m = metadata[i]
            if not date_in_scope(m, daterng): continue
            hay = haystack_from_meta(m)
            hits = sum(1 for e in ents if _matches(e, hay))
            if hits == 0: continue
            coverage = hits / len(ents)
            score = 0.7 * float(sim) + 0.3 * coverage
            candidates.append((score, i, hits, "faiss_any", float(sim)))

    # C) UNION: add every row that contains ALL entities (date-scoped)
    if ents:
        already = {i for (_s, i, _h, _src, _sim) in candidates}
        max_sim = float(max(D[0])) if len(D[0]) else 0.0
        union_boost = max(0.95, max_sim + 0.05)  # make exact keyword hits surface
        for i, m in enumerate(metadata):
            if i in already: continue
            if not date_in_scope(m, daterng): continue
            hay = haystack_from_meta(m)
            if all(_matches(e, hay) for e in entset):
                candidates.append((union_boost, i, len(entset), "union", 0.0))

    # D) Sort, slice, and stash debug
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_k]
    top_ids = [i for (_s, i, _h, _src, _sim) in top]

    # Debug rows for quick inspection
    dbg = []
    for rank, (score, i, hits, src, sim) in enumerate(top, 1):
        m = metadata[i]
        dbg.append({
            "rank": rank, "row_idx": i, "source": src,
            "score": round(float(score), 3), "sim": round(float(sim), 3), "hits": int(hits),
            "date": (m.get("Date") or "")[:10],
            "ws": (m.get("Workstream") or ""),
            "snippet": (m.get("Action / Comment") or m.get("Comment") or "")[:120]
        })
    st.session_state["last_debug"] = dbg
    st.session_state["last_query_ents"] = ents
    st.session_state["last_sweep_k"] = sweep_k
    st.session_state["last_top_k"] = top_k

    return [(i, metadata[i]) for i in top_ids]

def group_results(indexed_metas, all_metadata=None):
    """
    Group bullets by (Date, Notes Type, Participants, Workstream, Sub-workstream, Speaker).
    If all_metadata is provided, expand each matched group to include *all* bullets from the
    repository that share the same grouping key (not just the matched rows).
    """
    def key_of(m):
        date = (m.get("Date") or "").strip().split()[0]
        notes_type = (m.get("Notes Type") or "").strip()
        participants = (m.get("Participants") or "").strip()
        if notes_type.lower() == "eod notes":
            participants = ""
        ws = (m.get("Workstream") or "").strip()
        sub = (m.get("Sub-workstream") or "").strip()
        speaker = (m.get("Speaker") or "").strip()
        return (date, notes_type, participants, ws, sub, speaker)

    def comment_of(m):
        return ((m.get("Action / Comment") or m.get("Comment") or "") or "").strip()

    # 1) Seed groups from matched rows (as before)
    groups = {}
    for row_idx, m in indexed_metas:
        k = key_of(m)
        c = comment_of(m)
        if not c:
            continue
        g = groups.get(k)
        if g is None:
            g = {"seen": {}, "meta": m, "meta_idx": row_idx}
            groups[k] = g
        else:
            if row_idx < g["meta_idx"]:
                g["meta"] = m
                g["meta_idx"] = row_idx
        if c not in g["seen"] or row_idx < g["seen"][c]:
            g["seen"][c] = row_idx

    # 2) Expand bullets with all rows from the same group (entire meeting)
    if all_metadata is not None and groups:
        for i, m in enumerate(all_metadata):
            k = key_of(m)
            g = groups.get(k)
            if not g:
                continue
            c = comment_of(m)
            if not c:
                continue
            if c not in g["seen"] or i < g["seen"][c]:
                g["seen"][c] = i
            # Prefer earliest row as representative header
            if i < g.get("meta_idx", i + 1):
                g["meta"] = m
                g["meta_idx"] = i

    # 3) Finalize bullets
    for g in groups.values():
        g["bullets"] = [t[1] for t in sorted(((idx, c) for c, idx in g["seen"].items()), key=lambda t: t[0])]
        g.pop("meta_idx", None)

    return groups

with tab_search:
    st.subheader("Search your repository")

    # Lazy-load index & metadata from repository/
    repo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "repository")
    idx_path = os.path.join(repo_dir, "faiss_index.bin")
    meta_path = os.path.join(repo_dir, "metadata.pkl")

    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        st.info("No index found yet. Please ingest notes in the Input tab to build the index.")
    else:
        try:
            idx_mtime  = os.path.getmtime(idx_path)
            meta_mtime = os.path.getmtime(meta_path)
            needs_reload = (
                "faiss_index" not in st.session_state or
                "faiss_meta"  not in st.session_state  or
                st.session_state.get("idx_mtime")  != idx_mtime  or
                st.session_state.get("meta_mtime") != meta_mtime
            )
            if needs_reload:
                st.session_state["faiss_index"] = faiss.read_index(idx_path)
                with open(meta_path, "rb") as f:
                    st.session_state["faiss_meta"] = pickle.load(f)
                st.session_state["idx_mtime"]  = idx_mtime
                st.session_state["meta_mtime"] = meta_mtime
        except Exception as e:
            st.error(f"Failed to load index/metadata: {e}")
            st.stop()

        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        if not api_key:
            st.error("Missing OPENAI_API_KEY in .env at project root.")
            st.stop()

        question = st.text_input("Ask a question about past discussions:")
        require_all = st.toggle(
            "Strict match",
            value=False,
            help="ON = Precise (use for terms like ‘Doyon’, ‘Tracera’, etc.); OFF = Broad (use for open questions)"
        )

        if question:
            with st.spinner("Searching your meeting notes..."):
                try:
                    indexed = retrieve_hybrid(
                        index=st.session_state["faiss_index"],
                        metadata=st.session_state["faiss_meta"],
                        query=question,
                        model=embedding_model,
                        api_key=api_key,
                        top_k=15,
                        vec_k=min(2000, st.session_state["faiss_index"].ntotal),
                        require_all_entities=require_all
                    )
                    groups = group_results(indexed, st.session_state["faiss_meta"])
                    q_words = words_to_highlight(question, strict=require_all)
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.stop()

                show_dbg = st.toggle("Show debug for top results", value=False)
                if show_dbg:
                    dbg = st.session_state.get("last_debug", [])
                    if dbg:
                        st.caption(
                            f"Entities: {', '.join(st.session_state.get('last_query_ents', []))} · "
                            f"vec_k sweep: {st.session_state.get('last_sweep_k')} · "
                            f"top_k: {st.session_state.get('last_top_k')}"
                        )
                        st.dataframe(pd.DataFrame(dbg), use_container_width=True, height=260)

            if not groups:
                st.info("No matching notes found.")
            else:
                # --- light card styles ---
                st.markdown("""
                <style>
                .group-card { border:1px solid #e5e7eb; background:#f8fafc;
                  border-radius:16px; padding:18px 20px; margin:14px 0; }
                .group-header { font-weight:600; font-size:1.05rem; margin-bottom:8px; color:#111827; }
                .divider { height:1px; background:#e5e7eb; margin:10px 0 6px 0; }
                .group-bullets { margin: 0 0 0 1.1rem; padding: 0; color:#111827; }
                .group-bullets li { margin:6px 0; line-height:1.5; }
                </style>
                """, unsafe_allow_html=True)

                for (date, notes_type, participants, ws, sub, speaker) in sorted(
                    groups.keys(), key=lambda k: k[0], reverse=True
                ):
                    group = groups[(date, notes_type, participants, ws, sub, speaker)]
                    bullets = group["bullets"]
                    meta = group.get("meta", {})

                    file_link = _norm(meta.get("File Link") or meta.get("File Attached") or "")
                    
                    tokens = [f"📅 {hilite(date or '—', q_words)}",
                            f"🗒️ {hilite(notes_type or '—', q_words)}"]

                    if participants:
                        tokens.append(f"👥 {hilite(participants, q_words)}")
                    if ws:
                        tokens.append(f"🧩 {hilite(ws, q_words)}")
                    if sub:
                        tokens.append(f"🔗 {hilite(sub, q_words)}")
                    if speaker and (notes_type or "").strip() != "EoD Notes":
                        tokens.append(f"🗣️ {hilite(speaker, q_words)}")
                    if file_link:
                        tokens.append(f"📎 <a href='{html.escape(file_link)}' target='_blank'>Attachment</a>")

                    header = " &nbsp;&nbsp;•&nbsp;&nbsp; ".join(tokens)
                    card_html = (
                        f"<div class='group-card'>"
                        f"<div class='group-header'>{header}</div>"
                        f"<div class='divider'></div>"
                        f"<ul class='group-bullets'>"
                        + "".join(f"<li>{hilite(text, q_words)}</li>" for text in bullets)
                        + "</ul></div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.caption("Tip: try queries like “Discussions on Copastur”, “SBTi near-term target rationale”, “CWT duplicates”.")
