#!/usr/bin/env python3
"""
Streamlit UI:
- ⬆️ Upload & Inspect: upload PDF/MD, send to Vector Store and/or parse→JSONL→SQLite
  • Also includes local FTS (SQLite) search and chunk inspection
  • Optional “Quick parse” that extracts text per page directly (no JSONL step)
- 💬 Ask: RAG over OpenAI Vector Store (Responses API; Assistants fallback)
- 🗂️ Summarize ALL: per-file summaries
- 📁 Files: list files in Vector Store
"""

import os, time, sqlite3, tempfile
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads import Run

# Optional: direct PDF→SQLite quick-parse (no JSONL) path
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ---- page setup ----
st.set_page_config(page_title="PDF Q&A • Vector Store", page_icon="📄", layout="centered")

# ---- theme (Stanford palette + readable white cards) ----
st.markdown("""
<style>
:root{--accent:#8C1515;--accent-2:#B1040E;--border:#E5E7EB;--muted:#4D4F53;--card-bg:#FFFFFF;--chip-bg:#FBE9EA}
div.card,div.answer-card{border:1px solid var(--border);background:var(--card-bg);padding:16px 18px;border-radius:14px;box-shadow:0 1px 8px rgba(0,0,0,.06);margin:.4rem 0}
div.card,div.answer-card,div.card * ,div.answer-card * {color:#1f2937 !important}
span.chip{display:inline-block;padding:4px 10px;border-radius:999px;background:var(--chip-bg);margin-right:6px;font-size:.85rem;color:var(--accent);border:1px solid #F3C9CB}
div.stButton>button{background:var(--accent);color:#fff;border:1px solid var(--accent);border-radius:10px;padding:.55rem .9rem;font-weight:600}
div.stButton>button:hover{background:var(--accent-2);border-color:var(--accent-2)}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--accent);border-bottom:2px solid var(--accent);font-weight:700}
[data-testid="stProgressBar"]>div>div>div{background:var(--accent)!important}
a{color:var(--accent-2)}
</style>""", unsafe_allow_html=True)

# ---- env / secrets ----
if "OPENAI_API_KEY" in st.secrets: os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "OPENAI_VECTOR_STORE_ID" in st.secrets: os.environ["OPENAI_VECTOR_STORE_ID"] = st.secrets["OPENAI_VECTOR_STORE_ID"]
load_dotenv(Path(__file__).with_name(".env"))

DB_PATH = "papers.db"          # SQLite for local FTS/inspection
CHUNKS_DIR = "json_chunks"     # Folder for JSONL pipeline output

def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI()

# ------------------------- Local Structured DB helpers -------------------------

def ensure_local_db():
    """Create basic chunk tables + FTS (idempotent)."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            page INTEGER,
            section TEXT,
            text TEXT
        )
    """)
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(doc_id, page, section, text, tokenize='unicode61')
    """)
    con.commit()
    con.close()

def insert_chunk(doc_id: str, page: int, section: str, text: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO chunks (doc_id, page, section, text) VALUES (?,?,?,?)",
                (doc_id, page, section, text))
    con.execute("INSERT INTO chunks_fts (doc_id, page, section, text) VALUES (?,?,?,?)",
                (doc_id, page, section, text))
    con.commit(); con.close()

def run_local_search(query: str, limit: int = 10) -> pd.DataFrame:
    """Search via FTS if available, else fall back to LIKE."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    exists = con.execute("""
        SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'
    """).fetchone()
    try:
        if exists:
            rows = con.execute("""
                SELECT doc_id, page, section, substr(text,1,220) AS snippet
                FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?
            """, (query, limit)).fetchall()
        else:
            rows = con.execute("""
                SELECT doc_id, page, section, substr(text,1,220) AS snippet
                FROM chunks WHERE text LIKE '%'||?||'%' LIMIT ?
            """, (query, limit)).fetchall()
        df = pd.DataFrame(rows, columns=["doc_id","page","section","snippet"])
    except sqlite3.OperationalError as e:
        st.error(f"Local search failed: {e}")
        df = pd.DataFrame(columns=["doc_id","page","section","snippet"])
    finally:
        con.close()
    return df

def quick_pdf_to_sqlite(pdf_file: Path, doc_name: str):
    """Direct per-page text extraction into SQLite (no JSONL step)."""
    if not HAS_PYPDF2:
        st.error("PyPDF2 not installed (pip install pypdf2).")
        return
    ensure_local_db()
    reader = PdfReader(str(pdf_file))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        insert_chunk(doc_name, i+1, f"Page {i+1}", text)

# ------------------------- OpenAI RAG helpers -------------------------

DEFAULT_SYSTEM = """You are a retrieval-first assistant for scientific PDFs attached via File Search.
QUERY REFORMULATION: rewrite into 1–4 sub-queries with synonyms/acronyms.
RETRIEVAL: use File Search; retry broader once if recall is weak.
ANSWER: exactly one sentence with one short quote in “double quotes”, ending with [<filename> p.<page> §<section>] (use p.—/§— if unknown). If nothing relevant, output: No direct evidence found in the provided files.
"""

def list_vs_files(client: OpenAI, vs_id: str) -> List[Tuple[str, str]]:
    after=None; items=[]
    while True:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=100, after=after)
        items.extend(page.data)
        if not page.has_more: break
        after = page.last_id
    out=[]
    for it in items:
        try:
            f = client.files.retrieve(it.id)
            out.append((it.id, f.filename or it.id))
        except Exception:
            out.append((it.id, it.id))
    return out

def render_sources(client: OpenAI, file_ids: List[str]):
    if not file_ids:
        st.markdown('<span class="chip">No file citations</span>', unsafe_allow_html=True); return
    for fid in file_ids[:10]:
        try:
            f = client.files.retrieve(fid)
            st.markdown(f'<span class="chip">{getattr(f,"filename",fid) or fid}</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<span class="chip">{fid}</span>', unsafe_allow_html=True)

def extract_file_ids_from_responses(resp) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for block in resp.output:
            if getattr(block, "type", "") == "message":
                for c in (block.message.content or []):
                    if getattr(c,"type","")=="output_text":
                        for ann in (getattr(c,"annotations",[]) or []):
                            if getattr(ann,"type","")=="file_citation":
                                fc = getattr(ann,"file_citation",None)
                                if fc and getattr(fc,"file_id",None): file_ids.add(fc.file_id)
    except Exception:
        pass
    return list(file_ids)

def extract_file_ids_from_messages(messages_list) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for msg in messages_list:
            if getattr(msg,"role","")!="assistant": continue
            for item in (msg.content or []):
                if getattr(item,"type","")=="text":
                    for ann in (getattr(item.text,"annotations",[]) or []):
                        if getattr(ann,"type","")=="file_citation":
                            fid = getattr(ann.file_citation,"file_id",None)
                            if fid: file_ids.add(fid)
    except Exception:
        pass
    return list(file_ids)

def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    # current SDK supports tool_resources, but some envs accept file_search kw; try both.
    try:
        return client.responses.create(
            model=model,
            input=[{"role":"system","content":system.strip()},
                   {"role":"user","content":userq.strip()}],
            tools=[{"type":"file_search"}],
            file_search={"vector_store_ids":[vs_id]},
        ), "responses"
    except Exception:
        pass
    try:
        return client.responses.create(
            model=model,
            input=[{"role":"system","content":system.strip()},
                   {"role":"user","content":userq.strip()}],
            tools=[{"type":"file_search"}],
            tool_resources={"file_search":{"vector_store_ids":[vs_id]}},
        ), "responses"
    except Exception:
        return client.responses.create(
            model=model,
            input=[{"role":"system","content":system.strip()},
                   {"role":"user","content":userq.strip()}],
            tools=[{"type":"file_search"}],
            extra_body={"tool_resources":{"file_search":{"vector_store_ids":[vs_id]}}},
        ), "responses"

def ask_with_assistants(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    asst = client.beta.assistants.create(
        name="Streamlit PDF QA (temp)",
        model=model,
        instructions=system.strip(),
        tools=[{"type":"file_search"}],
        tool_resources={"file_search":{"vector_store_ids":[vs_id]}},
    )
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=userq.strip())
    run: Run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst.id)
    deadline = time.time()+60*6; sleep_s=.75
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in {"completed","failed","cancelled","expired"}: break
        if time.time()>deadline: raise RuntimeError("Run timed out.")
        time.sleep(sleep_s); sleep_s=min(sleep_s*1.5, 6.0)
    if run.status!="completed": raise RuntimeError(f"Run did not complete: {run.status}")
    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5)
    answer_text = ""
    for m in msgs.data:
        if m.role!="assistant": continue
        for item in m.content:
            if item.type=="text":
                answer_text = item.text.value or ""; break
        if answer_text: break
    file_ids = extract_file_ids_from_messages(msgs.data)
    return {"output_text":answer_text, "_raw":{"messages":msgs, "assistant_id":asst.id, "thread_id":thread.id}}, "assistants", file_ids

# ---- sidebar ----
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
vector_store_id = st.sidebar.text_input("OPENAI_VECTOR_STORE_ID", value=os.getenv("OPENAI_VECTOR_STORE_ID",""))
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
show_raw = st.sidebar.checkbox("Show raw response", value=False)
if api_key: os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id: os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

# ---- header ----
st.markdown('<div style="display:flex;gap:.6rem;align-items:center;"><span style="font-size:1.6rem">📄</span><h1 style="margin:0">PDF Q&A</h1></div>', unsafe_allow_html=True)
st.caption("Upload → Ingest → Search (Vector Store & Structured DB)")

# ---- tabs ----
tab_upload, tab_ask, tab_all, tab_files = st.tabs(["⬆️ Upload & Inspect", "💬 Ask", "🗂️ Summarize ALL", "📁 Files"])


# ================== Tab: Upload & Inspect ==================
with tab_upload:
    st.markdown('<div class="card"><h4 class="compact">Upload & Inspect</h4><div>PDF/Markdown → Vector Store and/or Structured DB. Includes local FTS search.</div></div>', unsafe_allow_html=True)

    sub_up, sub_search, sub_inspect = st.tabs(["Upload & Ingest", "Local FTS Search", "Inspect"])

    # ---- Upload & ingest
    with sub_up:
        uf = st.file_uploader("Drop a PDF or Markdown", type=["pdf","md","txt"])
        if uf is not None:
            up_dir = Path("Uploads"); up_dir.mkdir(exist_ok=True)
            save_path = up_dir / uf.name
            with open(save_path, "wb") as f: f.write(uf.read())
            st.success(f"Saved: {save_path}")

            c1, c2, c3 = st.columns(3)

            # 1) Upload to Vector Store
            if c1.button("Upload to Vector Store"):
                try:
                    client = get_client()
                    vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
                    if not vs_id: st.error("Set OPENAI_VECTOR_STORE_ID."); st.stop()
                    with open(save_path, "rb") as fh:
                        up = client.files.create(file=fh, purpose="assistants")
                    client.vector_stores.file_batches.create(vector_store_id=vs_id, file_ids=[up.id])
                    st.success("Queued for indexing in Vector Store.")
                except Exception as e:
                    st.error(f"Vector upload failed: {e}")

            # 2) JSONL→SQLite pipeline (your existing scripts)
            if c2.button("Parse → JSONL → SQLite"):
                try:
                    Path(CHUNKS_DIR).mkdir(exist_ok=True)
                    os.system(f'python3 parse_papers_to_json.py --input "{save_path}" --output "{CHUNKS_DIR}" --max-tokens 800 --overlap 120')
                    os.system(f'python3 build_jsonl_sqlite.py --input "{CHUNKS_DIR}" --db "{DB_PATH}"')
                    st.success("Parsed and loaded into SQLite.")
                except Exception as e:
                    st.error(f"Structured pipeline failed: {e}")

            # 3) Optional: Quick direct parse (no JSONL)
            if c3.button("Quick parse to SQLite (no JSONL)"):
                if save_path.suffix.lower() != ".pdf":
                    st.warning("Quick parse currently supports PDF only.")
                else:
                    with st.spinner("Extracting text by page..."):
                        quick_pdf_to_sqlite(save_path, uf.name)
                        st.success("Inserted pages into SQLite (chunks/chunks_fts).")

    # ---- Local FTS Search
    with sub_search:
        ensure_local_db()
        query = st.text_input("Search (FTS5 syntax supported, fallback to LIKE):", placeholder='e.g., "granule cells" NEAR/5 timing')
        limit = st.slider("Max results", 5, 100, 15)
        if st.button("Run local search"):
            if not query.strip():
                st.warning("Enter a query.")
            else:
                df = run_local_search(query.strip(), limit)
                if df.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(df, use_container_width=True)

    # ---- Inspect
    with sub_inspect:
        ensure_local_db()
        if st.button("Show 100 sample chunks"):
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT doc_id, page, substr(text,1,140) AS preview FROM chunks LIMIT 100", con)
            con.close()
            st.dataframe(df, use_container_width=True)


# ================== Tab: Ask (RAG over Vector Store) ==================
with tab_ask:
    with st.expander("Advanced (system prompt)", expanded=False):
        system_text = st.text_area("System", value=DEFAULT_SYSTEM, height=200, label_visibility="collapsed")
    question = st.text_input("Your question", placeholder="e.g., In ChenEtAl2020.pdf, which region was imaged and by what method?")
    if st.button("Ask", type="primary", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"): st.error("Provide OPENAI_API_KEY."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"): st.error("Provide OPENAI_VECTOR_STORE_ID."); st.stop()
            client = get_client()
            with st.spinner("Thinking..."):
                try:
                    resp, _ = ask_with_responses(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = getattr(resp, "output_text", None) or "(no text)"
                    file_ids = extract_file_ids_from_responses(resp); raw_obj = resp
                except Exception:
                    resp, _, file_ids = ask_with_assistants(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = resp["output_text"]; raw_obj = resp["_raw"]
            st.markdown('<div class="answer-card"><b>Answer</b></div>', unsafe_allow_html=True)
            st.write(answer_text)
            st.markdown('<div class="card"><b>Sources</b></div>', unsafe_allow_html=True)
            render_sources(client, file_ids)
            if show_raw:
                st.markdown("**Raw**")
                try: st.write(raw_obj.model_dump())
                except Exception: st.write(str(raw_obj))
        except Exception as e:
            st.error(f"Error: {e}")

# ================== Tab: Summarize ALL ==================
with tab_all:
    st.markdown('<div class="card"><h4 class="compact">Summarize every file</h4><div>Runs a focused question per PDF.</div></div>', unsafe_allow_html=True)
    if st.button("Summarize ALL files", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"): st.error("Provide OPENAI_API_KEY."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"): st.error("Provide OPENAI_VECTOR_STORE_ID."); st.stop()
            client = get_client(); vs_id = os.environ["OPENAI_VECTOR_STORE_ID"]
            files = list_vs_files(client, vs_id)
            if not files: st.warning("No files in Vector Store."); st.stop()
            progress = st.progress(0.0); total=len(files); results=[]
            for i,(fid,fname) in enumerate(files, start=1):
                per_q = (f"Summarize **{fname}** in 2–3 sentences. "
                         f"Use only content from {fname}. Include one short quote and end with "
                         f"[{fname} p.— §—].")
                with st.spinner(f"Summarizing {fname} ({i}/{total})..."):
                    try:
                        try:
                            resp,_ = ask_with_responses(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                            text = getattr(resp,"output_text",None) or "(no text)"
                            fids = extract_file_ids_from_responses(resp)
                        except Exception:
                            resp,_, fids = ask_with_assistants(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                            text = resp["output_text"]
                        st.markdown(f"**{fname}**")
                        st.write(text); render_sources(client, fids); st.markdown("---")
                        results.append(f"### {fname}\n\n{text}\n")
                    except Exception as e:
                        st.error(f"{fname}: {e}")
                progress.progress(i/total)
            if results:
                all_md = "# Summaries\n\n" + "\n\n".join(results)
                st.download_button("Download summaries (Markdown)", data=all_md.encode("utf-8"), file_name="summaries.md", mime="text/markdown")
            st.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")

# ================== Tab: Files ==================
with tab_files:
    st.markdown('<div class="card"><h4 class="compact">Files in your Vector Store</h4></div>', unsafe_allow_html=True)
    if st.button("Refresh file list"):
        try:
            if not os.getenv("OPENAI_API_KEY"): st.error("Provide OPENAI_API_KEY."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"): st.error("Provide OPENAI_VECTOR_STORE_ID."); st.stop()
            client = get_client()
            files = list_vs_files(client, os.environ["OPENAI_VECTOR_STORE_ID"])
            if not files: st.warning("No files found.")
            else:
                for _, fname in files: st.markdown(f"- {fname}")
        except Exception as e:
            st.error(f"Error: {e}")
