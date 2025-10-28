#!/usr/bin/env python3
"""
PDF Q&A with Vector Store + Login + Library

Tabs:
- ⬆️ Upload & Inspect: upload PDF/MD, send to Vector Store; optional parse→JSONL→SQLite
- 💬 Ask: RAG over OpenAI Vector Store (Responses API; Assistants fallback)
- 🗂️ Summarize ALL: short per-file summaries
- 📁 Files: list files in Vector Store
- 📚 Library: browse uploaded files with who/when (backed by local SQLite library.db)

Auth:
- streamlit-authenticator with secrets shim (secrets are read-only, so we deep-copy)
"""

import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads import Run

# ----------------------------- Page & Theme -----------------------------
st.set_page_config(page_title="PDF Q&A • Vector Store", page_icon="📄", layout="wide")

st.markdown("""
<style>
:root{
  --accent:#8C1515; --accent-2:#B1040E; --border:#E5E7EB;
  --muted:#4D4F53; --card-bg:#FFFFFF; --chip-bg:#FBE9EA
}
div.card,div.answer-card{
  border:1px solid var(--border);background:var(--card-bg);
  padding:16px 18px;border-radius:14px;box-shadow:0 1px 8px rgba(0,0,0,.06);
  margin:.4rem 0
}
div.card,div.answer-card,div.card * ,div.answer-card * {color:#1f2937 !important}
span.chip{display:inline-block;padding:4px 10px;border-radius:999px;
  background:var(--chip-bg);margin-right:6px;font-size:.85rem;color:var(--accent);
  border:1px solid #F3C9CB}
div.stButton>button{
  background:var(--accent);color:#fff;border:1px solid var(--accent);
  border-radius:10px;padding:.55rem .9rem;font-weight:600}
div.stButton>button:hover{background:var(--accent-2);border-color:var(--accent-2)}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color:var(--accent);border-bottom:2px solid var(--accent);font-weight:700}
[data-testid="stProgressBar"]>div>div>div{background:var(--accent)!important}
a{color:var(--accent-2)}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Env / Secrets -----------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "OPENAI_VECTOR_STORE_ID" in st.secrets:
    os.environ["OPENAI_VECTOR_STORE_ID"] = st.secrets["OPENAI_VECTOR_STORE_ID"]
load_dotenv(Path(__file__).with_name(".env"))

# ----------------------------- Helpers: Auth -----------------------------
def _to_plain_dict(obj):
    """Recursively convert Mapping-like obj (incl. Streamlit Secrets sections) to plain dict."""
    try:
        items = obj.items()
    except Exception:
        return obj
    return {k: _to_plain_dict(v) for k, v in items}

def build_authenticator():
    # From secrets if configured
    if "auth" in st.secrets:
        raw_auth = _to_plain_dict(st.secrets["auth"])
        cookie_cfg = _to_plain_dict(raw_auth.get("cookie", {}))
        creds = _to_plain_dict(raw_auth.get("credentials", {"usernames": {}}))

        if "usernames" not in creds:
            creds["usernames"] = {}

        cleaned = {"usernames": {}}
        for uname, u in creds["usernames"].items():
            cleaned["usernames"][uname] = {
                "name": u.get("name", uname),
                "email": u.get("email", ""),
                "password": u.get("password", ""),  # bcrypt hash
            }

        authenticator = stauth.Authenticate(
            credentials=cleaned,
            cookie_name=cookie_cfg.get("name", "vs_app_session"),
            key=cookie_cfg.get("key", "cookie-key"),
            cookie_expiry_days=int(cookie_cfg.get("expiry_days", 14)),
        )
        return authenticator, True

    # Fallback demo login (no secrets provided)
    hashed = stauth.Hasher(["demo"]).generate()
    authenticator = stauth.Authenticate(
        credentials={"usernames": {
            "demo": {"name": "Demo User", "email": "demo@example.com", "password": hashed[0]}
        }},
        cookie_name="vs_app_session",
        key="cookie-key",
        cookie_expiry_days=14,
    )
    return authenticator, False

# ----------------------------- Authentication -----------------------------
authenticator, using_secrets = build_authenticator()

# The login method must be called with the correct parameters
# Different versions of streamlit-authenticator have different APIs
try:
    # Try calling login with just the location parameter (newer versions)
    name, auth_status, username = authenticator.login(location='sidebar')
except TypeError as e:
    # If that fails, try with form_name parameter (some versions)
    try:
        name, auth_status, username = authenticator.login('Login', 'sidebar')
    except Exception as e2:
        # Last resort: try with key parameter
        try:
            name, auth_status, username = authenticator.login(key='login_form', location='sidebar')
        except Exception as e3:
            st.error(f"Authentication error: {str(e3)}")
            st.error("Please check your streamlit-authenticator version. Try: pip install streamlit-authenticator==0.2.3")
            st.stop()

# Now we can use the sidebar for status messages
with st.sidebar:
    if auth_status:
        st.success(f"Hello, {name}!")
        authenticator.logout("Logout", "sidebar")
    elif auth_status is False:
        st.error("Username/password is incorrect.")
    elif auth_status is None:
        st.info("Please log in to continue.")

if not auth_status:
    st.stop()

# ----------------------------- OpenAI Helpers -----------------------------
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI()

DEFAULT_SYSTEM = """You are a retrieval-first assistant for scientific PDFs attached via File Search.
QUERY REFORMULATION: rewrite into 1–4 sub-queries with synonyms/acronyms.
RETRIEVAL: always call File Search; if recall is weak, retry once broadly.
ANSWER: exactly one sentence with one short quote in "double quotes", ending with [<filename> p.<page> §<section>] (use p.—/§— if unknown).
If nothing relevant: No direct evidence found in the provided files.
"""

def list_vs_files(client: OpenAI, vs_id: str) -> List[Tuple[str, str]]:
    after = None
    items = []
    while True:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=100, after=after)
        items.extend(page.data)
        if not page.has_more:
            break
        after = page.last_id
    out = []
    for it in items:
        try:
            f = client.files.retrieve(it.id)
            out.append((it.id, f.filename or it.id))
        except Exception:
            out.append((it.id, it.id))
    return out

def render_sources(client: OpenAI, file_ids: List[str]):
    if not file_ids:
        st.markdown('<span class="chip">No file citations</span>', unsafe_allow_html=True)
        return
    for fid in file_ids[:10]:
        try:
            f = client.files.retrieve(fid)
            st.markdown(f'<span class="chip">{getattr(f, "filename", fid) or fid}</span>',
                        unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<span class="chip">{fid}</span>', unsafe_allow_html=True)

def extract_file_ids_from_responses(resp) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for block in resp.output:
            if getattr(block, "type", "") == "message":
                for c in (block.message.content or []):
                    if getattr(c, "type", "") == "output_text":
                        for ann in (getattr(c, "annotations", []) or []):
                            if getattr(ann, "type", "") == "file_citation":
                                fc = getattr(ann, "file_citation", None)
                                if fc and getattr(fc, "file_id", None):
                                    file_ids.add(fc.file_id)
    except Exception:
        pass
    return list(file_ids)

def extract_file_ids_from_assistant(msg) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for c in msg.content:
            if c.type == "text" and hasattr(c, "text"):
                for ann in c.text.annotations:
                    if ann.type == "file_citation":
                        if hasattr(ann.file_citation, "file_id"):
                            file_ids.add(ann.file_citation.file_id)
    except Exception:
        pass
    return list(file_ids)

def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, question: str):
    resp = client.responses.create(
        model=model,
        input_text=question,
        system=[{"type": "text", "text": system}],
        tools=[
            {
                "type": "file_search",
                "file_search": {"vector_store_ids": [vs_id]},
            }
        ],
        response_format={"type": "text"},
    )
    return resp, extract_file_ids_from_responses(resp)

def ask_with_assistants(client: OpenAI, model: str, vs_id: str, system: str, question: str):
    asst = client.beta.assistants.create(
        model=model,
        instructions=system,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )
    thread = client.beta.threads.create(messages=[{"role": "user", "content": question}])
    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=asst.id)
    if run.status != "completed":
        raise RuntimeError(f"Run {run.id} status: {run.status}")
    msgs = list(client.beta.threads.messages.list(thread_id=thread.id, order="asc"))
    if not msgs:
        raise RuntimeError("No messages returned.")
    last_msg = msgs[-1]
    text_parts = [c.text.value for c in last_msg.content if c.type == "text"]
    fids = extract_file_ids_from_assistant(last_msg)
    client.beta.assistants.delete(asst.id)
    return {"output_text": " ".join(text_parts), "_raw": last_msg}, run, fids

# ----------------------------- Library DB -----------------------------
DB_PATH = Path("library.db")

def init_library():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS library (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def upsert_file_meta(file_id: str, filename: str, uploaded_by: str):
    init_library()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT OR REPLACE INTO library (file_id, filename, uploaded_by, uploaded_at) VALUES (?, ?, ?, ?)",
        (file_id, filename, uploaded_by, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

def query_library(username_filter: str = None):
    init_library()
    conn = sqlite3.connect(str(DB_PATH))
    if username_filter:
        rows = conn.execute(
            "SELECT file_id, filename, uploaded_by, uploaded_at FROM library WHERE uploaded_by = ? ORDER BY uploaded_at DESC",
            (username_filter,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT file_id, filename, uploaded_by, uploaded_at FROM library ORDER BY uploaded_at DESC"
        ).fetchall()
    conn.close()
    return rows

# ----------------------------- Sidebar Config -----------------------------
st.sidebar.divider()
st.sidebar.markdown("### Settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
vector_store_id = st.sidebar.text_input("OPENAI_VECTOR_STORE_ID", value=os.getenv("OPENAI_VECTOR_STORE_ID", ""))
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
show_raw = st.sidebar.checkbox("Show raw response", value=False)
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id:
    os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

# ----------------------------- Header -----------------------------
st.markdown('<div style="display:flex;gap:.6rem;align-items:center;"><span style="font-size:1.6rem">📄</span><h1 style="margin:0">PDF Q&A</h1></div>', unsafe_allow_html=True)
st.caption("Upload → Ingest → Search (Vector Store & Structured DB) • Signed in as **{}**".format(username))

# ----------------------------- Tabs -----------------------------
tab_upload, tab_ask, tab_all, tab_files, tab_library = st.tabs(
    ["⬆️ Upload & Inspect", "💬 Ask", "🗂️ Summarize ALL", "📁 Files", "📚 Library"]
)

# ================== Upload & Inspect ==================
with tab_upload:
    st.markdown('<div class="card"><h4 class="compact">Upload unstructured content</h4><div>PDF/Markdown → Vector Store + Structured DB</div></div>', unsafe_allow_html=True)
    uf = st.file_uploader("Drop a PDF or Markdown", type=["pdf", "md", "txt"])
    if uf is not None:
        up_dir = Path("Uploads")
        up_dir.mkdir(exist_ok=True)
        save_path = up_dir / uf.name
        with open(save_path, "wb") as f:
            f.write(uf.read())
        st.success(f"Saved: {save_path}")

        c1, c2 = st.columns(2)
        if c1.button("Upload to Vector Store"):
            try:
                client = get_client()
                vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
                if not vs_id:
                    st.error("Set OPENAI_VECTOR_STORE_ID in sidebar or secrets.")
                    st.stop()
                with open(save_path, "rb") as fh:
                    up = client.files.create(file=fh, purpose="assistants")
                client.vector_stores.file_batches.create(vector_store_id=vs_id, file_ids=[up.id])
                # Track in library
                upsert_file_meta(up.id, uf.name, username)
                st.success("Queued for indexing in Vector Store and added to Library.")
            except Exception as e:
                st.error(f"Vector upload failed: {e}")

        if c2.button("Parse → JSONL → SQLite"):
            try:
                out_dir = Path("json_chunks")
                out_dir.mkdir(exist_ok=True)
                os.system(f'python3 parse_papers_to_json.py --input "{save_path}" --output "{out_dir}" --max-tokens 800 --overlap 120')
                os.system('python3 build_jsonl_sqlite.py --input "./json_chunks" --db papers.db')
                st.success("Parsed and loaded into SQLite.")
            except Exception as e:
                st.error(f"Structured pipeline failed: {e}")

# ================== Ask ==================
with tab_ask:
    with st.expander("Advanced (system prompt)", expanded=False):
        system_text = st.text_area("System", value=DEFAULT_SYSTEM, height=200, label_visibility="collapsed")
    question = st.text_input("Your question", placeholder="e.g., In ChenEtAl2020.pdf, which region was imaged and by what method?")
    if st.button("Ask", type="primary", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Provide OPENAI_API_KEY.")
                st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"):
                st.error("Provide OPENAI_VECTOR_STORE_ID.")
                st.stop()
            client = get_client()
            with st.spinner("Searching & answering..."):
                try:
                    resp, _ = ask_with_responses(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = getattr(resp, "output_text", None) or "(no text)"
                    file_ids = extract_file_ids_from_responses(resp)
                    raw_obj = resp
                except Exception:
                    resp, _, file_ids = ask_with_assistants(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = resp["output_text"]
                    raw_obj = resp["_raw"]
            st.markdown('<div class="answer-card"><b>Answer</b></div>', unsafe_allow_html=True)
            st.write(answer_text)
            st.markdown('<div class="card"><b>Sources</b></div>', unsafe_allow_html=True)
            render_sources(client, file_ids)
            if show_raw:
                st.markdown("**Raw**")
                try:
                    st.write(raw_obj.model_dump())
                except Exception:
                    st.write(str(raw_obj))
        except Exception as e:
            st.error(f"Error: {e}")

# ================== Summarize ALL ==================
with tab_all:
    st.markdown('<div class="card"><h4 class="compact">Summarize every file</h4><div>Runs a focused question per PDF.</div></div>', unsafe_allow_html=True)
    if st.button("Summarize ALL files", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Provide OPENAI_API_KEY.")
                st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"):
                st.error("Provide OPENAI_VECTOR_STORE_ID.")
                st.stop()
            client = get_client()
            vs_id = os.environ["OPENAI_VECTOR_STORE_ID"]
            files = list_vs_files(client, vs_id)
            if not files:
                st.warning("No files in Vector Store.")
                st.stop()
            progress = st.progress(0.0)
            total = len(files)
            results = []
            for i, (fid, fname) in enumerate(files, start=1):
                per_q = (f"Summarize **{fname}** in 2–3 sentences. "
                         f"Use only content from {fname}. Include one short quote and end with "
                         f"[{fname} p.— §—].")
                with st.spinner(f"Summarizing {fname} ({i}/{total})..."):
                    try:
                        try:
                            resp, _ = ask_with_responses(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                            text = getattr(resp, "output_text", None) or "(no text)"
                            fids = extract_file_ids_from_responses(resp)
                        except Exception:
                            resp, _, fids = ask_with_assistants(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                            text = resp["output_text"]
                        st.markdown(f"**{fname}**")
                        st.write(text)
                        render_sources(client, fids)
                        st.markdown("---")
                        results.append(f"### {fname}\n\n{text}\n")
                    except Exception as e:
                        st.error(f"{fname}: {e}")
                progress.progress(i / total)
            if results:
                all_md = "# Summaries\n\n" + "\n\n".join(results)
                st.download_button("Download summaries (Markdown)", data=all_md.encode("utf-8"),
                                   file_name="summaries.md", mime="text/markdown")
            st.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")

# ================== Files ==================
with tab_files:
    st.markdown('<div class="card"><h4 class="compact">Files in your Vector Store</h4></div>', unsafe_allow_html=True)
    if st.button("Refresh file list"):
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Provide OPENAI_API_KEY.")
                st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"):
                st.error("Provide OPENAI_VECTOR_STORE_ID.")
                st.stop()
            client = get_client()
            files = list_vs_files(client, os.environ["OPENAI_VECTOR_STORE_ID"])
            if not files:
                st.warning("No files found.")
            else:
                for _, fname in files:
                    st.markdown(f"- {fname}")
        except Exception as e:
            st.error(f"Error: {e}")

# ================== Library ==================
with tab_library:
    st.markdown('<div class="card"><h4 class="compact">Library</h4><div>Browse uploads by user and time</div></div>', unsafe_allow_html=True)
    colA, colB = st.columns([1, 3])
    with colA:
        who = st.text_input("Filter by username (optional)", value="")
    rows = query_library(who.strip() or None)
    if not rows:
        st.info("No uploads recorded yet.")
    else:
        st.write(f"**{len(rows)}** file(s).")
        for file_id, filename, uploaded_by, uploaded_at in rows:
            st.markdown(
                f"- **{filename}**  \n"
                f"  Uploaded by **{uploaded_by}** at `{uploaded_at}`  \n"
                f"  File ID: `{file_id}`"
            )
