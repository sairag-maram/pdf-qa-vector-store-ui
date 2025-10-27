#!/usr/bin/env python3
"""
Streamlit UI:
- 🔐 Login (streamlit-authenticator)
- ⬆️ Upload & Inspect: upload PDF/MD, send to Vector Store and parse→JSONL→SQLite
- 💬 Ask: RAG over OpenAI Vector Store (Responses API; Assistants fallback)
- 🗂️ Summarize ALL: per-file summaries
- 📁 Files: list files in Vector Store
- 📚 Library: list/search files uploaded by all users (who & when)
"""

import os, time, sqlite3, re
from pathlib import Path
from typing import List, Set, Tuple, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads import Run

# ---- auth ----
import streamlit_authenticator as stauth

# ---- page setup ----
st.set_page_config(page_title="PDF Q&A • Vector Store", page_icon="📄", layout="wide")

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

# ====== Small metadata DB (for Library) ======
DB_PATH = "app.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        full_name TEXT,
        email TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS uploads(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vs_file_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        uploaded_by TEXT NOT NULL,
        uploaded_at TEXT NOT NULL
    );
    """)
    con.commit()
    con.close()

def upsert_user(username: str, full_name: str, email: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO users(username, full_name, email) VALUES(?,?,?)",
                (username, full_name, email))
    con.commit(); con.close()

def add_upload(vs_file_id: str, filename: str, uploaded_by: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO uploads(vs_file_id, filename, uploaded_by, uploaded_at) VALUES(?,?,?,?)",
                (vs_file_id, filename, uploaded_by, datetime.utcnow().isoformat()))
    con.commit(); con.close()

def query_uploads(search: str = "", only_user: Optional[str]=None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    q = "SELECT filename, uploaded_by, uploaded_at, vs_file_id FROM uploads"
    args: List[str] = []
    where = []
    if only_user:
        where.append("uploaded_by = ?")
        args.append(only_user)
    if search:
        where.append("LOWER(filename) LIKE ?")
        args.append(f"%{search.lower()}%")
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY uploaded_at DESC"
    df = pd.read_sql_query(q, con, params=args)
    con.close()
    return df

# ====== OpenAI helpers ======
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI()

DEFAULT_SYSTEM = """You are a retrieval-first assistant for scientific PDFs attached via File Search.
QUERY REFORMULATION: rewrite into 1–4 sub-queries with synonyms/acronyms.
RETRIEVAL: use File Search; retry broader once if recall is weak.
ANSWER: exactly one sentence with one short quote in “double quotes”, ending with [<filename> p.<page> §<section>] (use p.—/§— if unknown). If nothing relevant, output: No direct evidence found in the provided files.
"""

# ---- sidebar ----
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
vector_store_id = st.sidebar.text_input("OPENAI_VECTOR_STORE_ID", value=os.getenv("OPENAI_VECTOR_STORE_ID",""))
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
show_raw = st.sidebar.checkbox("Show raw response", value=False)
if api_key: os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id: os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

# ====== Auth config (secrets → fallback demo) ======
def build_authenticator():
    if "auth" in st.secrets:
        cfg = st.secrets["auth"]
        cookie_cfg = cfg.get("cookie", {"name":"vs_app_session","key":"cookie-key","expiry_days":14})
        creds = cfg.get("credentials", {"usernames":{}})
        authenticator = stauth.Authenticate(
            credentials=creds,
            cookie_name=cookie_cfg.get("name","vs_app_session"),
            key=cookie_cfg.get("key","cookie-key"),
            cookie_expiry_days=int(cookie_cfg.get("expiry_days",14)),
        )
        return authenticator, True
    # fallback demo user
    hashed = stauth.Hasher(["demo"]).generate()
    authenticator = stauth.Authenticate(
        credentials={"usernames":{
            "demo":{"name":"Demo User","email":"demo@example.com","password":hashed[0]}
        }},
        cookie_name="vs_app_session",
        key="cookie-key",
        cookie_expiry_days=14,
    )
    return authenticator, False

authenticator, using_secrets_auth = build_authenticator()
name, auth_status, username = authenticator.login("Login", "sidebar")

# ====== App body gating ======
if auth_status is False:
    st.error("Invalid username/password")
    st.stop()
elif auth_status is None:
    st.info("Please login.")
    st.stop()

# Logged in — ensure DB + user present
init_db()
# Try to derive email from secrets; else blank
email_from_secrets = ""
if using_secrets_auth:
    try:
        email_from_secrets = st.secrets["auth"]["credentials"]["usernames"][username].get("email","")
    except Exception:
        email_from_secrets = ""
upsert_user(username=username, full_name=name, email=email_from_secrets or "")

# Header with logout
col_l, col_r = st.columns([6,1])
with col_l:
    st.markdown(
        f'<div style="display:flex;gap:.6rem;align-items:center;"><span style="font-size:1.6rem">📄</span>'
        f'<h1 style="margin:0">PDF Q&A</h1></div>'
        f'<div style="color:#6b7280">Signed in as <b>{name}</b> (<code>{username}</code>)</div>',
        unsafe_allow_html=True,
    )
with col_r:
    authenticator.logout("Logout", "primary", key="logout_btn")

st.caption("Upload → Ingest → Search (Vector Store & Structured DB)")

# ---- tabs ----
tab_upload, tab_ask, tab_all, tab_files, tab_library = st.tabs(
    ["⬆️ Upload & Inspect", "💬 Ask", "🗂️ Summarize ALL", "📁 Files", "📚 Library"]
)

# ---- vector store helpers ----
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
    except Exception: pass
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
    except Exception: pass
    return list(file_ids)

def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
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

# ================== Tab: Upload & Inspect ==================
with tab_upload:
    st.markdown('<div class="card"><h4 class="compact">Upload unstructured content</h4><div>PDF/Markdown → Vector Store + Structured DB</div></div>', unsafe_allow_html=True)
    uf = st.file_uploader("Drop a PDF or Markdown", type=["pdf","md","txt"])
    if uf is not None:
        up_dir = Path("Uploads"); up_dir.mkdir(exist_ok=True)
        save_path = up_dir / uf.name
        with open(save_path, "wb") as f: f.write(uf.read())
        st.success(f"Saved: {save_path}")

        c1, c2 = st.columns(2)
        if c1.button("Upload to Vector Store", use_container_width=True):
            try:
                client = get_client()
                vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
                if not vs_id: st.error("Set OPENAI_VECTOR_STORE_ID."); st.stop()
                with open(save_path, "rb") as fh:
                    up = client.files.create(file=fh, purpose="assistants")
                client.vector_stores.file_batches.create(vector_store_id=vs_id, file_ids=[up.id])
                # record in library db
                add_upload(vs_file_id=up.id, filename=uf.name, uploaded_by=username)
                st.success("Queued for indexing in Vector Store & added to Library.")
            except Exception as e:
                st.error(f"Vector upload failed: {e}")

        if c2.button("Parse → JSONL → SQLite", use_container_width=True):
            try:
                out_dir = Path("json_chunks"); out_dir.mkdir(exist_ok=True)
                os.system(f'python3 parse_papers_to_json.py --input "{save_path}" --output "{out_dir}" --max-tokens 800 --overlap 120')
                os.system('python3 build_jsonl_sqlite.py --input "./json_chunks" --db papers.db')
                st.success("Parsed and loaded into SQLite.")
            except Exception as e:
                st.error(f"Structured pipeline failed: {e}")

# ================== Tab: Ask ==================
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
            client = get_client()
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
                st.download_button("Download summaries (Markdown)", data=all_md.encode("utf-8"),
                                   file_name="summaries.md", mime="text/markdown")
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

# ================== Tab: Library (multi-user catalog) ==================
with tab_library:
    st.markdown('<div class="card"><h4 class="compact">Team Library</h4><div>Browse files uploaded by anyone.</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3,1])
    with c1:
        search = st.text_input("Search filename", placeholder="e.g., hippocampus or ChenEtAl2020")
    with c2:
        scope = st.selectbox("Scope", ["All files", "My files"], index=0)
    only_me = username if scope == "My files" else None
    df = query_uploads(search=search, only_user=only_me)
    if df.empty:
        st.info("No matching uploads")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
