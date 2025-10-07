#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import List, Set, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads import Run

# ---------------- Base setup ----------------
st.set_page_config(
    page_title="PDF Q&A • Vector Store",
    page_icon="📄",
    layout="centered",
    menu_items={"Get help": None, "Report a bug": None, "About": "PDF Q&A on OpenAI Vector Stores"},
)

# --- Stanford color theme (Cardinal / Bright Red / Cool Gray)
st.markdown("""
<style>
:root {
  --card-bg: rgba(140,21,21,0.05);     /* Cardinal tint */
  --chip-bg: rgba(177,4,14,0.10);      /* Bright red tint */
  --border: rgba(140,21,21,0.35);      /* Cardinal border */
  --muted: #4D4F53;                    /* Stanford Cool Gray */
  --accent: #8C1515;                   /* Cardinal */
  --accent-2: #B1040E;                 /* Bright Red */
}

/* subtle page background with Cardinal haze */
html, body, .stApp {
  background: linear-gradient(180deg, rgba(140,21,21,.03), transparent 25%) !important;
}

/* header + kicker */
div.app-header{display:flex;gap:.75rem;align-items:center;margin:0 0 .5rem 0}
div.app-header h1{font-weight:800;margin:0;color:var(--accent)}
div.kicker{color:var(--muted);font-size:.95rem;margin:-.25rem 0 .5rem 0}

/* cards */
div.card{border:1px solid var(--border);background:var(--card-bg);padding:16px 18px;border-radius:16px;box-shadow:0 4px 18px rgba(140,21,21,.10)}
div.answer-card{border:1px solid var(--border);background:var(--card-bg);padding:16px 18px;border-radius:16px;margin:.5rem 0 1rem 0;box-shadow:0 4px 18px rgba(140,21,21,.10)}
h4.compact{margin:.1rem 0 .6rem 0}
.small{color:var(--muted);font-size:.9rem}

/* chips (source tags) */
span.chip{display:inline-block;padding:4px 10px;border-radius:999px;background:var(--chip-bg);margin-right:6px;font-size:.85rem;color:var(--accent)}

/* soft divider */
hr.soft{border:none;height:1px;background:var(--border);margin:.65rem 0}

/* primary actions: buttons + download buttons */
div.stButton > button, .stDownloadButton > button{
  background: var(--accent); color:#fff; border:1px solid var(--accent-2);
  border-radius:12px; padding:.55rem .9rem; font-weight:600;
}
div.stButton > button:hover, .stDownloadButton > button:hover{
  background: var(--accent-2); border-color: var(--accent-2);
}

/* tabs with Cardinal underlines */
.stTabs [data-baseweb="tab"]{
  color: var(--muted); border-bottom:2px solid transparent; padding-bottom:.35rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  color: var(--accent); border-bottom-color: var(--accent); font-weight:700;
}

/* progress bar in Cardinal */
[data-testid="stProgressBar"] > div > div > div{ background: var(--accent) !important; }

/* link color */
a{color:var(--accent-2)}
</style>
""", unsafe_allow_html=True)

# Pull from Streamlit Secrets (cloud) if present
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "OPENAI_VECTOR_STORE_ID" in st.secrets:
    os.environ["OPENAI_VECTOR_STORE_ID"] = st.secrets["OPENAI_VECTOR_STORE_ID"]

# Load .env for local dev
load_dotenv(Path(__file__).with_name(".env"))

def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY (set in .env or Streamlit Secrets).")
    return OpenAI()

DEFAULT_SYSTEM = """You are a retrieval-first assistant for scientific PDFs attached via File Search.

QUERY REFORMULATION (before searching)
- Rewrite the user’s question into 1–4 compact sub-queries with synonyms, expanded acronyms, and constraints (species, region, methods, metrics). Do not show these to the user.

RETRIEVAL
- Always call File Search with your sub-queries; if recall is weak, retry once more broadly.
- Treat retrieved passages as ground truth.

ANSWER FORMAT & CITATION RULES
- Return EXACTLY ONE sentence; concise and precise.
- Include one short verbatim quote from the passage in double quotes (“like this”).
- End the sentence with one citation tag in the form [<filename> p.<page> §<section>].
  * Use the filename reported by File Search (no path).
  * If page/section are unknown, use p.— and §—.
- Never invent page/section/quotes; if nothing relevant is found after the broadened retry, output exactly:
  No direct evidence found in the provided files.
"""

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
vector_store_id = st.sidebar.text_input("OPENAI_VECTOR_STORE_ID", value=os.getenv("OPENAI_VECTOR_STORE_ID", ""))
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
show_raw = st.sidebar.checkbox("Show raw response", value=False)

# Version chip
try:
    import openai as _openai_pkg
    st.sidebar.caption(f"openai version: {_openai_pkg.__version__}")
except Exception:
    pass

if api_key: os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id: os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

# ---------------- Header ----------------
st.markdown('<div class="app-header"><span style="font-size:1.75rem">📄</span><h1>PDF Q&A</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="kicker">Ask questions against your OpenAI Vector Store, or summarize every file. Answers include a short quote and sources.</div>', unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab_ask, tab_all, tab_files = st.tabs(["💬 Ask", "🗂️ Summarize ALL", "📁 Files"])

# ---------------- Helpers (citations, files, dual-API logic) ----------------
def extract_file_ids_from_responses(resp) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for block in resp.output:
            if block.type == "message":
                for c in (block.message.content or []):
                    if getattr(c, "type", None) == "output_text":
                        for ann in (getattr(c, "annotations", []) or []):
                            if getattr(ann, "type", None) == "file_citation":
                                fc = getattr(ann, "file_citation", None)
                                fid = getattr(fc, "file_id", None) if fc else None
                                if fid: file_ids.add(fid)
    except Exception:
        pass
    return list(file_ids)

def extract_file_ids_from_messages(messages_list) -> List[str]:
    file_ids: Set[str] = set()
    try:
        for msg in messages_list:
            if getattr(msg, "role", "") != "assistant": continue
            for item in (msg.content or []):
                if getattr(item, "type", None) == "text":
                    for ann in (getattr(item.text, "annotations", []) or []):
                        if getattr(ann, "type", None) == "file_citation":
                            fid = getattr(ann.file_citation, "file_id", None)
                            if fid: file_ids.add(fid)
    except Exception:
        pass
    return list(file_ids)

def list_vs_files(client: OpenAI, vector_store_id: str) -> List[Tuple[str, str]]:
    after = None; files = []
    while True:
        page = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100, after=after)
        files.extend(page.data)
        if not page.has_more: break
        after = page.last_id
    out: List[Tuple[str, str]] = []
    for it in files:
        try:
            f = client.files.retrieve(it.id)
            out.append((it.id, f.filename or it.id))
        except Exception:
            out.append((it.id, it.id))
    return out

def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    # 1) Newest: file_search top-level
    try:
        return client.responses.create(
            model=model,
            input=[{"role": "system", "content": system.strip()},
                   {"role": "user", "content": userq.strip()}],
            tools=[{"type": "file_search"}],
            file_search={"vector_store_ids": [vs_id]},
        ), "responses"
    except Exception:
        pass
    # 2) Mid: tool_resources kwarg
    try:
        return client.responses.create(
            model=model,
            input=[{"role": "system", "content": system.strip()},
                   {"role": "user", "content": userq.strip()}],
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
        ), "responses"
    except Exception:
        pass
    # 3) Oldest: extra_body
    return client.responses.create(
        model=model,
        input=[{"role": "system", "content": system.strip()},
               {"role": "user", "content": userq.strip()}],
        tools=[{"type": "file_search"}],
        extra_body={"tool_resources": {"file_search": {"vector_store_ids": [vs_id]}}},
    ), "responses"

def ask_with_assistants(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    asst = client.beta.assistants.create(
        name="Streamlit PDF QA (temp)",
        model=model,
        instructions=system.strip(),
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
    )
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role="user", content=userq.strip())
    run: Run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=asst.id)

    deadline = time.time() + 60 * 6
    sleep_s = 0.75
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in {"completed", "failed", "cancelled", "expired"}: break
        if time.time() > deadline: raise RuntimeError("Run timed out.")
        time.sleep(sleep_s); sleep_s = min(sleep_s * 1.5, 6.0)

    if run.status != "completed": raise RuntimeError(f"Run did not complete: {run.status}")

    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5)
    answer_text = ""
    for m in msgs.data:
        if m.role != "assistant": continue
        for item in m.content:
            if item.type == "text":
                answer_text = item.text.value or ""
                break
        if answer_text: break

    file_ids = extract_file_ids_from_messages(msgs.data)
    return {"output_text": answer_text, "_raw": {"messages": msgs, "assistant_id": asst.id, "thread_id": thread.id}}, "assistants", file_ids

def render_sources(client: OpenAI, file_ids: List[str]):
    if not file_ids:
        st.markdown('<span class="small">No file citations returned (or none detected).</span>', unsafe_allow_html=True)
        return
    for fid in file_ids[:10]:
        try:
            f = client.files.retrieve(fid)
            st.markdown(f'<span class="chip">{getattr(f, "filename", fid) or fid}</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown(f'<span class="chip">{fid}</span>', unsafe_allow_html=True)

# ---------------- Tab: Ask ----------------
with tab_ask:
    with st.expander("Advanced (system prompt)", expanded=False):
        system_text = st.text_area("System", value=DEFAULT_SYSTEM, height=200, label_visibility="collapsed")
    question = st.text_input("Your question", placeholder="e.g., In ChenEtAl2020.pdf, which region was imaged and by what method?")
    if st.button("Ask", type="primary", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Provide your OpenAI API key in the sidebar."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"):
                st.error("Provide your Vector Store ID in the sidebar."); st.stop()
            if not question.strip():
                st.error("Please enter a question."); st.stop()

            client = get_client()
            with st.spinner("Thinking..."):
                try:
                    resp, _ = ask_with_responses(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = getattr(resp, "output_text", None) or "(no text)"
                    file_ids = extract_file_ids_from_responses(resp)
                    raw_obj = resp
                except Exception:
                    resp, _, file_ids = ask_with_assistants(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                    answer_text = resp["output_text"]; raw_obj = resp["_raw"]

            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.markdown("**Answer**")
            st.write(answer_text)
            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
            st.markdown('<span class="small">Sources</span>', unsafe_allow_html=True)
            render_sources(client, file_ids)
            st.markdown('</div>', unsafe_allow_html=True)

            if show_raw:
                st.markdown("**Raw**")
                try: st.write(raw_obj.model_dump())
                except Exception: st.write(str(raw_obj))

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Tab: Summarize ALL ----------------
with tab_all:
    st.markdown('<div class="card"><h4 class="compact">Summarize every file</h4><div class="small">Runs one focused question per PDF to guarantee coverage.</div></div>', unsafe_allow_html=True)
    if st.button("Summarize ALL files (one by one)", use_container_width=True):
        try:
            if not os.getenv("OPENAI_API_KEY"): st.error("Provide your OpenAI API key in the sidebar."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"): st.error("Provide your Vector Store ID in the sidebar."); st.stop()
            client = get_client()
            vs_id = os.environ["OPENAI_VECTOR_STORE_ID"]
            files = list_vs_files(client, vs_id)

            if not files:
                st.warning("No files found in this vector store.")
            else:
                progress = st.progress(0.0)
                total = len(files)
                results_md = []
                for idx, (fid, fname) in enumerate(files, start=1):
                    per_q = (
                        f"Summarize the paper **{fname}** in 2–3 sentences. "
                        f"Use only content from {fname}. Include one short quote and end with "
                        f"[{fname} p.— §—]."
                    )
                    with st.spinner(f"Summarizing {fname} ({idx}/{total})..."):
                        try:
                            try:
                                resp, _ = ask_with_responses(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                                answer_text = getattr(resp, "output_text", None) or "(no text)"
                                file_ids = extract_file_ids_from_responses(resp)
                            except Exception:
                                resp, _, file_ids = ask_with_assistants(client, model, vs_id, DEFAULT_SYSTEM, per_q)
                                answer_text = resp["output_text"]

                            st.markdown(f'**{fname}**')
                            st.write(answer_text)
                            render_sources(client, file_ids)
                            st.markdown('<hr class="soft"/>', unsafe_allow_html=True)
                            results_md.append(f"### {fname}\n\n{answer_text}\n")
                        except Exception as inner_e:
                            st.error(f"{fname}: {inner_e}")
                    progress.progress(idx / total)

                if results_md:
                    all_md = "# Summaries\n\n" + "\n\n".join(results_md)
                    st.download_button("Download all summaries (Markdown)", data=all_md.encode("utf-8"),
                                       file_name="summaries.md", mime="text/markdown")
                st.success("Done.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Tab: Files ----------------
with tab_files:
    st.markdown('<div class="card"><h4 class="compact">Files in your Vector Store</h4></div>', unsafe_allow_html=True)
    if st.button("Refresh file list"):
        try:
            if not os.getenv("OPENAI_API_KEY"): st.error("Provide your OpenAI API key in the sidebar."); st.stop()
            if not os.getenv("OPENAI_VECTOR_STORE_ID"): st.error("Provide your Vector Store ID in the sidebar."); st.stop()
            client = get_client()
            files = list_vs_files(client, os.environ["OPENAI_VECTOR_STORE_ID"])
            if not files:
                st.warning("No files found.")
            else:
                for _, fname in files: st.markdown(f"- {fname}")
        except Exception as e:
            st.error(f"Error: {e}")
