#!/usr/bin/env python3
import os
import time
from pathlib import Path
from typing import List, Set, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.beta.threads import Run

# ---------------- Setup ----------------
st.set_page_config(page_title="PDF Q&A (Vector Store)", page_icon="📄", layout="centered")

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

# Show SDK version (helps debugging)
try:
    import openai as _openai_pkg
    st.sidebar.caption(f"openai version: {_openai_pkg.__version__}")
except Exception:
    pass

# Update env if user types new values
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id:
    os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

# ---------------- Main UI ----------------
st.title("📄 PDF Q&A (Vector Store)")
st.caption("Ask about your PDFs or summarize ALL files. Answers include a quote and sources.")

system_text = st.text_area("System instructions (optional override)", value=DEFAULT_SYSTEM, height=220)
question = st.text_input("Ask a question about your PDFs")
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    ask = st.button("Ask")
with colB:
    list_files_btn = st.button("List files in vector store")
with colC:
    summarize_all = st.button("Summarize ALL files (one by one)")

# ---------------- Helpers ----------------
def extract_file_ids_from_responses(resp) -> List[str]:
    """Collect file IDs from file_citation annotations in a Responses result."""
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
                                if fid:
                                    file_ids.add(fid)
    except Exception:
        pass
    return list(file_ids)

def extract_file_ids_from_messages(messages_list) -> List[str]:
    """For Assistants API messages: pull file citations."""
    file_ids: Set[str] = set()
    try:
        for msg in messages_list:
            if getattr(msg, "role", "") != "assistant":
                continue
            for item in (msg.content or []):
                if getattr(item, "type", None) == "text":
                    for ann in (getattr(item.text, "annotations", []) or []):
                        if getattr(ann, "type", None) == "file_citation":
                            fid = getattr(ann.file_citation, "file_id", None)
                            if fid:
                                file_ids.add(fid)
    except Exception:
        pass
    return list(file_ids)

def list_vs_files(client: OpenAI, vector_store_id: str) -> List[Tuple[str, str]]:
    """Return list of (file_id, filename) for all files in a vector store."""
    after = None
    files = []
    while True:
        page = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100, after=after)
        files.extend(page.data)
        if not page.has_more:
            break
        after = page.last_id
    out: List[Tuple[str, str]] = []
    for it in files:
        try:
            f = client.files.retrieve(it.id)
            out.append((it.id, f.filename or it.id))
        except Exception:
            out.append((it.id, it.id))
    return out

# ---- Responses primary (multiple shapes), Assistants fallback ----
def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    """Try several known shapes for Responses+FileSearch. Return (resp_obj, 'responses')."""
    # Newer: file_search top-level
    try:
        return (
            client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": userq.strip()},
                ],
                tools=[{"type": "file_search"}],
                file_search={"vector_store_ids": [vs_id]},
            ),
            "responses",
        )
    except Exception:
        pass

    # Mid: tool_resources kwarg
    try:
        return (
            client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": userq.strip()},
                ],
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [vs_id]}},
            ),
            "responses",
        )
    except Exception:
        pass

    # Oldest: extra_body
    return (
        client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": userq.strip()},
            ],
            tools=[{"type": "file_search"}],
            extra_body={"tool_resources": {"file_search": {"vector_store_ids": [vs_id]}}},
        ),
        "responses",
    )

def ask_with_assistants(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    """Compatibility fallback using Assistants API (beta)."""
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

    # Poll
    deadline = time.time() + 60 * 6
    sleep_s = 0.75
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in {"completed", "failed", "cancelled", "expired"}:
            break
        if time.time() > deadline:
            raise RuntimeError("Run timed out.")
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 1.5, 6.0)

    if run.status != "completed":
        raise RuntimeError(f"Run did not complete: {run.status}")

    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5)
    answer_text = ""
    for m in msgs.data:
        if m.role != "assistant":
            continue
        for item in m.content:
            if item.type == "text":
                answer_text = item.text.value or ""
                break
        if answer_text:
            break

    file_ids = extract_file_ids_from_messages(msgs.data)
    return {"output_text": answer_text, "_raw": {"messages": msgs, "assistant_id": asst.id, "thread_id": thread.id}}, "assistants", file_ids

def render_sources(client: OpenAI, file_ids: List[str]):
    if not file_ids:
        st.info("No file citations returned (or none detected).")
        return
    st.markdown("**Sources**")
    for fid in file_ids[:10]:
        try:
            f = client.files.retrieve(fid)
            st.write(f"- {getattr(f, 'filename', fid) or fid}")
        except Exception:
            st.write(f"- {fid}")

# ---------------- Actions ----------------
if list_files_btn:
    try:
        client = get_client()
        vs_id = os.environ.get("OPENAI_VECTOR_STORE_ID", "")
        if not vs_id:
            st.error("Please provide your Vector Store ID.")
        else:
            files = list_vs_files(client, vs_id)
            if not files:
                st.warning("No files found in this vector store.")
            else:
                st.markdown("### Files in vector store")
                for _, fname in files:
                    st.write(f"- {fname}")
    except Exception as e:
        st.error(f"Error: {e}")

if ask:
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API key.")
            st.stop()
        if not os.getenv("OPENAI_VECTOR_STORE_ID"):
            st.error("Please provide your Vector Store ID.")
            st.stop()
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        client = get_client()

        with st.spinner("Thinking..."):
            # Try Responses first; if it fails, fallback to Assistants
            try:
                resp, mode = ask_with_responses(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                answer_text = getattr(resp, "output_text", None) or "(no text)"
                file_ids = extract_file_ids_from_responses(resp)
                raw_obj = resp
            except Exception:
                resp, mode, file_ids = ask_with_assistants(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                answer_text = resp["output_text"]
                raw_obj = resp["_raw"]

        st.subheader("Answer")
        st.write(answer_text)
        st.subheader("Sources")
        render_sources(client, file_ids)

        if show_raw:
            st.markdown("### Raw")
            try:
                st.write(raw_obj.model_dump())
            except Exception:
                st.write(str(raw_obj))

    except Exception as e:
        st.error(f"Error: {e}")

if summarize_all:
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("Please provide your OpenAI API key.")
            st.stop()
        if not os.getenv("OPENAI_VECTOR_STORE_ID"):
            st.error("Please provide your Vector Store ID.")
            st.stop()

        client = get_client()
        vs_id = os.environ["OPENAI_VECTOR_STORE_ID"]
        files = list_vs_files(client, vs_id)

        if not files:
            st.warning("No files found in this vector store.")
        else:
            st.subheader("Summaries")
            progress = st.progress(0.0)
            total = len(files)
            for idx, (fid, fname) in enumerate(files, start=1):
                # Per-file prompt (nudges model to use only that file)
                per_q = (
                    f"Summarize the paper **{fname}** in 2–3 sentences. "
                    f"Use only content from {fname}. Include one short quote and end with "
                    f"[{fname} p.— §—]."
                )
                with st.spinner(f"Summarizing {fname} ({idx}/{total})..."):
                    try:
                        try:
                            resp, mode = ask_with_responses(client, model, vs_id, system_text, per_q)
                            answer_text = getattr(resp, "output_text", None) or "(no text)"
                            file_ids = extract_file_ids_from_responses(resp)
                            raw_obj = resp
                        except Exception:
                            resp, mode, file_ids = ask_with_assistants(client, model, vs_id, system_text, per_q)
                            answer_text = resp["output_text"]
                            raw_obj = resp["_raw"]

                        st.markdown(f"**{fname}**")
                        st.write(answer_text)
                        render_sources(client, file_ids)
                        st.markdown("---")
                    except Exception as inner_e:
                        st.error(f"{fname}: {inner_e}")
                progress.progress(idx / total)
            st.success("Done.")

    except Exception as e:
        st.error(f"Error: {e}")
