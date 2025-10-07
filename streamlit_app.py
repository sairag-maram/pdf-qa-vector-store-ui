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
st.caption("Asks your question against your OpenAI Vector Store and returns a one-sentence, quoted, fully-cited answer.")

system_text = st.text_area("System instructions (optional override)", value=DEFAULT_SYSTEM, height=220)
question = st.text_input("Ask a question about your PDFs")
ask = st.button("Ask")

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
    """For Assistants API messages."""
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

# ---- Responses primary, Assistants fallback ----
def ask_with_responses(client: OpenAI, model: str, vs_id: str, system: str, userq: str):
    """
    Try multiple known argument shapes for Responses+FileSearch.
    Returns (resp_obj, 'responses') on success, raises otherwise.
    """
    # Newer shape (might not exist in your runtime)
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
    except Exception as e1:
        # Older: tool_resources kwarg
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
        except Exception as e2:
            # Very old: stuff it into extra_body
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
    """
    Use Assistants API (beta) as a compatibility fallback.
    """
    # Create or reuse a tiny ephemeral assistant
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

# ---------------- Action ----------------
if ask:
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

    try:
        with st.spinner("Thinking..."):
            # Try Responses API first (with several shapes)
            try:
                resp, mode = ask_with_responses(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                answer_text = getattr(resp, "output_text", None) or "(no text)"
                file_ids = extract_file_ids_from_responses(resp)
                raw_obj = resp
            except Exception:
                # Fallback to Assistants API
                resp, mode, file_ids = ask_with_assistants(client, model, os.environ["OPENAI_VECTOR_STORE_ID"], system_text, question)
                answer_text = resp["output_text"]
                raw_obj = resp["_raw"]

        st.markdown(f"### Answer\n{answer_text}")

        # Basic Sources section (filenames from citation IDs)
        if file_ids:
            st.markdown("### Sources")
            for fid in file_ids[:5]:
                try:
                    f = client.files.retrieve(fid)
                    st.write(f"- {getattr(f, 'filename', fid) or fid}")
                except Exception:
                    st.write(f"- {fid}")
        else:
            st.info("No file citations returned (or none detected).")

        if show_raw:
            st.markdown("### Raw response")
            try:
                st.write(raw_obj.model_dump())
            except Exception:
                st.write(str(raw_obj))

    except Exception as e:
        st.error(f"Error: {e}")
