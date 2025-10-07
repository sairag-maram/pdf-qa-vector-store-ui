#!/usr/bin/env python3
import os
import streamlit as st
from pathlib import Path
from typing import List, Set

from dotenv import load_dotenv
from openai import OpenAI

# ---------- setup ----------
st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")

# pull from Streamlit Secrets (cloud) if present
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "OPENAI_VECTOR_STORE_ID" in st.secrets:
    os.environ["OPENAI_VECTOR_STORE_ID"] = st.secrets["OPENAI_VECTOR_STORE_ID"]

# load .env for local dev
load_dotenv(Path(__file__).with_name(".env"))

def get_client():
    if not os.getenv("OPENAI_API_KEY"):
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

# ---------- sidebar ----------
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
vector_store_id = st.sidebar.text_input("OPENAI_VECTOR_STORE_ID", value=os.getenv("OPENAI_VECTOR_STORE_ID", ""))
model = st.sidebar.text_input("Model", value="gpt-4o-mini")
show_raw = st.sidebar.checkbox("Show raw response", value=False)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if vector_store_id:
    os.environ["OPENAI_VECTOR_STORE_ID"] = vector_store_id

st.title("📄 PDF Q&A (Vector Store)")
st.caption("Asks your question against your OpenAI Vector Store and returns a one-sentence, quoted, fully-cited answer.")

system_text = st.text_area("System instructions (optional override)", value=DEFAULT_SYSTEM, height=220)
question = st.text_input("Ask a question about your PDFs")
ask = st.button("Ask")

def extract_file_ids(resp) -> List[str]:
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
            resp = client.responses.create(
                model=model,
                tools=[{"type": "file_search"}],
                tool_resources={"file_search": {"vector_store_ids": [os.environ["OPENAI_VECTOR_STORE_ID"]]}},
                input=[
                    {"role": "system", "content": system_text.strip()},
                    {"role": "user", "content": question.strip()},
                ],
            )

        st.markdown(f"### Answer\n{resp.output_text or '(no text)'}")

        file_ids = extract_file_ids(resp)
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
            st.write(resp.model_dump())

    except Exception as e:
        st.error(f"Error: {e}")
