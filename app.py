from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings
from src.local_search import LocalSearchEngine
from src.rag import generate_answer


KB_PATH = Path("data/processed/knowledge_base.csv")


@st.cache_resource(show_spinner=False)
def load_local_engine(path: str) -> LocalSearchEngine | None:
    if not Path(path).exists():
        return None
    return LocalSearchEngine.from_csv(path)


def search_knowledge_base(query: str, category: str, top_k: int, threshold: float):
    try:
        from src.pgvector_store import semantic_search

        return semantic_search(query, top_k=top_k, category=category, threshold=threshold), "PgVector"
    except Exception as exc:
        engine = load_local_engine(str(KB_PATH))
        if engine is None:
            raise RuntimeError(
                "Knowledge base belum tersedia. Jalankan scripts/prepare_kb.py terlebih dahulu."
            ) from exc
        return engine.search(query, top_k=top_k, category=category, threshold=0.05), "Local TF-IDF"


st.set_page_config(page_title="Intelligent Customer Assistant", page_icon="chat", layout="wide")

st.title("Intelligent Customer Assistant")

with st.sidebar:
    st.subheader("Retrieval Settings")
    category = st.selectbox(
        "Category filter",
        ["all", "faq", "policy", "troubleshooting", "contact", "general"],
        index=0,
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=settings.top_k)
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.similarity_threshold),
        step=0.05,
    )

    st.divider()
    if KB_PATH.exists():
        kb = pd.read_csv(KB_PATH)
        st.metric("Knowledge entries", len(kb))
        st.caption("Local fallback ready")
    else:
        st.warning("Knowledge base CSV belum dibuat.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya siap membantu menjawab pertanyaan berdasarkan knowledge base.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tulis pertanyaan customer...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari konteks dan menyusun jawaban..."):
            (results, latency_ms), backend = search_knowledge_base(prompt, category, top_k, threshold)
            answer = generate_answer(prompt, results)
            st.markdown(answer)

        with st.expander(f"Retrieved context ({backend}, {latency_ms:.1f} ms)"):
            if not results:
                st.info("Tidak ada konteks yang melewati threshold.")
            for result in results:
                st.write(f"Score: `{result.score:.3f}` | Category: `{result.category}`")
                st.write(result.question)
                st.caption(result.answer[:600])

    st.session_state.messages.append({"role": "assistant", "content": answer})
