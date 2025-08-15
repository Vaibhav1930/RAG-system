import streamlit as st
from rag_pipeline import RAG

st.set_page_config(page_title="Scientific RAG", layout="wide")
st.title("🔬 Domain-Specific Scientific RAG")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K passages", 3, 10, 5)
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        help="Flash = faster, more free quota. Pro = deeper reasoning but lower quota."
    )
    st.markdown("Upload papers into `data/papers/` then run `python ingest.py`.")

# Initialize or update RAG in session state dynamically
if (
    "rag" not in st.session_state
    or st.session_state["top_k"] != top_k
    or st.session_state["model_choice"] != model_choice
):
    st.session_state["rag"] = RAG(top_k=top_k, model_name=model_choice)
    st.session_state["top_k"] = top_k
    st.session_state["model_choice"] = model_choice

# Question input
q = st.text_input("Ask a technical question (equations are okay, e.g., E=mc^2):")

if st.button("Ask") and q.strip():
    with st.spinner("Thinking…"):
        answer, citations, hits = st.session_state["rag"].answer(q)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources"):
        st.code(citations)

    with st.expander("Retrieved Context"):
        for doc, meta, dist in hits:
            st.markdown(
                f"**{meta['doc_id']}** • {meta['chunk_index']} • d={dist:.3f} • {meta['source']}"
            )
            st.write(doc)
            st.markdown("---")
