import os
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- Retriever ---------------- #
class Retriever:
    def __init__(self, top_k=5):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.col = self.client.get_collection("scientific")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.top_k = top_k

    def query(self, q: str):
        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        res = self.col.query(
            query_embeddings=q_emb,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return list(zip(docs, metas, dists))


# ---------------- Citations ---------------- #
def format_citations(hits: List[Dict]):
    seen = {}
    out = []
    for _, m, _ in hits:
        src = os.path.basename(m["source"])
        key = f'{m["doc_id"]}'
        seen.setdefault(key, src)
    for k, v in seen.items():
        out.append(f"[{k}] {v}")
    return "\n".join(out)


# ---------------- System Prompt ---------------- #
SYSTEM_PROMPT = """You are a domain-specific scientific assistant.
- Use ONLY the provided context to answer.
- Include short IEEE-style inline citations like [doc_id].
- If unsure or missing context, say so briefly.
- Preserve equations exactly as shown (e.g., $E=mc^2$).
"""


# ---------------- RAG with Gemini + Auto-Fallback ---------------- #
class RAG:
    def __init__(self, top_k=5, model_name="gemini-1.5-pro"):
        self.retriever = Retriever(top_k=top_k)

        # Load API key from .env
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables or .env file")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def _generate_with_fallback(self, prompt):
        try:
            return self.model.generate_content(prompt)
        except Exception as e:
            if "quota" in str(e).lower() or "ResourceExhausted" in str(e):
                print("⚠️ Quota hit for gemini-1.5-pro, switching to gemini-1.5-flash")
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                return self.model.generate_content(prompt)
            raise

    def answer(self, question: str):
        hits = self.retriever.query(question)

        # Build context
        context_blocks = []
        for i, (doc, meta, dist) in enumerate(hits):
            header = f"[{meta['doc_id']}] chunk#{meta['chunk_index']} (d={dist:.3f}, {os.path.basename(meta['source'])})"
            context_blocks.append(header + "\n" + doc)
        context = "\n\n---\n\n".join(context_blocks)

        # Create prompt
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer with citations."

        # Generate answer with fallback
        resp = self._generate_with_fallback(prompt)

        answer = resp.text if hasattr(resp, "text") and resp.text else "(No response generated)"
        citations = format_citations(hits)
        return answer, citations, hits
