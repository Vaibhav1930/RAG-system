import patch_sqlite  
import os
import hashlib
from pathlib import Path
import chromadb
from typing import List, Dict
from rapidfuzz import fuzz
import groq
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.environ.get("PERSIST_DATA_DIR", "data")
DB_DIR = os.path.join(PERSIST_DIR, "vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- Retriever ---------------- #
class Retriever:
    def __init__(self, user_id="default", top_k=5):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.user_id = user_id
        self.top_k = top_k

    @property
    def embedder(self):
        if not hasattr(self, "_embedder"):
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    @property
    def col(self):
        # Dynamically fetch/create the collection to ensure it exists
        safe_user_id = "".join([c if c.isalnum() or c in "-_" else "_" for c in self.user_id])
        if len(safe_user_id) < 3:
            safe_user_id = safe_user_id.ljust(3, "0")
        collection_name = f"user_{safe_user_id}"[:63]
        return self.client.get_or_create_collection(collection_name)

    def query(self, q: str):
        if self.col.count() == 0:
            return []
        q_emb = self.embedder.encode([q], normalize_embeddings=True)
        # Avoid asking for more results than items in the collection
        n_results = min(self.top_k, self.col.count())
        res = self.col.query(
            query_embeddings=q_emb,
            n_results=n_results,
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


# ---------------- RAG with Groq + Auto-Fallback ---------------- #
class RAG:
    def __init__(self, user_id="default", top_k=5, model_name="llama-3.3-70b-versatile"):
        self.retriever = Retriever(user_id=user_id, top_k=top_k)

        # Load API key from .env
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables or .env file")

        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name

    def _generate_with_fallback(self, question, context):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations."}
                ],
                model=self.model_name,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"⚠️ Rate limit hit for {self.model_name}, switching to llama-3.1-8b-instant")
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations."}
                    ],
                    model="llama-3.1-8b-instant",
                )
                return response.choices[0].message.content
            raise

    def answer(self, question: str):
        hits = self.retriever.query(question)

        # Build context
        context_blocks = []
        for i, (doc, meta, dist) in enumerate(hits):
            header = f"[{meta['doc_id']}] chunk#{meta['chunk_index']} (d={dist:.3f}, {os.path.basename(meta['source'])})"
            context_blocks.append(header + "\n" + doc)
        context = "\n\n---\n\n".join(context_blocks)

        # Generate answer with fallback
        answer = self._generate_with_fallback(question, context)

        answer = answer if answer else "(No response generated)"
        citations = format_citations(hits)
        return answer, citations, hits


# ---------------- Ingestion & Deletion Helpers ---------------- #
from utils.parsing import parse_pdf, chunk_equation_aware

def file_id(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]

def ingest_pdf_to_user(pdf_path: Path, user_id: str) -> int:
    retriever = Retriever(user_id=user_id)
    fid = file_id(pdf_path)
    meta_common = {"source": str(pdf_path).replace("\\", "/"), "doc_id": fid}

    # Parse and chunk PDF
    sections = parse_pdf(str(pdf_path))
    all_chunks = []
    for s in sections:
        all_chunks.extend(chunk_equation_aware(s, target=900, overlap=150))

    if not all_chunks:
        return 0

    # Create embeddings
    embeddings = retriever.embedder.encode(
        all_chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Create IDs and metadata for each chunk
    ids = [f"{fid}_{i}" for i in range(len(all_chunks))]
    metas = [{**meta_common, "chunk_index": i} for i in range(len(all_chunks))]

    # Add to user's Chroma collection
    retriever.col.add(
        documents=all_chunks,
        metadatas=metas,
        embeddings=embeddings,
        ids=ids
    )
    return len(all_chunks)

def delete_user_pdf(pdf_path: Path, user_id: str):
    retriever = Retriever(user_id=user_id)
    normalized_path = str(pdf_path).replace("\\", "/")
    retriever.col.delete(where={"source": normalized_path})
    if pdf_path.exists():
        pdf_path.unlink()

def clear_user_data(user_id: str):
    retriever = Retriever(user_id=user_id)
    safe_user_id = "".join([c if c.isalnum() or c in "-_" else "_" for c in user_id])
    if len(safe_user_id) < 3:
        safe_user_id = safe_user_id.ljust(3, "0")
    collection_name = f"user_{safe_user_id}"[:63]
    try:
        retriever.client.delete_collection(collection_name)
    except Exception:
        pass
    
    # Delete uploaded files on disk
    user_dir = Path("data") / "users" / user_id / "papers"
    if user_dir.exists():
        for file in user_dir.glob("*.pdf"):
            try:
                file.unlink()
            except Exception:
                pass
