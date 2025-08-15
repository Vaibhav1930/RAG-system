import os
import hashlib
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from utils.parsing import parse_pdf, chunk_equation_aware

# Folders
DATA_DIR = Path("data/papers")
DB_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Helper: create unique ID for each file
def file_id(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]

def main():
    # Create Chroma persistent client
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="scientific")

    # Load embedding model
    embedder = SentenceTransformer(EMBED_MODEL)

    # Get all PDFs
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDFs found in data/papers/")
        return

    for pdf in tqdm(pdfs, desc="Indexing PDFs"):
        fid = file_id(pdf)
        meta_common = {"source": str(pdf), "doc_id": fid}

        # Parse and chunk PDF
        sections = parse_pdf(str(pdf))
        all_chunks = []
        for s in sections:
            all_chunks.extend(chunk_equation_aware(s, target=900, overlap=150))

        if not all_chunks:
            print(f"⚠ No chunks extracted from {pdf.name}")
            continue

        # Create embeddings
        embeddings = embedder.encode(
            all_chunks,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Create IDs and metadata for each chunk
        ids = [f"{fid}_{i}" for i in range(len(all_chunks))]
        metas = [{**meta_common, "chunk_index": i} for i in range(len(all_chunks))]

        # Add to Chroma
        collection.add(
            documents=all_chunks,
            metadatas=metas,
            embeddings=embeddings,
            ids=ids
        )

        print(f"✅ Indexed {len(all_chunks)} chunks from {pdf.name}")

    print("🎯 Ingestion complete!")
    print("📦 Total chunks in DB:", collection.count())

if __name__ == "__main__":
    main()
