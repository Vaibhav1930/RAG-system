import os
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from utils.parsing import parse_pdf, chunk_equation_aware

# Configuration
DB_DIR = "vectorstore"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Helper: create unique ID for each file
def file_id(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:12]

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into user isolated workspace")
    parser.add_argument("--user", type=str, default="default", help="User workspace ID")
    args = parser.parse_args()

    user_id = args.user
    data_dir = Path("data") / "users" / user_id / "papers"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create Chroma persistent client
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Sanitize collection name for Chroma: 3-63 characters, alphanumeric, underscore or hyphen
    safe_user_id = "".join([c if c.isalnum() or c in "-_" else "_" for c in user_id])
    if len(safe_user_id) < 3:
        safe_user_id = safe_user_id.ljust(3, "0")
    collection_name = f"user_{safe_user_id}"[:63]
    collection = client.get_or_create_collection(name=collection_name)

    # Load embedding model
    embedder = SentenceTransformer(EMBED_MODEL)

    # Get all PDFs in user directory
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {data_dir}")
        return

    for pdf in tqdm(pdfs, desc=f"Indexing PDFs for user '{user_id}'"):
        fid = file_id(pdf)
        meta_common = {"source": str(pdf).replace("\\", "/"), "doc_id": fid}

        # Parse and chunk PDF
        sections = parse_pdf(str(pdf))
        all_chunks = []
        for s in sections:
            all_chunks.extend(chunk_equation_aware(s, target=900, overlap=150))

        if not all_chunks:
            print(f"[WARN] No chunks extracted from {pdf.name}")
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

        # Add to Chroma user collection
        collection.add(
            documents=all_chunks,
            metadatas=metas,
            embeddings=embeddings,
            ids=ids
        )

        print(f"[OK] Indexed {len(all_chunks)} chunks from {pdf.name}")

    print("[SUCCESS] Ingestion complete!")
    print(f"[DB] Total chunks in user '{user_id}' DB:", collection.count())

if __name__ == "__main__":
    main()
