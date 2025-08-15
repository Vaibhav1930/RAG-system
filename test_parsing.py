from utils.parsing import parse_pdf, chunk_equation_aware
from pathlib import Path

# Find first PDF in data/papers
pdf_dir = Path("data/papers")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/papers/ — please add at least one.")

pdf_path = pdf_files[0]
print(f"Using PDF: {pdf_path.name}")

# Step 1: Parse PDF into sections
sections = parse_pdf(str(pdf_path))
print(f"Found {len(sections)} sections in {pdf_path.name}")

# Step 2: Chunk each section
all_chunks = []
for sec in sections:
    chunks = chunk_equation_aware(sec)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")
print("\n--- First chunk preview ---\n")
print(all_chunks[0] if all_chunks else "No chunks created.")
