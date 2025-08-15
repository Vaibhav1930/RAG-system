import re
from pathlib import Path
from pypdf import PdfReader

# 1) Read the PDF into plain text
def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)

# 2) Fix common text issues (keep equations safe)
def preserve_equations(text: str) -> str:
    # Remove hyphen at line breaks: "en-\nergy" -> "energy"
    text = re.sub(r'(\S)-\n(\S)', r'\1\2', text)
    return text

# 3) Split text into logical sections (like Abstract, Methods, etc.)
HEADING_RX = re.compile(
    r'^\s*(abstract|introduction|methods?|materials?|results?|discussion|conclusion|references)\b',
    re.I
)

def split_sections(text: str):
    blocks, current = [], []
    for para in text.split("\n\n"):
        if HEADING_RX.match(para.strip()):
            if current:
                blocks.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)
    if current:
        blocks.append("\n\n".join(current))
    return blocks

# 4) Cut sections into smaller chunks (≈900 characters) with overlap
def chunk_equation_aware(section: str, target=900, overlap=150):
    paras = [p.strip() for p in section.split("\n") if p.strip()]
    chunks, buff = [], []
    size = 0

    for p in paras:
        # Detect if paragraph has an equation
        has_eq = bool(re.search(r'(\$.*?\$)|([=]+)|([A-Za-z]\^\{?\d+\}?)', p))
        p_len = len(p)

        # If adding this paragraph makes chunk too big, save it and start new
        if size + p_len > target and buff:
            chunk = "\n".join(buff)
            chunks.append(chunk)

            # Add a small overlap from the end of the last chunk
            tail = chunk[-overlap:]
            buff = [tail]
            size = len(tail)

        buff.append(p)
        size += p_len

        # If there's an equation and chunk is already big enough, save early
        if has_eq and size > target * 0.7:
            chunks.append("\n".join(buff))
            buff, size = [], 0

    if buff:
        chunks.append("\n".join(buff))

    return [c for c in chunks if c.strip()]

# 5) Main function to run everything
def parse_pdf(path: str):
    raw_text = pdf_to_text(path)
    cleaned_text = preserve_equations(raw_text)
    sections = split_sections(cleaned_text)
    return sections

