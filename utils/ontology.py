# Simple alias map improves recall for terms (ATP -> adenosine triphosphate)
ALIASES = {
    "ATP": ["adenosine triphosphate"],
    "PCM": ["phase change material", "phase-change material"],
    # add domain-specific equivalents
}

def expand_query(q: str):
    for k, vals in ALIASES.items():
        if k.lower() in q.lower():
            q += " " + " ".join(vals)
    return q
