import os

def ieee_from_meta(meta):
    # If you add title/authors/year in ingestion, render them here.
    # Fallback to filename.
    return f"[{meta['doc_id']}] {os.path.basename(meta['source'])}"

def render_ieee_list(metas):
    seen = {}
    for m in metas:
        seen[m["doc_id"]] = ieee_from_meta(m)
    return "\n".join(seen.values())
