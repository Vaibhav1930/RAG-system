import json, statistics
from rag_pipeline import RAG

def load_qas(path="data/qas.jsonl"):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def hit_at_k(ref_docids, retrieved_hits):
    got = {h[1]["doc_id"] for h in retrieved_hits}
    return int(bool(set(ref_docids) & got))

def main():
    rag = RAG(top_k=5)
    hits = []
    for ex in load_qas():
        _, _, retrieved = rag.answer(ex["question"])
        hits.append(hit_at_k(ex.get("refs", []), retrieved))
    print("Hit@K:", sum(hits)/len(hits))

if __name__ == "__main__":
    main()
