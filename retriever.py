import sys
from chunker import Chunk
from vector_store import VectorStore

QUERIES = [
    "bidder eligibility criteria requirements",
    "pre-qualification criteria minimum qualification",
    "annual turnover financial eligibility net worth",        
    "experience requirement similar projects completed years",
    "technical qualification certification ISO CMMI",
    "key personnel manpower team requirement",
    "blacklisted debarred PAN GST registration compliance",
    "documents required submitted with bid",
    "consortium joint venture lead member conditions",
]

TOP_K = 50
PER_QUERY_K = 25 


def retrieve(store: VectorStore) -> list[Chunk]:
    scores: dict[int, float] = {}

    for query in QUERIES:
        for chunk,_, boosted in store.search(query, top_k=PER_QUERY_K):
            scores[chunk.chunk_id] = max(scores.get(chunk.chunk_id, 0), boosted)

    for c in store.eligibility_chunks():
        if c.chunk_id not in scores:
            scores[c.chunk_id] = 0.5 

    chunk_map = {c.chunk_id: c for c in store.chunks}
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:TOP_K]
    selected = [chunk_map[cid] for cid in ranked if cid in chunk_map]

    print(f"→ {len(selected)} chunks selected)", file=sys.stderr)
    return selected


def format_context(chunks: list[Chunk]) -> str:
    out = []
    for c in sorted(chunks, key=lambda c: c.page_start):
        pages = f"Page {c.page_start}" if c.page_start == c.page_end else f"Pages {c.page_start}-{c.page_end}"
        section = f" | {c.heading}" if c.heading else ""
        out.append(f"[{pages}{section}]\n{c.text}\n")
    return "\n".join(out)