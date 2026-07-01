import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunker import Chunk

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class VectorStore:
    def __init__(self):
        print(f"→ Loading {EMBEDDING_MODEL}", file=sys.stderr)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.index = None

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embs = []
        for i in tqdm(range(0, len(texts), 64), desc="Embedding", unit="batch"):
            embs.append(self.model.encode(texts[i:i+64], normalize_embeddings=True,show_progress_bar=False))
        matrix = np.vstack(embs).astype("float32")
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

        del self.model
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
            print("→ Embedding model unloaded from VRAM", file=sys.stderr)
        except ImportError:
            pass

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float, float]]:
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        q = self.model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        raw_scores, idxs = self.index.search(q, min(top_k * 3, self.index.ntotal))
        results = [
            (self.chunks[i], float(s), float(s) * self.chunks[i].boost_score)
            for s, i in zip(raw_scores[0], idxs[0]) if i >= 0
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def eligibility_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if c.is_eligibility_section]