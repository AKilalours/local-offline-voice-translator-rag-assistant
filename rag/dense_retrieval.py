# rag/dense_retrieval.py
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("rag_index_dense")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "chunks.pkl"


@dataclass
class RetrievedChunk:
    text: str
    similarity: float
    source: str
    chunk_id: str


class DenseRetriever:
    def __init__(self, top_k: int = 12, min_similarity: float = 0.25):
        if not FAISS_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Missing rag_index_dense/. Run: python ingest/build_dense_index.py"
            )

        self.top_k = top_k
        self.min_similarity = min_similarity

        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)

        self.metas = payload["metas"]
        self.texts = payload["texts"]
        self.model_name = payload.get("model", "BAAI/bge-small-en-v1.5")

        self.model = SentenceTransformer(self.model_name)

    def retrieve(self, query: str) -> Tuple[List[RetrievedChunk], float]:
        q = (query or "").strip()
        if not q:
            return [], 0.0

        q_emb = self.model.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)

        scores, idxs = self.index.search(q_emb, self.top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        chunks: List[RetrievedChunk] = []
        max_sim = -1.0
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            sim = float(score)  # cosine similarity because vectors are normalized
            max_sim = max(max_sim, sim)
            if sim < self.min_similarity:
                continue

            meta = self.metas[idx]
            chunks.append(
                RetrievedChunk(
                    text=self.texts[idx],
                    similarity=sim,
                    source=meta["source"],
                    chunk_id=meta["chunk_id"],
                )
            )

        chunks.sort(key=lambda c: c.similarity, reverse=True)
        if not chunks:
            return [], max_sim
        return chunks, max_sim

