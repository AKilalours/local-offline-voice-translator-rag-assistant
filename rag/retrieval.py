# rag/retrieval.py
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = Path("rag_index")
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"


@dataclass
class RetrievedChunk:
    text: str
    similarity: float  # cosine-like similarity in [~ -1, 1]
    source: str
    chunk_id: str


class TfidfRetriever:
    """
    LangChain FAISS default index uses L2 distance.
    With TF-IDF vectors L2-normalized (norm='l2'), squared L2 distance relates to cosine:
      ||a - b||^2 = 2 - 2*cos(a,b)  ->  cos = 1 - dist/2
    similarity_search_with_score() returns the L2 distance (typically squared L2).
    """

    def __init__(self, top_k: int = 3, min_similarity: float = 0.25):
        if not INDEX_DIR.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError("Missing rag_index/. Run ingest/build_index.py first.")

        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)

        # NOTE: allow_dangerous_deserialization is fine for your own local files,
        # but do NOT load untrusted indexes with this enabled.
        from ingest.build_index import TfidfEmbeddings  # local import to avoid circular at module import time

        self.embeddings = TfidfEmbeddings(self.vectorizer)
        self.vectordb: FAISS = FAISS.load_local(
            str(INDEX_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        self.top_k = top_k
        self.min_similarity = min_similarity

    @staticmethod
    def _l2_to_cosine(dist: float) -> float:
        # cos = 1 - dist/2 (for normalized vectors, dist is squared L2)
        return 1.0 - float(dist) / 2.0

    def retrieve(self, query: str) -> Tuple[List[RetrievedChunk], float]:
        docs_and_dists = self.vectordb.similarity_search_with_score(query, k=self.top_k)

        chunks: List[RetrievedChunk] = []
        max_sim = -1.0

        for doc, dist in docs_and_dists:
            sim = self._l2_to_cosine(dist)
            max_sim = max(max_sim, sim)

            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    similarity=sim,
                    source=doc.metadata.get("source", "unknown"),
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                )
            )

        # Filter below threshold -> enables real "empty retrieval" for unanswerables
        chunks = [c for c in chunks if c.similarity >= self.min_similarity]
        chunks.sort(key=lambda c: c.similarity, reverse=True)

        if not chunks:
            return [], max_sim

        return chunks, max_sim
