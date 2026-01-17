# ingest/build_index.py
import os
import re
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from sklearn.feature_extraction.text import TfidfVectorizer

DOCS_DIR = Path("docs")
INDEX_DIR = Path("rag_index")
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"

ROLE_ORDER = {"public": 0, "internal": 1, "confidential": 2, "admin": 3}


def parse_access_level(text: str, default: str = "public") -> str:
    if not text:
        return default
    # Look at first few lines for ACCESS_LEVEL: ...
    head = "\n".join(text.splitlines()[:5])
    m = re.search(r"^\s*ACCESS_LEVEL\s*:\s*([a-zA-Z]+)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return default
    lvl = m.group(1).strip().lower()
    return lvl if lvl in ROLE_ORDER else default


class TfidfEmbeddings(Embeddings):
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        mat = self.vectorizer.transform(texts)
        return mat.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        mat = self.vectorizer.transform([text])
        return mat.toarray()[0].tolist()


def main():
    if not DOCS_DIR.exists():
        raise SystemExit("Missing docs/ folder. Create docs/ and add .txt files first.")

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        raise SystemExit("No .txt files found in docs/. Add some docs and rerun.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

    chunks = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown.txt"
        filename = os.path.basename(str(src))

        access_level = parse_access_level(doc.page_content, default="public")

        doc_chunks = splitter.split_documents([doc])
        for j, c in enumerate(doc_chunks):
            c.metadata["source"] = filename
            c.metadata["chunk_id"] = f"{filename}::chunk{j}"
            c.metadata["access_level"] = access_level
            chunks.append(c)

    texts = [c.page_content for c in chunks]

    vectorizer = TfidfVectorizer(
        norm="l2",
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
    )
    vectorizer.fit(texts)

    INDEX_DIR.mkdir(exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    embeddings = TfidfEmbeddings(vectorizer)
    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=[c.metadata for c in chunks],
    )
    vectordb.save_local(str(INDEX_DIR))

    print("\n[INGEST] Done.")
    print(f"[INGEST] Docs loaded: {len(docs)}")
    print(f"[INGEST] Chunks built: {len(chunks)}")
    print(f"[INGEST] Saved index to: {INDEX_DIR}")
    print(f"[INGEST] Saved vectorizer to: {VECTORIZER_PATH}\n")


if __name__ == "__main__":
    main()
