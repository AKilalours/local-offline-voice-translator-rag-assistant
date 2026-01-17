# ingest/build_dense_index.py
import os
import re
import pickle
from pathlib import Path
from typing import List

import numpy as np

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_DIR = Path("docs")
OUT_DIR = Path("rag_index_dense")
FAISS_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "chunks.pkl"

ROLE_ORDER = {"public": 0, "internal": 1, "confidential": 2, "admin": 3}


def parse_access_level(text: str, default: str = "public") -> str:
    if not text:
        return default
    head = "\n".join(text.splitlines()[:5])
    m = re.search(r"^\s*ACCESS_LEVEL\s*:\s*([a-zA-Z]+)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return default
    lvl = m.group(1).strip().lower()
    return lvl if lvl in ROLE_ORDER else default


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

    texts: List[str] = []
    metas: List[dict] = []

    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown.txt"
        filename = os.path.basename(str(src))
        access_level = parse_access_level(doc.page_content, default="public")

        doc_chunks = splitter.split_documents([doc])
        for j, c in enumerate(doc_chunks):
            chunk_id = f"{filename}::chunk{j}"
            texts.append(c.page_content)
            metas.append({"source": filename, "chunk_id": chunk_id, "access_level": access_level})

    from sentence_transformers import SentenceTransformer
    import faiss

    model_name = "BAAI/bge-small-en-v1.5"
    print(f"[DENSE] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    embs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(embs)

    OUT_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)

    print("\n[DENSE] Done.")
    print(f"[DENSE] Docs loaded: {len(docs)}")
    print(f"[DENSE] Chunks built: {len(texts)}")
    print(f"[DENSE] Saved FAISS index to: {FAISS_PATH}")
    print(f"[DENSE] Saved metadata to: {META_PATH}\n")


if __name__ == "__main__":
    main()
