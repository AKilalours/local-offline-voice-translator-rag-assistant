import os
import pickle
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from sklearn.feature_extraction.text import TfidfVectorizer


DOCS_DIR = "docs"
INDEX_DIR = "rag_index"
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")


class TfidfEmbeddings(Embeddings):
    """Simple TF-IDF embedding wrapper compatible with LangChain."""

    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        mat = self.vectorizer.transform(texts)
        return mat.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        mat = self.vectorizer.transform([text])
        return mat.toarray()[0].tolist()


def load_docs():
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    return docs


def main():
    docs = load_docs()
    if not docs:
        print("No documents found in docs/. Add some .txt files and run again.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    texts = [d.page_content for d in chunks]

    # Fit TF-IDF on your chunks
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    print("Fitted TF-IDF vectorizer")

    # Save vectorizer so you can reuse it at query time
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # Use FAISS.from_texts with our embedding wrapper
    embeddings = TfidfEmbeddings(vectorizer)
    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=[d.metadata for d in chunks],
    )

    vectordb.save_local(INDEX_DIR)
    print(f"Saved TF-IDF index with {len(chunks)} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
