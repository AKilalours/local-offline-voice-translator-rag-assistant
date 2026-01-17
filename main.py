# main.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import queue
import sys
import signal
import time
import pickle
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# ----------------------------
# Index paths
# ----------------------------
INDEX_DIR = "rag_index"
VECTORIZER_PATH = os.path.join(INDEX_DIR, "tfidf_vectorizer.pkl")

DENSE_INDEX_DIR = "rag_index_dense"
DENSE_FAISS_PATH = os.path.join(DENSE_INDEX_DIR, "faiss.index")
DENSE_META_PATH = os.path.join(DENSE_INDEX_DIR, "chunks.pkl")

REFUSAL = "I don't know based on the provided documents."

# ----------------------------
# Phase 5: RBAC
# ----------------------------
VALID_ROLES = {"public", "internal", "confidential", "admin"}
ROLE_ORDER = {"public": 0, "internal": 1, "confidential": 2, "admin": 3}


def _normalize_role(role: Optional[str]) -> str:
    r = (role or "public").strip().lower()
    return r if r in VALID_ROLES else "public"


def _role_allows(chunk_level: str, user_role: str) -> bool:
    cl = (chunk_level or "public").strip().lower()
    ur = _normalize_role(user_role)
    return ROLE_ORDER.get(cl, 0) <= ROLE_ORDER.get(ur, 0)


# ----------------------------
# Config
# ----------------------------
@dataclass
class AppConfig:
    # Audio
    sample_rate: int = 16000
    block_size: int = 1024
    device: Optional[int] = None

    # Voice capture / VAD
    max_record_secs: float = 10.0
    min_record_secs: float = 1.0
    start_speech_rms: float = 0.008
    end_silence_rms: float = 0.005
    end_silence_secs: float = 0.9
    trim_rms: float = 0.004

    # ASR
    whisper_model_size: str = "small"
    whisper_language: str = "en"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral"

    # RAG (generic)
    rag_max_context_chars: int = 7000
    rag_min_keyword_hits: int = 1
    enable_general_fallback: bool = True

    # Retrieval backend: "tfidf" | "dense" | "dense_rerank"
    retrieval_backend: str = "dense_rerank"

    # TF-IDF
    rag_top_k: int = 4
    rag_min_similarity: float = 0.06

    # Dense
    dense_embed_model: str = "BAAI/bge-small-en-v1.5"
    dense_top_k: int = 20
    dense_min_similarity: float = 0.25

    # Rerank
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 6

    # TTS
    tts_speaker_id: str = "p225"
    tts_max_chars: int = 320

    # Default role for CLI voice app
    default_user_role: str = "public"

    # Recruiter-grade logging
    log_level: str = "demo"          # "demo" | "debug"
    show_chunk_text: bool = False    # only used when log_level="debug"
    max_chunks_to_print: int = 4     # limit stdout noise


# ----------------------------
# Audio utilities
# ----------------------------
def _rms(x: np.ndarray) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def _trim_silence(audio: np.ndarray, sr: int, trim_rms: float) -> np.ndarray:
    if audio is None or len(audio) == 0:
        return audio
    win = int(sr * 0.02)
    if win <= 0 or len(audio) < win:
        return audio

    start = 0
    while start + win < len(audio):
        if _rms(audio[start:start + win]) >= trim_rms:
            break
        start += win

    end = len(audio)
    while end - win > start:
        if _rms(audio[end - win:end]) >= trim_rms:
            break
        end -= win

    trimmed = audio[start:end]
    return trimmed if len(trimmed) > 0 else audio


def _strip_code_blocks(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    had_code = bool(re.search(r"```.*?```", text, flags=re.DOTALL))
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    return cleaned, had_code


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.?!,;:]+$", "", t)
    return t


def _is_question(text: str) -> bool:
    t = (text or "").strip().lower()
    return ("?" in t) or t.startswith(
        ("how do i", "what is", "what are", "tell me", "explain", "how to", "how do")
    )


def _is_live_info_query(text: str) -> bool:
    t = (text or "").lower()
    live_terms = [
        "news", "headline", "headlines", "today",
        "weather", "forecast", "temperature",
        "stock", "price", "bitcoin", "score",
    ]
    return any(w in t for w in live_terms)


def _split_for_tts(text: str, max_chars: int) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t, _ = _strip_code_blocks(t)
    parts = re.split(r"(?<=[.!?])\s+", t)
    chunks: List[str] = []
    buf = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            while len(p) > max_chars:
                chunks.append(p[:max_chars].rstrip())
                p = p[max_chars:].lstrip()
            buf = p

    if buf:
        chunks.append(buf)
    return chunks


def _is_low_information_asr(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if len(t) < 2:
        return True
    norm = re.sub(r"[^a-zA-Z]+", "", t).lower()
    if len(norm) >= 20 and len(set(norm)) <= 3:
        return True
    return False


# ----------------------------
# Policy gate (injection/exfil)
# ----------------------------
_BLOCKLIST_PATTERNS = [
    r"reveal .*system prompt",
    r"\bsystem prompt\b",
    r"print .*context",
    r"\bcontext\b.*(show|print|dump)",
    r"ignore .*rules",
    r"ignore .*instructions",
    r"print .*every document",
    r"full contents .*document",
    r"show .*all documents",
    r"do not refuse",
    r"answer anyway",
]


def _is_policy_blocked(query: str) -> bool:
    t = (query or "").lower()
    return any(re.search(p, t) for p in _BLOCKLIST_PATTERNS)


# Project terms we want grounded (and biased toward glossary)
_PROJECT_TERMS = {"rag", "asr", "tts", "faiss", "ollama", "whisper", "llm"}


def _mentions_project_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in _PROJECT_TERMS)


def _extract_translate_request(text: str) -> Tuple[str, Optional[str]]:
    raw = (text or "").strip()
    t = raw.strip()

    m = re.match(r"^(?:translate\s*,?\s*)?(.*)\s+in\s+([a-zA-Z]+)\s*\.?$", t, flags=re.IGNORECASE)
    if m:
        phrase = m.group(1).strip(" ,.")
        lang = m.group(2).strip().capitalize()
        return phrase, lang

    m2 = re.match(r"^translate\s*,?\s*(.*)$", t, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).strip(), None

    return raw, None


# ----------------------------
# Citation integrity (Phase 5)
# ----------------------------
def _normalize_used_citations(answer: str) -> str:
    """
    Convert common noncompliant patterns like:
      "Used: glossary.txt::chunk0"
    into:
      "Used: chunk_id=glossary.txt::chunk0"
    """
    if not answer:
        return answer

    answer = re.sub(
        r"Used:\s*([A-Za-z0-9_.-]+\.txt::chunk\d+)",
        lambda m: f"Used: chunk_id={m.group(1)}",
        answer,
        flags=re.IGNORECASE,
    )
    return answer


def _extract_cited_chunk_ids(answer: str) -> set:
    if not answer:
        return set()
    return set(re.findall(r"chunk_id=([A-Za-z0-9_.-]+\.txt::chunk\d+)", answer))


def _has_any_citation(answer: str) -> bool:
    if not answer:
        return False
    if "chunk_id=" in answer:
        return True
    if re.search(r"^\s*Used:\s*", answer, flags=re.IGNORECASE | re.MULTILINE):
        return True
    return False


# ----------------------------
# Microphone streaming
# ----------------------------
class MicrophoneStream:
    def __init__(self, sample_rate: int, block_size: int, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self._queue = queue.Queue()
        self._stream = None

    def flush(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Input status: {status}", file=sys.stderr)
        self._queue.put(indata.copy())

    def start(self):
        if self._stream is not None:
            return
        import sounddevice as sd
        print("[AUDIO] Microphone stream started")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self):
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        print("[AUDIO] Microphone stream stopped")

    def read_utterance(
        self,
        max_secs: float,
        min_secs: float,
        start_speech_rms: float,
        end_silence_rms: float,
        end_silence_secs: float,
    ) -> Optional[np.ndarray]:
        sr = self.sample_rate
        max_frames = int(max_secs * sr)
        min_frames = int(min_secs * sr)
        silence_frames_needed = int(end_silence_secs * sr)

        chunks: List[np.ndarray] = []
        started = False
        total = 0
        silence_run = 0

        while total < max_frames:
            try:
                chunk = self._queue.get(timeout=2.0)
            except queue.Empty:
                if not started:
                    return None
                break

            x = chunk.reshape(-1)
            r = _rms(x)

            if not started:
                if r >= start_speech_rms:
                    started = True
                    chunks.append(x)
                    total += len(x)
                    silence_run = 0
            else:
                chunks.append(x)
                total += len(x)

                if r < end_silence_rms:
                    silence_run += len(x)
                else:
                    silence_run = 0

                if total >= min_frames and silence_run >= silence_frames_needed:
                    break

        if not chunks:
            return None

        return np.concatenate(chunks, axis=0)


# ----------------------------
# ASR
# ----------------------------
class ASRPipeline:
    def __init__(self, cfg: AppConfig):
        print("[ASR] Loading Whisper model (first time may take a while)...")
        self.cfg = cfg
        from faster_whisper import WhisperModel
        self.model = WhisperModel(cfg.whisper_model_size, device="cpu", compute_type="int8")
        print("[ASR] Whisper model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        audio = _trim_silence(audio, self.cfg.sample_rate, self.cfg.trim_rms)
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.whisper_language,
            beam_size=5,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = "".join([seg.text for seg in segments]).strip()
        print(f"[ASR] Detected text: {text}")
        return text


# ----------------------------
# Retrieved chunk object (Phase 5: includes access_level)
# ----------------------------
@dataclass
class RetrievedChunk:
    text: str
    similarity: float
    source: str
    chunk_id: str
    access_level: str = "public"
    rerank_score: Optional[float] = None


# ----------------------------
# TF-IDF embeddings + retriever
# ----------------------------
class TfidfEmbeddings(Embeddings):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        mat = self.vectorizer.transform(texts)
        return mat.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        mat = self.vectorizer.transform([text])
        return mat.toarray()[0].tolist()


_COVER_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "my", "your",
    "do", "does", "how", "i", "you", "it", "this", "that", "please", "tell", "me", "about",
    "explain", "define", "difference", "between", "things", "can", "all", "works", "work",
    "simple", "terms", "means", "meaning", "stand", "basics", "brief", "overview"
}


def _extract_keywords(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    toks = [t for t in toks if t not in _COVER_STOPWORDS]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


class GroundedTfidfRetriever:
    def __init__(self, cfg: AppConfig):
        if not os.path.isdir(INDEX_DIR) or not os.path.isfile(VECTORIZER_PATH):
            raise FileNotFoundError(f"Missing {INDEX_DIR}/ or {VECTORIZER_PATH}. Run: python ingest/build_index.py")

        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.embeddings = TfidfEmbeddings(self.vectorizer)
        self.vectordb = FAISS.load_local(
            INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.top_k = cfg.rag_top_k
        self.min_similarity = cfg.rag_min_similarity

    @staticmethod
    def _dist_to_similarity(dist: float) -> float:
        # for normalized vectors, dist is squared L2; cosine ~ 1 - dist/2
        sim = 1.0 - float(dist) / 2.0
        return max(-1.0, min(1.0, sim))

    def retrieve(self, query: str, user_role: str = "public") -> Tuple[List[RetrievedChunk], float]:
        user_role = _normalize_role(user_role)

        if _is_policy_blocked(query):
            return [], 0.0

        q_sparse = self.vectorizer.transform([query])
        if q_sparse.nnz == 0:
            return [], 0.0

        q_vec = q_sparse.toarray()[0].tolist()

        if hasattr(self.vectordb, "similarity_search_with_score_by_vector"):
            docs_and_dists = self.vectordb.similarity_search_with_score_by_vector(q_vec, k=self.top_k)
        else:
            docs_and_dists = self.vectordb.similarity_search_with_score(query, k=self.top_k)

        chunks: List[RetrievedChunk] = []
        max_sim = -1.0

        for rank, (doc, dist) in enumerate(docs_and_dists, start=1):
            sim = self._dist_to_similarity(dist)
            max_sim = max(max_sim, sim)

            source = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
            source = os.path.basename(str(source))
            chunk_id = doc.metadata.get("chunk_id") or f"{source}::rank{rank}"
            access_level = doc.metadata.get("access_level", "public")

            # Phase 5 RBAC enforcement
            if not _role_allows(access_level, user_role):
                continue

            if sim >= self.min_similarity:
                chunks.append(RetrievedChunk(
                    text=doc.page_content,
                    similarity=sim,
                    source=str(source),
                    chunk_id=str(chunk_id),
                    access_level=str(access_level),
                ))

        chunks.sort(key=lambda c: c.similarity, reverse=True)
        if not chunks:
            return [], max_sim
        return chunks, max_sim


# ----------------------------
# Dense retriever (Phase 3) with Phase 5 RBAC
# ----------------------------
class DenseRetriever:
    def __init__(self, model_name: str, top_k: int = 20, min_similarity: float = 0.25):
        if not os.path.isfile(DENSE_FAISS_PATH) or not os.path.isfile(DENSE_META_PATH):
            raise FileNotFoundError("Missing rag_index_dense/. Run: python ingest/build_dense_index.py")

        import faiss
        from sentence_transformers import SentenceTransformer

        self._faiss = faiss
        self.model = SentenceTransformer(model_name)

        self.top_k = top_k
        self.min_similarity = min_similarity

        self.index = faiss.read_index(DENSE_FAISS_PATH)
        with open(DENSE_META_PATH, "rb") as f:
            payload = pickle.load(f)

        self.texts = payload["texts"]
        self.metas = payload["metas"]

    def retrieve(self, query: str, user_role: str = "public") -> Tuple[List[RetrievedChunk], float]:
        user_role = _normalize_role(user_role)

        if _is_policy_blocked(query):
            return [], 0.0

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

        for sim, idx in zip(scores, idxs):
            if idx < 0:
                continue

            sim = float(sim)
            max_sim = max(max_sim, sim)

            meta = self.metas[idx]
            access_level = meta.get("access_level", "public")

            # Phase 5 RBAC enforcement
            if not _role_allows(access_level, user_role):
                continue

            if sim < self.min_similarity:
                continue

            chunks.append(RetrievedChunk(
                text=self.texts[idx],
                similarity=sim,
                source=meta.get("source", "unknown"),
                chunk_id=meta.get("chunk_id", "unknown"),
                access_level=access_level,
            ))

        chunks.sort(key=lambda c: c.similarity, reverse=True)
        if not chunks:
            return [], max_sim
        return chunks, max_sim


# ----------------------------
# Reranker (Phase 3)
# ----------------------------
class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 6) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)

        out: List[RetrievedChunk] = []
        for c, s in zip(chunks, scores):
            out.append(RetrievedChunk(
                text=c.text,
                similarity=c.similarity,
                source=c.source,
                chunk_id=c.chunk_id,
                access_level=c.access_level,
                rerank_score=float(s),
            ))

        out.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else x.similarity, reverse=True)
        return out[: min(top_k, len(out))]


def _merge_keep(dense_chunks: List[RetrievedChunk], reranked: List[RetrievedChunk], keep_dense_top: int = 1) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    seen = set()

    for c in reranked:
        if c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)

    for c in dense_chunks[:keep_dense_top]:
        if c.chunk_id not in seen:
            out.append(c)
            seen.add(c.chunk_id)

    return out


# ----------------------------
# RAG pipeline (Phase 5 RBAC + citation integrity)
# ----------------------------
class RAGPipeline:
    def __init__(self, cfg: AppConfig):
        print("[RAG] Loading vector store...")
        self.cfg = cfg
        self.backend = cfg.retrieval_backend

        self.tfidf = GroundedTfidfRetriever(cfg)

        self.dense = None
        self.reranker = None
        if self.backend in {"dense", "dense_rerank"}:
            self.dense = DenseRetriever(cfg.dense_embed_model, top_k=cfg.dense_top_k, min_similarity=cfg.dense_min_similarity)
        if self.backend == "dense_rerank":
            self.reranker = CrossEncoderReranker(cfg.rerank_model)

        self.ollama_url = cfg.ollama_url
        self.ollama_model = cfg.ollama_model

        print(f"[RAG] Ready (backend={self.backend})")
        self._check_ollama()

    def _check_ollama(self):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            r.raise_for_status()
        except Exception:
            raise RuntimeError(
                "Ollama is not reachable at http://localhost:11434.\n"
                "Fix: in another terminal run: ollama serve\n"
                "Then confirm: ollama list"
            )

    def _call_ollama(self, prompt: str) -> str:
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        resp = requests.post(self.ollama_url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def _retrieve(self, user_text: str, user_role: str) -> List[RetrievedChunk]:
        user_role = _normalize_role(user_role)

        if self.backend == "tfidf":
            chunks, _ = self.tfidf.retrieve(user_text, user_role=user_role)
            return chunks

        assert self.dense is not None
        dense_chunks, _ = self.dense.retrieve(user_text, user_role=user_role)

        # Bias toward glossary for project term questions
        if _mentions_project_terms(user_text) and not any(c.source == "glossary.txt" for c in dense_chunks):
            dense_chunks2, _ = self.dense.retrieve(user_text + " glossary", user_role=user_role)
            if dense_chunks2:
                dense_chunks = dense_chunks2

        if self.backend == "dense":
            return dense_chunks[: self.cfg.rerank_top_k]

        assert self.reranker is not None
        reranked = self.reranker.rerank(user_text, dense_chunks, top_k=self.cfg.rerank_top_k)
        return _merge_keep(dense_chunks, reranked, keep_dense_top=1)

    def query(self, user_text: str, user_role: str = "public") -> str:
        user_role = _normalize_role(user_role)

        if _is_policy_blocked(user_text):
            return REFUSAL

        chunks = self._retrieve(user_text, user_role=user_role)

        # Recruiter-grade terminal output (DEMO vs DEBUG)
        print("\n[RAG] Retrieved chunks:")
        if not chunks:
            print("  <none> (below threshold OR empty query OR policy blocked OR RBAC filtered)")
            return REFUSAL

        for i, c in enumerate(chunks[: self.cfg.max_chunks_to_print], start=1):
            if c.rerank_score is None:
                line = f"{i}. {c.chunk_id} src={c.source} sim={c.similarity:.3f} acl={c.access_level}"
            else:
                line = f"{i}. {c.chunk_id} src={c.source} sim={c.similarity:.3f} rerank={c.rerank_score:.3f} acl={c.access_level}"
            print("  " + line)

            if self.cfg.show_chunk_text and self.cfg.log_level == "debug":
                snippet = c.text[:160].replace("\n", " ")
                print("     " + snippet + "...")

        # Coverage gate
        q_keys = _extract_keywords(user_text)
        ctx_lower = " ".join(c.text for c in chunks).lower()
        if q_keys:
            covered = sum(1 for k in q_keys if k in ctx_lower)
            required = 1
            if self.cfg.log_level == "debug":
                print(f"[RAG] Keyword coverage: {covered}/{len(q_keys)} (required={required}) keys={q_keys[:8]}")
            if covered < required:
                if self.cfg.log_level == "debug":
                    print("[RAG] Coverage gate failed → refusing.")
                return REFUSAL

        # Build context (internal only)
        context_parts = []
        for c in chunks:
            header = f"[chunk_id={c.chunk_id} source={c.source} sim={c.similarity:.3f} acl={c.access_level}]"
            context_parts.append(header + "\n" + c.text)
        context = "\n\n".join(context_parts)

        if len(context) > self.cfg.rag_max_context_chars:
            context = context[: self.cfg.rag_max_context_chars] + "\n\n[truncated]\n"

        system_instruction = (
            "You are a grounded assistant for this offline voice translator + RAG project.\n"
            "Use ONLY the provided CONTEXT. Do NOT use outside knowledge.\n"
            f"If unsupported, reply exactly: \"{REFUSAL}\"\n"
            "You MUST include citations as: Used: chunk_id=<...> (one per line).\n"
            "Do NOT cite any chunk_id that is not present in CONTEXT.\n"
        )

        prompt = system_instruction + "\n\nCONTEXT:\n" + context + f"\n\nUSER: {user_text}\n\nASSISTANT:"
        if self.cfg.log_level == "debug":
            print("[RAG] Querying local LLM via Ollama...")

        answer = self._call_ollama(prompt)
        answer = _normalize_used_citations(answer)

        if self.cfg.log_level == "debug":
            print(f"[RAG] LLM raw answer: {answer}")

        # Citation presence enforcement
        if answer.strip() != REFUSAL and not _has_any_citation(answer):
            if self.cfg.log_level == "debug":
                print("[RAG] Missing citations → forcing refusal.")
            return REFUSAL

        # Phase 5: citation subset enforcement (cannot cite chunks it didn't retrieve)
        retrieved_ids = {c.chunk_id for c in chunks}
        cited_ids = _extract_cited_chunk_ids(answer)
        if cited_ids and not cited_ids.issubset(retrieved_ids):
            if self.cfg.log_level == "debug":
                print(f"[RAG] Citation integrity failed. cited={cited_ids} retrieved={retrieved_ids}")
            return REFUSAL

        return answer


# ----------------------------
# General fallback (offline)
# ----------------------------
class GeneralLLMPipeline:
    def __init__(self, cfg: AppConfig):
        self.ollama_url = cfg.ollama_url
        self.ollama_model = cfg.ollama_model

    @staticmethod
    def _wants_code(text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in ["code", "python", "function", "script", "fastapi", "endpoint", "uvicorn", "example"])

    def _call_ollama(self, prompt: str) -> str:
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        resp = requests.post(self.ollama_url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def query(self, user_text: str) -> str:
        wants_code = self._wants_code(user_text)
        system_instruction = (
            "You are a helpful offline assistant.\n"
            "Be concise.\n"
            "Do NOT output code unless the user explicitly asks for code.\n"
            "Do NOT claim real-time facts.\n"
        )
        prompt = system_instruction + f"\n\nUSER: {user_text}\n\nASSISTANT:"
        ans = self._call_ollama(prompt)
        if not wants_code:
            ans, _ = _strip_code_blocks(ans)
        return ans.strip()


# ----------------------------
# Translator
# ----------------------------
class TranslatorPipeline:
    def __init__(self, cfg: AppConfig):
        self.ollama_url = cfg.ollama_url
        self.ollama_model = cfg.ollama_model

    @staticmethod
    def _postprocess_translation(raw: str) -> str:
        if not raw:
            return raw
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        out = lines[0] if lines else raw.strip()
        out = re.sub(r"\(.*?\)", "", out).strip()
        out = re.sub(r"^translation\s*[:\-]\s*", "", out, flags=re.IGNORECASE).strip()
        out = re.sub(r"^(traduit|translated)\s+.*?:\s*", "", out, flags=re.IGNORECASE).strip()
        return out

    def translate(self, text: str, target_lang: str) -> str:
        prompt = (
            f"Translate into {target_lang}.\n"
            "Return ONLY the translated text. No labels. No explanations. No extra words.\n"
            "Do NOT expand acronyms. Preserve proper nouns unless a standard translation exists.\n\n"
            f"Text: {text}\n\n"
            "Translation:"
        )
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        resp = requests.post(self.ollama_url, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return self._postprocess_translation(raw)


# ----------------------------
# TTS
# ----------------------------
class TTSPipeline:
    def __init__(self, cfg: AppConfig):
        print("[TTS] Loading TTS model...")
        from TTS.api import TTS
        self.tts = TTS(model_name="tts_models/en/vctk/vits")
        self.speaker_id = cfg.tts_speaker_id
        self.max_chars = cfg.tts_max_chars
        print("[TTS] TTS model loaded")

    def synthesize(self, text: str, out_path: str = "output_tts.wav"):
        t = (text or "").strip()
        if not t:
            return

        import soundfile as sf
        import sounddevice as sd

        chunks = _split_for_tts(t, self.max_chars)
        if not chunks:
            return

        for i, chunk in enumerate(chunks, start=1):
            print(f"[TTS] Synthesizing chunk {i}/{len(chunks)}...")
            self.tts.tts_to_file(text=chunk, file_path=out_path, speaker=self.speaker_id)
            data, sr = sf.read(out_path)
            sd.play(data, sr)
            sd.wait()

        print(f"[TTS] Saved to {out_path}")


# ----------------------------
# Main voice app
# ----------------------------
class VoiceRAGTranslatorApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.mic = MicrophoneStream(cfg.sample_rate, cfg.block_size, cfg.device)
        self.asr = ASRPipeline(cfg)
        self.rag = RAGPipeline(cfg)
        self.general = GeneralLLMPipeline(cfg)
        self.translator = TranslatorPipeline(cfg)
        self.tts = TTSPipeline(cfg)

        self.mode: str = "chat"
        self.current_target_lang: str = "English"
        self._running = True

    def stop(self):
        self._running = False
        self.mic.stop()

    def _speak(self, text: str):
        self.mic.stop()
        try:
            self.tts.synthesize(text)
        finally:
            self.mic.start()

    def _handle_info_intents(self, text: str) -> Optional[str]:
        # Only in CHAT mode (in translate mode, everything translates)
        t = _normalize(text)

        if t in {"how are you", "how are you doing", "how r you", "how you doing"}:
            return (
                "I’m an AI assistant, so I don’t have feelings, but I’m ready to help. "
                "You can ask about the project, use translation mode, ask for Python code, or general questions."
            )

        if ("what can you do" in t) or ("capabilities" in t) or ("features" in t) or ("functions" in t):
            return (
                "Here are 5 things I can do:\n"
                "1) Speech to text (ASR)\n"
                "2) Answer from your local documents with citations (RAG)\n"
                "3) Translate speech into another language (translation mode)\n"
                "4) Speak responses aloud (text to speech)\n"
                "5) Answer general questions offline (local LLM)\n"
                "Commands: 'start translation mode', 'translate into <language>', 'stop translation mode'."
            )

        return None

    def _handle_mode_commands(self, text: str) -> bool:
        raw = text or ""
        t = _normalize(raw)

        # Help question only in chat mode (avoid hijacking translation mode)
        if self.mode == "chat" and _is_question(raw) and ("translation mode" in t or "translate into" in t or "target language" in t):
            self._speak(
                "To start: say 'start translation mode'. "
                "To change language: say 'translate into French' or 'translate into Spanish'. "
                "To stop: say 'stop translation mode'."
            )
            return True

        if t in {"start translation mode", "translate", "translation"}:
            self.mode = "translate"
            self._speak(f"Translation mode on. I will translate into {self.current_target_lang}.")
            return True

        if t in {"stop translation mode", "exit translation mode"}:
            self.mode = "chat"
            self._speak("Translation mode off. I will answer normally.")
            return True

        m = re.match(r"^translate (into|to)\s+([a-zA-Z]+)$", t)
        if m:
            self.current_target_lang = m.group(2).capitalize()
            self.mode = "translate"
            self._speak(f"Okay, I will translate into {self.current_target_lang}.")
            return True

        return False

    def run_once(self):
        print("\n[APP] Speak now (Ctrl+C to exit). Listening...")
        self.mic.flush()

        audio = self.mic.read_utterance(
            max_secs=self.cfg.max_record_secs,
            min_secs=self.cfg.min_record_secs,
            start_speech_rms=self.cfg.start_speech_rms,
            end_silence_rms=self.cfg.end_silence_rms,
            end_silence_secs=self.cfg.end_silence_secs,
        )
        if audio is None:
            print("[APP] No speech detected, try again.")
            return

        t0 = time.perf_counter()
        asr_text = self.asr.transcribe(audio)
        t_asr = (time.perf_counter() - t0) * 1000
        print(f"[LATENCY] ASR: {t_asr:.1f} ms")

        if not asr_text:
            print("[APP] Empty transcription.")
            return

        text = asr_text.strip()

        if _is_low_information_asr(text):
            self._speak("I didn’t catch that clearly. Please repeat.")
            return

        # Commands must work in both modes
        if self._handle_mode_commands(text):
            return

        # Translation mode: translate everything except commands (already handled above)
        if self.mode == "translate":
            phrase, explicit_lang = _extract_translate_request(text)
            if explicit_lang:
                self.current_target_lang = explicit_lang

            src = (phrase or "").strip()
            if not src:
                self._speak("Say a phrase to translate.")
                return

            print(f"[APP] Source text: {src}")
            print(f"[APP] Translating to: {self.current_target_lang}")

            t1 = time.perf_counter()
            translated = self.translator.translate(src, self.current_target_lang)
            t_trans = (time.perf_counter() - t1) * 1000
            print(f"[LATENCY] Translation (LLM): {t_trans:.1f} ms")
            print(f"\n[APP] Translated text:\n{translated}\n")
            self._speak(translated)
            return

        # Chat mode: live info -> offline constraint
        if _is_live_info_query(text):
            msg = (
                "I run offline, so I can’t fetch live news, weather, or market prices. "
                "If you want real-time results, you’d need to connect an online API."
            )
            print(f"\n[APP] Final Answer (OFFLINE-LIMIT):\n{msg}\n")
            self._speak(msg)
            return

        # Chat mode: info intents
        info = self._handle_info_intents(text)
        if info:
            print(f"\n[APP] Answer (INFO):\n{info}\n")
            self._speak(info)
            return

        # Chat mode: RAG then general fallback (but NOT for project terms)
        t1 = time.perf_counter()
        answer = self.rag.query(text, user_role=self.cfg.default_user_role)
        used = "RAG"

        if answer == REFUSAL and self.cfg.enable_general_fallback and not _mentions_project_terms(text):
            answer = self.general.query(text)
            used = "GENERAL"

        t_llm = (time.perf_counter() - t1) * 1000
        print(f"[LATENCY] Answer ({used}): {t_llm:.1f} ms")
        print(f"\n[APP] Final Answer ({used}):\n{answer}\n")
        self._speak(answer)

    def run_loop(self):
        self.mic.start()
        print("[APP] Voice chat + translator running. Press Ctrl+C to stop.")
        try:
            while self._running:
                self.run_once()
        except KeyboardInterrupt:
            print("\n[APP] Keyboard interrupt, stopping...")
        finally:
            self.stop()


def main():
    cfg = AppConfig(
        retrieval_backend="dense_rerank",
        log_level="demo",
        show_chunk_text=False,
        max_chunks_to_print=4,
        default_user_role="public",
    )
    app = VoiceRAGTranslatorApp(cfg)

    def handle_sigint(sig, frame):
        print("\n[APP] Caught SIGINT, shutting down...")
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    app.run_loop()


if __name__ == "__main__":
    main()
