# scripts/benchmark_real.py
import os
import statistics
import time
from typing import Callable, List, Tuple

# Ensure repo root on PYTHONPATH
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import AppConfig, GroundedTfidfRetriever, RAGPipeline, TranslatorPipeline, TTSPipeline  # noqa: E402


WARMUP_RUNS = 1
BENCH_RUNS = 10


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(p * (len(s) - 1))
    return s[idx]


def bench(name: str, fn: Callable[[], None]) -> Tuple[float, float]:
    times: List[float] = []
    for i in range(WARMUP_RUNS + BENCH_RUNS):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000
        if i >= WARMUP_RUNS:
            times.append(dt)
    med = statistics.median(times)
    p95 = pct(times, 0.95)
    print(f"{name}: median={med:.1f} ms, p95={p95:.1f} ms")
    return med, p95


def main():
    cfg = AppConfig()

    # Load components once (important)
    retriever = GroundedTfidfRetriever(cfg)
    rag = RAGPipeline(cfg)
    translator = TranslatorPipeline(cfg)
    tts = TTSPipeline(cfg)

    # Fixed benchmark queries (repeatable)
    retrieval_query = "What does RAG mean in this project?"
    rag_query = "What does RAG mean in this project?"
    translation_text = "How are you?"
    translation_lang = "French"
    tts_text = "This is a benchmark test."

    # Bench retrieval only
    def run_retrieval():
        chunks, _ = retriever.retrieve(retrieval_query)
        _ = len(chunks)

    # Bench RAG generation (calls Ollama)
    def run_rag():
        _ = rag.query(rag_query)

    # Bench translation (calls Ollama)
    def run_translate():
        _ = translator.translate(translation_text, translation_lang)

    # Bench TTS (runs Coqui + audio playback)
    # If you want silent benchmarks, comment out sd.play in main.py or run without speakers.
    def run_tts():
        tts.synthesize(tts_text)

    print("\n=== Phase 2 Benchmarks (Real) ===")
    bench("Retrieval (TF-IDF+FAISS)", run_retrieval)
    bench("RAG LLM (Ollama)", run_rag)
    bench("Translation LLM (Ollama)", run_translate)
    bench("TTS (Coqui)", run_tts)


if __name__ == "__main__":
    main()
