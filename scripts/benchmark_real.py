# scripts/benchmark_real.py
import os
import statistics
import time
from typing import Callable, List

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import AppConfig, GroundedTfidfRetriever, RAGPipeline, TranslatorPipeline  # noqa: E402

WARMUP_RUNS = 1
BENCH_RUNS = 10


def pct(values: List[float], p: float) -> float:
    s = sorted(values)
    idx = int(p * (len(s) - 1))
    return s[idx]


def bench(name: str, fn: Callable[[], None]) -> None:
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


def main():
    cfg = AppConfig()

    retriever = GroundedTfidfRetriever(cfg)
    rag = RAGPipeline(cfg)
    translator = TranslatorPipeline(cfg)

    retrieval_query = "What does RAG mean in this project?"
    rag_query = "What does RAG mean in this project?"
    translation_text = "How are you?"
    translation_lang = "French"

    def run_retrieval():
        chunks, _ = retriever.retrieve(retrieval_query)
        _ = len(chunks)

    def run_rag():
        _ = rag.query(rag_query)

    def run_translate():
        _ = translator.translate(translation_text, translation_lang)

    print("\n=== Phase 2 Benchmarks (Real, TTS excluded to avoid native segfault) ===")
    bench("Retrieval (TF-IDF+FAISS)", run_retrieval)
    bench("RAG LLM (Ollama)", run_rag)
    bench("Translation LLM (Ollama)", run_translate)
    print("\nNote: Coqui TTS benchmarking is excluded here due to an intermittent native segfault when initialized in a non-interactive benchmark run on this environment. TTS latency is measured during interactive runs via in-app timing logs.")


if __name__ == "__main__":
    main()
