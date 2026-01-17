# scripts/benchmark_phase3.py
import os
import sys
import statistics
import time
from typing import Callable, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import AppConfig, DenseRetriever, CrossEncoderReranker, _merge_keep  # noqa: E402

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
    q = "What does RAG mean in this project?"

    dense = DenseRetriever(cfg.dense_embed_model, top_k=cfg.dense_top_k, min_similarity=cfg.dense_min_similarity)
    reranker = CrossEncoderReranker(cfg.rerank_model)

    def run_dense():
        chunks, _ = dense.retrieve(q)
        _ = len(chunks)

    def run_dense_rerank():
        chunks, _ = dense.retrieve(q)
        rr = reranker.rerank(q, chunks, top_k=cfg.rerank_top_k)
        _ = _merge_keep(chunks, rr, keep_dense_top=1)

    print("\n=== Phase 3 Benchmarks (Dense + Rerank) ===")
    bench("Dense retrieval", run_dense)
    bench("Dense retrieval + rerank", run_dense_rerank)


if __name__ == "__main__":
    main()
