# eval/eval_phase3.py
import os
import sys
import json
from pathlib import Path
from typing import Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import AppConfig, GroundedTfidfRetriever, DenseRetriever, CrossEncoderReranker, _merge_keep  # noqa: E402

EVAL_PATH = Path("eval/rag_qa.jsonl")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def hit(expected: Set[str], retrieved: Set[str]) -> bool:
    if not expected:
        return len(retrieved) > 0
    return bool(expected & retrieved)


def main():
    cfg = AppConfig()

    tfidf = GroundedTfidfRetriever(cfg)
    dense = DenseRetriever(cfg.dense_embed_model, top_k=cfg.dense_top_k, min_similarity=cfg.dense_min_similarity)
    reranker = CrossEncoderReranker(cfg.rerank_model)

    tf_ok = dn_ok = rr_ok = 0
    total = 0

    for item in load_jsonl(EVAL_PATH):
        if not item.get("is_answerable", True):
            continue
        total += 1
        q = item["question"]
        expected = set(item.get("expected_sources", []))

        tf_chunks, _ = tfidf.retrieve(q)
        tf_sources = {c.source for c in tf_chunks}

        dn_chunks, _ = dense.retrieve(q)
        dn_sources = {c.source for c in dn_chunks}

        rr_chunks = reranker.rerank(q, dn_chunks, top_k=cfg.rerank_top_k)
        rr_chunks = _merge_keep(dn_chunks, rr_chunks, keep_dense_top=1)
        rr_sources = {c.source for c in rr_chunks}

        tf_ok += int(hit(expected, tf_sources))
        dn_ok += int(hit(expected, dn_sources))
        rr_ok += int(hit(expected, rr_sources))

    print("\n=== Phase 3 Retrieval Comparison (answerables) ===")
    print(f"TF-IDF        recall@k: {tf_ok/total:.3f} ({tf_ok}/{total})")
    print(f"Dense         recall@k: {dn_ok/total:.3f} ({dn_ok}/{total})")
    print(f"Dense+Rerank   recall@k: {rr_ok/total:.3f} ({rr_ok}/{total})")


if __name__ == "__main__":
    main()
