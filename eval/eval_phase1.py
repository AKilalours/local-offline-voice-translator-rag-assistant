# eval/eval_phase1.py
import os
import sys
import json
import re
from pathlib import Path
from typing import Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import GroundedTfidfRetriever, RAGPipeline, AppConfig, REFUSAL  # noqa: E402

EVAL_PATH = Path("eval/rag_qa.jsonl")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(run_llm: bool = False):
    if not EVAL_PATH.exists():
        raise SystemExit(f"Missing eval file: {EVAL_PATH}")

    cfg = AppConfig()
    retriever = GroundedTfidfRetriever(cfg)

    rag = RAGPipeline(cfg) if run_llm else None

    ans_total = ans_hit = 0
    un_total = un_refuse = 0
    cited_total = cited_ok = 0

    for item in load_jsonl(EVAL_PATH):
        q = item["question"]
        expected_sources: Set[str] = set(item.get("expected_sources", []))
        is_answerable = bool(item.get("is_answerable", True))

        chunks, max_sim = retriever.retrieve(q)
        retrieved_sources = {c.source for c in chunks}

        if is_answerable:
            ans_total += 1
            hit = bool(expected_sources & retrieved_sources) if expected_sources else (len(chunks) > 0)
            ans_hit += int(hit)

            if run_llm and rag:
                cited_total += 1
                ans = rag.query(q)
                cited_ok += int((ans != REFUSAL) and bool(re.search(r"chunk_id\s*=", ans)))
        else:
            un_total += 1
            refused = (len(chunks) == 0)
            un_refuse += int(refused)

        print(f"\nQ: {q}")
        print(f"  answerable={is_answerable} expected_sources={expected_sources}")
        print(f"  retrieved_sources={retrieved_sources} max_sim={max_sim:.3f} kept={len(chunks)}")

    print("\n=== Phase 1 Metrics ===")
    if ans_total:
        print(f"Retrieval recall@{cfg.rag_top_k} (answerables): {ans_hit/ans_total:.3f} ({ans_hit}/{ans_total})")
    if un_total:
        print(f"Unanswerable refusal via threshold: {un_refuse/un_total:.3f} ({un_refuse}/{un_total})")
    if run_llm and cited_total:
        print(f"Citation presence (generation): {cited_ok/cited_total:.3f} ({cited_ok}/{cited_total})")


if __name__ == "__main__":
    main(run_llm=False)
