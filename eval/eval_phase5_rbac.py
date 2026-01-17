# eval/eval_phase5_rbac.py
import os
import sys
import json
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import AppConfig, RAGPipeline, REFUSAL  # noqa: E402

EVAL_PATH = Path("eval/rag_qa_rbac.jsonl")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    cfg = AppConfig(retrieval_backend="dense_rerank")
    cfg.enable_general_fallback = False
    rag = RAGPipeline(cfg)

    total = ok = 0
    for item in load_jsonl(EVAL_PATH):
        total += 1
        q = item["question"]
        role = item.get("role", "public")
        should_answer = bool(item.get("is_answerable", True))

        ans = rag.query(q, user_role=role)  # requires Phase 5 role support
        refused = (ans.strip() == REFUSAL)

        passed = (not refused) if should_answer else refused
        ok += int(passed)

        print(f"\nrole={role} answerable={should_answer}")
        print(f"Q: {q}")
        print(f"refused={refused}")

    print("\n=== Phase 5 RBAC Eval ===")
    print(f"Pass rate: {ok}/{total} = {ok/total:.3f}")

if __name__ == "__main__":
    main()
