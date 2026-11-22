"""Quality report for data/raw/translated/hotpot_ctx_ar.jsonl (for the thesis' data-quality section)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from translate_hotpot_context import leaks_answer  # noqa: E402


def main(path="data/raw/translated/hotpot_ctx_ar.jsonl"):
    recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    leaks = [r for r in recs if leaks_answer(r["question_en"], r["answer_en"], r["question_ar"], r["answer_ar"])]
    latin_answer = [r for r in recs if r["answer_ar"] and not re.search(r"[\u0600-\u06FF0-9]", r["answer_ar"])]
    gold = Counter(sum(p["gold"] for p in r["paragraphs"]) for r in recs)
    n_par = Counter(len(r["paragraphs"]) for r in recs)
    print(f"items: {len(recs):,} | models: {dict(Counter(r.get('model') for r in recs))}")
    print(f"answer copied into the Arabic question: {len(leaks)} ({100 * len(leaks) / max(1, len(recs)):.1f}%)")
    print(f"answers left in Latin script (e.g. names of works): {len(latin_answer)}")
    print(f"gold paragraphs per item: {dict(gold)} | paragraphs per item: {dict(sorted(n_par.items()))}")
    for r in leaks[:10]:
        print(f"  LEAK  answer '{r['answer_en']}' | {r['question_ar']}")
    return len(leaks)


if __name__ == "__main__":
    main(*sys.argv[1:])
