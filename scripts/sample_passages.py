"""Step 1 of the confirmatory test: draw passages for question writers, AFTER the pipeline is frozen.

  python scripts/sample_passages.py --n 600 --seed 20261101 --out data/confirm/passages_to_write.csv

Draws random passages from the corpus, skipping every passage that is the gold passage of an existing test question
(--test) and passages that are too short or too long. Writers receive one passage per row and write, in their own
words: q_egy (a question in natural Egyptian Arabic that the passage answers), q_msa (the same question in MSA),
answer (a short answer copied or paraphrased from the passage). Draw ~20% more than the 500 you need, because some
passages will not yield a good factual question and are skipped by the writer (leave the row blank).
The seed is part of the record: write it in the paper.
"""
import argparse, csv, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--out", default="data/confirm/passages_to_write.csv")
    a = ap.parse_args()
    used = set()
    tp = Path(a.test)
    if tp.exists():
        for x in tp.read_text(encoding="utf-8").splitlines():
            if x.strip():
                used.update(json.loads(x).get("gold_pids") or [])
    pool = []
    with (Path(a.corpus) / "passages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["id"] not in used and a.min_chars <= len(p["text"]) <= a.max_chars:
                    pool.append(p)
    random.Random(a.seed).shuffle(pool)
    pick = pool[:a.n]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "passage", "q_egy", "q_msa", "answer"])
        for p in pick:
            w.writerow([p["id"], p["text"], "", "", ""])
    print(f"{len(pool)} eligible passages ({len(used)} gold passages of existing questions excluded); wrote {len(pick)} to {out}")


if __name__ == "__main__":
    main()
