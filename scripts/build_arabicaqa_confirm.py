"""Second confirmatory set: unused ArabicaQA questions, rewritten into Egyptian Arabic by gpt-4.1-mini.

  python scripts/build_arabicaqa_confirm.py --n 600 --seed 20261023 --budget_usd 1.0

Steps (all fixed here, before any question is drawn):
 1. Candidates: data/processed/arabicaqa.jsonl, official split "test" only.
 2. Excluded: every ArabicaQA id that appears anywhere in data/labels/, data/splits/ or data/egy_test/ (router
    training, dialect-head training, held-out and earlier test questions), any question whose text equals one of
    those, and any question whose gold passage is a gold passage of an earlier test question.
 3. Kept only if: reference answer of 1-8 words that appears word for word in the gold passage; gold passage in the
    corpus; question of 4-30 words; no reference to "the passage/text" (الفقرة، النص، المقطع، الوثيقة); no duplicate.
 4. Seeded random sample of --n questions.
 5. Each MSA question is rewritten into Egyptian with EXACTLY the procedure used for the 300 held-out test questions
    (scripts/build_egy_testset.py: same prompt, same automatic checks, one retry), gpt-4.1-mini, temperature 0.
 6. Dropped after rewriting if a check still fails: not Arabic, unchanged, no Egyptian marker word, number lost,
    answer leaked, much longer than the original.
Writes data/confirm_arabicaqa/:
  confirm_test.jsonl   the question file for run_confirmatory.py
  review.csv           60 random rewrites for a native speaker to check (ok = y/n, fix, notes)
  protocol.txt         seed, counts at every step, date, and the SHA-256 of freeze_manifest.json
Every LLM call is cached and capped by --budget_usd.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

ID_RE = re.compile(r"arabicaqa:[^\"\s|,\]]+")
PASSAGE_REF = re.compile(r"(الفقرة|الفقره|النص|المقطع|الوثيقة)")
DROP = ("not Arabic", "unchanged", "no Egyptian marker word", "number lost", "answer leaked", "much longer")


def used_ids_and_texts(dirs):
    ids, texts = set(), set()
    for d in dirs:
        for f in Path(d).rglob("*"):
            if f.is_file() and f.suffix in (".jsonl", ".json", ".csv", ".txt"):
                t = f.read_text(encoding="utf-8", errors="ignore")
                ids.update(ID_RE.findall(t))
                if f.suffix == ".jsonl":
                    for line in t.splitlines():
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        for k in ("question", "q_msa", "q"):
                            if isinstance(r.get(k), str):
                                texts.add(" ".join(r[k].split()))
    return ids, texts


def select(exs, used_ids, used_texts, used_gold, corpus_ids, n, seed, answer_in_context, passages):
    counts = {"arabicaqa questions": len(exs)}
    pool = [e for e in exs if (e.split or "").lower() == "test"]
    counts["official test split"] = len(pool)
    if not pool:
        from collections import Counter
        raise SystemExit(f"no question has split == 'test'; split values found: {dict(Counter(e.split for e in exs))}")
    pool = [e for e in pool if e.id not in used_ids and " ".join(e.question.split()) not in used_texts
            and not (set(e.gold_passage_ids) & used_gold)]
    counts["after excluding everything used before"] = len(pool)
    seen, keep = set(), []
    for e in pool:
        q = " ".join(e.question.split())
        a = str(e.answer or "").strip()
        nw = len(q.split())
        gold = [p for p in e.gold_passage_ids if p in corpus_ids]
        if (q in seen or not gold or not (1 <= len(a.split()) <= 8) or not (4 <= nw <= 30) or PASSAGE_REF.search(q)
                or not any(answer_in_context(a, passages[p]) for p in gold)):
            continue
        seen.add(q)
        keep.append(e)
    counts["after quality filters"] = len(keep)
    keep.sort(key=lambda e: e.id)
    Random(seed).shuffle(keep)
    pick = keep[:n]
    counts["sampled"] = len(pick)
    return pick, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget_usd", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="data/confirm_arabicaqa")
    a = ap.parse_args()

    from arroute.data import read_jsonl
    from arroute.egy.testset import answer_in_context
    from arroute.llm import Budget
    from arroute.llm import complete_with_usage as complete_fn
    from build_egy_testset import translate

    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    if not model.endswith("gpt-4.1-mini"):
        raise SystemExit(f"ARROUTE_LLM_MODEL is {model}: set it to gpt-4.1-mini (the model used for the held-out set)")
    used_ids, used_texts = used_ids_and_texts(["data/labels", "data/splits", "data/egy_test"])
    used_gold = set()
    for x in Path("data/egy_test/egy_test.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            used_gold.update(json.loads(x).get("gold_pids") or [])
    passages = {}
    with (Path(a.corpus) / "passages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                passages[p["id"]] = p["text"]
    exs = read_jsonl(Path(a.processed) / "arabicaqa.jsonl")
    pick, counts = select(exs, used_ids, used_texts, used_gold, set(passages), a.n, a.seed, answer_in_context, passages)
    for k, v in counts.items():
        print(f"  {k:42s} {v:,}")
    if len(pick) < a.n:
        print(f"WARNING: only {len(pick)} questions available")

    budget = Budget(a.budget_usd)
    gen = {"fn": complete_fn, "model": model, "budget": budget}
    recs = [{"id": e.id, "question": e.question, "answer": e.answer} for e in pick]
    with ThreadPoolExecutor(a.workers) as ex:
        out = list(ex.map(lambda r: translate(r, gen), recs))
    by_id = {e.id: e for e in pick}
    good, dropped = [], []
    for t in out:
        (dropped if any(f.startswith(DROP) for f in t["flags"]) else good).append(t)
    counts["rewritten into Egyptian"] = len(out)
    counts["dropped by automatic checks"] = len(dropped)
    counts["final questions"] = len(good)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    with (o / "confirm_test.jsonl").open("w", encoding="utf-8") as f:
        for t in good:
            e = by_id[t["id"]]
            f.write(json.dumps({"id": f"confirm2:{e.id}", "set": "confirm_arabicaqa", "q_egy": t["q_egy"], "q_msa": e.question,
                                "answer": e.answer, "refs": [e.answer], "gold_pids": list(e.gold_passage_ids),
                                "origin_id": e.id, "flags": t["flags"], "tries": t["tries"]}, ensure_ascii=False) + "\n")
    sample = good[:]
    Random(a.seed + 1).shuffle(sample)
    with (o / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "msa_question", "egyptian_question", "ok (y/n)", "fix", "notes"])
        for i, t in enumerate(sample[:60], 1):
            w.writerow([i, t["q_msa"], t["q_egy"], "", "", ""])
    fm = Path("freeze_manifest.json")
    lines = [f"created {datetime.now(timezone.utc).isoformat(timespec='seconds')}", f"seed {a.seed}", f"model {model}",
             f"freeze_manifest.json sha256 {hashlib.sha256(fm.read_bytes()).hexdigest() if fm.exists() else 'NOT FOUND'}"]
    lines += [f"{k}: {v}" for k, v in counts.items()]
    lines += ["dropped examples: " + "; ".join(f"{t['id']} ({', '.join(t['flags'])})" for t in dropped[:20])]
    (o / "protocol.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:-1]))
    print(f"spend ${budget.spent:.4f} | wrote {o / 'confirm_test.jsonl'}, {o / 'review.csv'}, {o / 'protocol.txt'}")


if __name__ == "__main__":
    main()
