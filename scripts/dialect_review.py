"""Native-speaker review of the non-native parts of the Egyptian test set (and a native control sample).

  python scripts/dialect_review.py make --reviewers 2               # writes review_dialect_R1.csv, _R2.csv
  python scripts/dialect_review.py summarize                        # after both are filled in

Rows: all 200 pilot questions (written with LLM help), all 300 held-out questions (machine-translated), and 50 random
native ArDQA questions as a control (reviewers are not told which is which; rows are shuffled per reviewer).
Each reviewer fills, per row:
  natural   1 = not how an Egyptian would say it, 2 = understandable but stiff, 3 = natural Egyptian
  same      y if the Egyptian question asks exactly what the MSA question asks, else n
  answer_ok y if the reference answer is correct and complete for the question, judging by the passage, else n
  notes     optional
summarize writes runs_audit/dialect_review_summary.txt (per part: share natural=3, share >=2, meaning kept, answer ok,
agreement between reviewers, Cohen's kappa on same/answer_ok) and runs_audit/clean_ids.txt: the test questions that
BOTH reviewers rated natural >= 2, same = y and answer_ok = y (all ArDQA test questions are kept as native).
Then:  python scripts/subset_results.py --keep runs_audit/clean_ids.txt
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
FIELDS = ["row", "question_msa", "question_egy", "reference_answer", "passage", "natural (1-3)", "same (y/n)",
          "answer_ok (y/n)", "notes"]


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def make(a):
    test = load(a.test)
    ids = None
    if Path(a.per_question).exists():
        with open(a.per_question, encoding="utf-8-sig", newline="") as f:
            ids = {r["id"] for r in csv.DictReader(f)}
    test = [r for r in test if ids is None or r["id"] in ids]
    passages = {}
    need = {p for r in test for p in r.get("gold_pids") or []}
    with (Path(a.corpus) / "passages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p["id"] in need:
                    passages[p["id"]] = p["text"]
    rng = random.Random(a.seed)
    ard = [r for r in test if r["set"] == "ardqa"]
    rows = [r for r in test if r["set"] in ("pilot", "heldout")] + rng.sample(ard, min(a.control, len(ard)))
    key = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(1, a.reviewers + 1):
        order = rows[:]
        random.Random(a.seed + k).shuffle(order)
        with (out / f"review_dialect_R{k}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for i, r in enumerate(order, 1):
                ptxt = " ".join(passages.get(p, "") for p in r.get("gold_pids") or [])[:900]
                w.writerow([f"R{k}-{i:04d}", r["q_msa"], r["q_egy"], " | ".join(r.get("refs") or [r.get("answer") or ""]),
                            ptxt, "", "", "", ""])
                key.append({"row": f"R{k}-{i:04d}", "id": r["id"], "set": r["set"]})
    with (out / "review_dialect_key.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["row", "id", "set"])
        w.writeheader()
        w.writerows(key)
    print(f"{len(rows)} rows per reviewer ({sum(r['set'] == 'pilot' for r in rows)} pilot, "
          f"{sum(r['set'] == 'heldout' for r in rows)} held-out, {min(a.control, len(ard))} ArDQA control)")
    print(f"written to {out}: review_dialect_R1..R{a.reviewers}.csv (give out) and review_dialect_key.csv (keep)")


def yn(v):
    v = str(v).strip().lower()
    return True if v in ("y", "yes", "1", "نعم") else False if v in ("n", "no", "0", "لا") else None


def kappa(x, y):
    n = len(x)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    p1, p2 = sum(x) / n, sum(y) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def summarize(a):
    out = Path(a.out)
    key = {r["row"]: r for r in csv.DictReader(open(out / "review_dialect_key.csv", encoding="utf-8-sig"))}
    per = defaultdict(dict)                          # id -> reviewer -> ratings
    sets = {}
    for f in sorted(out.glob("review_dialect_R*.csv")):
        rev = f.stem.split("_")[-1]
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            k = key[r["row"]]
            try:
                nat = int(float(r["natural (1-3)"]))
            except ValueError:
                nat = None
            per[k["id"]][rev] = {"nat": nat, "same": yn(r["same (y/n)"]), "ans": yn(r["answer_ok (y/n)"])}
            sets[k["id"]] = k["set"]
    revs = sorted({rv for d in per.values() for rv in d})
    missing = [i for i, d in per.items() for rv in revs if rv not in d or None in d[rv].values()]
    if missing:
        print(f"WARNING: {len(missing)} ratings missing or not in the allowed values (e.g. {missing[:5]}); those rows count as failing")
    L = ["NATIVE-SPEAKER REVIEW OF THE EGYPTIAN TEST QUESTIONS", f"reviewers: {', '.join(revs)}", ""]
    clean = []
    for part in ("pilot", "heldout", "ardqa"):
        ids = [i for i in per if sets[i] == part]
        if not ids:
            continue
        def share(f):
            v = [f(per[i][rv]) for i in ids for rv in revs if rv in per[i]]
            return f"{100 * sum(bool(x) for x in v) / max(1, len(v)):5.1f}%"
        L.append(f"{part} ({len(ids)} questions{', native control' if part == 'ardqa' else ''})")
        L.append(f"  natural = 3 {share(lambda d: d['nat'] == 3)}   natural >= 2 {share(lambda d: (d['nat'] or 0) >= 2)}"
                 f"   meaning kept {share(lambda d: d['same'])}   reference answer ok {share(lambda d: d['ans'])}")
        if len(revs) >= 2:
            r1, r2 = revs[:2]
            both = [i for i in ids if all(rv in per[i] and None not in per[i][rv].values() for rv in (r1, r2))]
            for fld, lab in (("same", "meaning kept"), ("ans", "answer ok")):
                x = [int(per[i][r1][fld]) for i in both]
                y = [int(per[i][r2][fld]) for i in both]
                agree = sum(p == q for p, q in zip(x, y)) / max(1, len(x))
                L.append(f"  agreement on {lab}: {100 * agree:.1f}%, kappa {kappa(x, y):.2f}")
        ok = [i for i in ids if all(rv in per[i] and (per[i][rv]["nat"] or 0) >= 2 and per[i][rv]["same"] and per[i][rv]["ans"]
                                    for rv in revs)]
        L.append(f"  passed by every reviewer (natural >= 2, meaning kept, answer ok): {len(ok)} of {len(ids)}")
        L.append("")
        if part != "ardqa":
            clean += ok
    text = "\n".join(L)
    print(text)
    (out / "dialect_review_summary.txt").write_text(text + "\n", encoding="utf-8")
    (out / "clean_ids.txt").write_text("\n".join(sorted(clean)) + "\n", encoding="utf-8")
    print(f"clean pilot/held-out ids written to {out / 'clean_ids.txt'} ({len(clean)}); ArDQA test questions are always kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "summarize"])
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--per_question", default="runs_final/per_question.csv")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--reviewers", type=int, default=2)
    ap.add_argument("--control", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="runs_audit")
    a = ap.parse_args()
    make(a) if a.mode == "make" else summarize(a)


if __name__ == "__main__":
    main()
