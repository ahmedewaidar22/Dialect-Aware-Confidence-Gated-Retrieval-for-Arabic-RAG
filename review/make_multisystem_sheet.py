"""Blind annotation sheets that cover ALL systems on the SAME sampled questions.

Why: judge bias can differ by answer style (a system that answers verbosely or hedges may be scored differently), so the
judge has to be checked per system, and the system ranking has to be re-computed with human labels.

Input (CSV or JSONL), one row per (question, system) with columns/fields:
  qid, system, question, reference, answer, judge (1/0), set (optional)
Pick the systems with --systems (default: all found). Identical answers to one question (after light normalisation) are
shown to the annotators ONCE, so cost is the number of UNIQUE answers, printed before anything is written (use --dry-run).

  python make_multisystem_sheet.py --input graded_long.csv --systems "standard_rag,always_c2,adaptive_rag,router_v1_quality,final" \
      --n-questions 100 --out annotation_multi --dry-run
  python make_multisystem_sheet.py ... --out annotation_multi            # writes the files

Questions are stratified by (set, agreement bucket): all selected systems judged correct / all incorrect / mixed. Mixed
questions are where the ranking is decided, so they are guaranteed at least half the sample when available.

Writes annotation_multi_A.csv and _B.csv (same rows, different random order, no judge verdict, no system names),
annotation_multi_key.csv (private: item id -> qid, system, judge verdict).
Annotators fill `label` with 1 (answer states the reference fact; paraphrase/transliteration ok; numbers/dates must
match) or 0 (wrong, vague, or "I don't know").
"""
import argparse, csv, json, random, re, sys
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def b(v):
    return v.strip().lower() in {"1", "true", "yes", "correct"} if isinstance(v, str) else bool(v)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s or ""))).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--systems", default="")
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="annotation_multi")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = read_rows(a.input)
    systems = [s for s in a.systems.split(",") if s] or sorted({r["system"] for r in rows})
    by_q = defaultdict(dict)
    meta = {}
    for r in rows:
        if r["system"] in systems:
            by_q[r["qid"]][r["system"]] = r
            meta[r["qid"]] = r
    qids = [q for q, d in by_q.items() if len(d) == len(systems)]
    if len(qids) < len(by_q):
        print(f"note: {len(by_q) - len(qids)} questions lack answers from every selected system and are skipped")

    def bucket(q):
        v = {b(by_q[q][s]["judge"]) for s in systems}
        return "all_correct" if v == {True} else "all_wrong" if v == {False} else "mixed"

    rng = random.Random(a.seed)
    strata = defaultdict(list)
    for q in qids:
        strata[(meta[q].get("set", ""), bucket(q))].append(q)
    for g in strata.values():
        rng.shuffle(g)
    mixed = [g for k, g in strata.items() if k[1] == "mixed"]
    others = [g for k, g in strata.items() if k[1] != "mixed"]
    chosen = []
    half = a.n_questions // 2
    for pool, quota in ((mixed, half), (others, a.n_questions - half)):
        per = max(1, quota // max(1, len(pool)))
        for g in pool:
            chosen += g[:per]
    left = [q for g in mixed + others for q in g if q not in set(chosen)]
    rng.shuffle(left)
    chosen += left[:max(0, a.n_questions - len(chosen))]
    chosen = chosen[:a.n_questions]

    items, key = [], []
    for q in chosen:
        seen = {}
        for s in systems:
            r = by_q[q][s]
            k = norm(r["answer"])
            if k not in seen:
                seen[k] = f"{len(items):05d}"
                items.append({"item": seen[k], "question": r["question"], "reference": r["reference"], "answer": r["answer"]})
            key.append({"item": seen[k], "qid": q, "system": s, "judge": int(b(r["judge"])), "set": r.get("set", "")})
    print(f"{len(chosen)} questions x {len(systems)} systems = {len(chosen)*len(systems)} answers -> {len(items)} unique answers to grade")
    print("strata:", {f'{k[0]}/{k[1]}': len(v) for k, v in strata.items()})
    if a.dry_run:
        return
    for tag, seed in (("A", 1), ("B", 2)):
        order = items[:]
        random.Random(a.seed + seed).shuffle(order)
        with open(f"{a.out}_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, ["item", "question", "reference", "answer", "label"])
            w.writeheader()
            for it in order:
                w.writerow({**it, "label": ""})
    with open(f"{a.out}_key.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["item", "qid", "system", "judge", "set"])
        w.writeheader()
        w.writerows(key)
    print(f"wrote {a.out}_A.csv, {a.out}_B.csv, {a.out}_key.csv")


if __name__ == "__main__":
    main()
