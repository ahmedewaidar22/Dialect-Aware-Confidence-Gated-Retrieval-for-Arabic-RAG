"""Results of the 1,000-question test restricted to a subset, from saved per-question outcomes (no API calls).

  python scripts/subset_results.py                                   # native ArDQA only, then each part separately
  python scripts/subset_results.py --keep runs_audit/clean_ids.txt   # ArDQA + the pilot/held-out questions that passed review
  python scripts/subset_results.py --drop runs_audit/leaked_ids.txt  # without questions the provenance audit confirmed as leaks

Reads runs_final/per_question.csv (written by final_report.py). For each subset it prints every system's accuracy with
a Wilson 95% interval and cost, and the final system against standard RAG and against always C2 (paired bootstrap 95%
interval and exact McNemar p).
"""
import argparse
import csv
import random
from math import comb, sqrt
from pathlib import Path

FINAL = "FINAL: rewrite + C1 on reranked top-k, 'I don't know' -> C2"
STD = "Standard RAG (C1, hybrid top-5)"
C2 = "Iterative RAG (always C2)"
SHOW = [STD, C2, "Adaptive-RAG (text classifier)", "Router v1 (quality)", FINAL]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(x, y, reps=4000, seed=0):
    d = [p - q for p, q in zip(x, y)]
    rng = random.Random(seed)
    n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    b = sum(1 for p, q in zip(x, y) if p and not q)
    c = sum(1 for p, q in zip(x, y) if q and not p)
    return sum(d) / n, bs[int(0.025 * reps)], bs[int(0.975 * reps)], mcnemar(b, c)


def report(name, rows):
    n = len(rows)
    L = [f"{name}: {n} questions"]
    if not n:
        return L + [""]
    cols = [s for s in SHOW if f"{s} | correct" in rows[0]]
    H = {s: [int(r[f"{s} | correct"]) for r in rows] for s in cols}
    for s in cols:
        k = sum(H[s])
        lo, hi = wilson(k, n)
        usd = 1000 * sum(float(r[f"{s} | usd"]) for r in rows) / n
        L.append(f"  {s[:44]:44s} {100 * k / n:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  US$/1k {usd:.3f}")
    for base in (STD, C2):
        if base in H and FINAL in H:
            d, lo, hi, p = paired(H[FINAL], H[base])
            L.append(f"  final vs {base[:30]:30s} {100 * d:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]  McNemar p = {p:.4f}")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_question", default="runs_final/per_question.csv")
    ap.add_argument("--keep", default="")
    ap.add_argument("--drop", default="")
    a = ap.parse_args()
    with open(a.per_question, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    read = lambda p: {x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()}   # noqa: E731
    if a.drop:
        drop = read(a.drop)
        rows = [r for r in rows if r["id"] not in drop]
        print(f"dropped {len(drop)} ids listed in {a.drop}")
    L = []
    if a.keep:
        keep = read(a.keep)
        sub = [r for r in rows if r["set"] == "ardqa" or r["id"] in keep]
        L += report("NATIVE ArDQA + REVIEWED pilot/held-out questions", sub)
    L += report("NATIVE ONLY (ArDQA)", [r for r in rows if r["set"] == "ardqa"])
    for part in ("pilot", "heldout"):
        L += report(f"{part} only", [r for r in rows if r["set"] == part])
    L += report("ALL", rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
