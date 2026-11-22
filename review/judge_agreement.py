"""Agreement between two human annotators and between the LLM judge and the human consensus.

Input CSV (UTF-8) with columns:  id, ann1, ann2, judge            labels 1/0 (or correct/incorrect, yes/no)
optional columns:                adjudicated (final label where ann1 != ann2), set, question, reference, answer

  python scripts/judge_agreement.py --csv merged.csv --out agreement_report

Prints and writes  <out>.md  (the numbers for the paper) and  <out>_disagreements.csv  (rows to read for the error
analysis: judge wrong, or annotators disagreed). Cohen's kappa gets a bootstrap 95% interval. With skewed labels
kappa can be low although raw agreement is high, so raw agreement and prevalence-adjusted kappa (PABAK) are reported.
Final human label = the annotators' label if they agree, else the adjudicated one (rows still unresolved are dropped
and counted).
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def lab(v):
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "correct", "y"}:
        return 1
    if s in {"0", "false", "no", "incorrect", "n"}:
        return 0
    return None


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(a, b, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([a[i] for i in idx], [b[i] for i in idx])
        if k == k:
            vals.append(k)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (float("nan"),) * 2


def landis(k):
    for lim, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= lim:
            return name
    return "almost perfect"


def stats(rows):
    L = []
    a1, a2 = [r["a1"] for r in rows], [r["a2"] for r in rows]
    po = sum(x == y for x, y in zip(a1, a2)) / len(rows)
    k = kappa(a1, a2)
    lo, hi = boot_ci(a1, a2)
    L.append(f"Annotator 1 vs 2 (n={len(rows)}): raw agreement {100 * po:.1f}%, Cohen's kappa {k:.3f} "
             f"[{lo:.3f}, {hi:.3f}] ({landis(k)}), PABAK {2 * po - 1:.3f}")
    fin = [r for r in rows if r["final"] is not None]
    L.append(f"Resolved final labels: {len(fin)} of {len(rows)} ({len(rows) - len(fin)} disagreements unresolved)")
    h, j = [r["final"] for r in fin], [r["judge"] for r in fin]
    po = sum(x == y for x, y in zip(h, j)) / len(fin)
    k = kappa(h, j)
    lo, hi = boot_ci(h, j)
    tp = sum(x == 1 and y == 1 for x, y in zip(h, j))
    fp = sum(x == 0 and y == 1 for x, y in zip(h, j))
    fn = sum(x == 1 and y == 0 for x, y in zip(h, j))
    tn = sum(x == 0 and y == 0 for x, y in zip(h, j))
    L.append(f"LLM judge vs human final label: agreement {100 * po:.1f}%, Cohen's kappa {k:.3f} [{lo:.3f}, {hi:.3f}] "
             f"({landis(k)})")
    L.append(f"  judge says correct and human says wrong (false accept): {fp} of {fp + tn} human-wrong answers "
             f"({100 * fp / max(1, fp + tn):.1f}%)")
    L.append(f"  judge says wrong and human says correct (false reject): {fn} of {fn + tp} human-correct answers "
             f"({100 * fn / max(1, fn + tp):.1f}%)")
    L.append(f"  accuracy by judge {100 * sum(j) / len(j):.1f}% vs by humans {100 * sum(h) / len(h):.1f}% "
             f"(a gap here biases every reported accuracy)")
    return L, fin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="agreement_report")
    a = ap.parse_args(argv)
    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        a1, a2, j = lab(r.get("ann1")), lab(r.get("ann2")), lab(r.get("judge"))
        if a1 is None or a2 is None or j is None:
            continue
        adj = lab(r.get("adjudicated", ""))
        final = a1 if a1 == a2 else adj
        rows.append({**r, "a1": a1, "a2": a2, "judge": j, "final": final})
    if not rows:
        raise SystemExit("no complete rows (need ann1, ann2, judge)")
    L = [f"{len(raw)} rows read, {len(rows)} complete"]
    s, fin = stats(rows)
    L += s
    by = defaultdict(list)
    for r in fin:
        by[r.get("set") or "all"].append(r)
    if len(by) > 1:
        L.append("By test part (judge vs human final label):")
        for name, rs in sorted(by.items()):
            ag = sum(r["final"] == r["judge"] for r in rs) / len(rs)
            L.append(f"  {name}: n={len(rs)}, agreement {100 * ag:.1f}%, kappa {kappa([r['final'] for r in rs], [r['judge'] for r in rs]):.3f}")
    err = [r for r in rows if r["a1"] != r["a2"] or (r["final"] is not None and r["final"] != r["judge"])]
    cause = Counter("annotators disagree" if r["a1"] != r["a2"] else
                    ("judge accepted a wrong answer" if r["judge"] == 1 else "judge rejected a correct answer") for r in err)
    L.append("Rows for error analysis: " + ", ".join(f"{k}: {v}" for k, v in cause.items()))
    text = "\n".join(L)
    Path(a.out + ".md").write_text(text + "\n", encoding="utf-8")
    keep = ["id", "set", "question", "reference", "answer", "ann1", "ann2", "adjudicated", "judge"]
    with open(a.out + "_disagreements.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(err)
    print(text)


if __name__ == "__main__":
    main()
