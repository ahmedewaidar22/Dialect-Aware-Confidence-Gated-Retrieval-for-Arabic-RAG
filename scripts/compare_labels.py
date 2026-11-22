"""Compare two outcome-label runs on the questions both labelled (e.g. GPT-4.1 mini vs GPT-6.1 Sol).

  python scripts/compare_labels.py main_mini sens_sol
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
LABELS = ["C0", "C1", "C2", "none"]


def load(run, root="data/labels"):
    f = Path(root) / run / "labels.jsonl"
    return {r["id"]: r for r in (json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip())}


def main(a, b, root="data/labels"):
    A, B = load(a, root), load(b, root)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("no questions in common")
    print(f"{a} vs {b}: {len(common):,} questions in common")
    agree = sum(A[i]["label"] == B[i]["label"] for i in common)
    print(f"same label: {agree:,} ({100 * agree / len(common):.1f}%)\n")
    print(f"rows = {a}, columns = {b}")
    print(f"{'':<8}" + "".join(f"{lab:>8}" for lab in LABELS))
    cm = Counter((A[i]["label"], B[i]["label"]) for i in common)
    for ra in LABELS:
        print(f"{ra:<8}" + "".join(f"{cm[(ra, cb)]:>8}" for cb in LABELS))
    print("\nstrategy accuracy on questions where BOTH runs ran that strategy:")
    for st in ("C0", "C1", "C2"):
        both = [i for i in common if st in A[i]["results"] and st in B[i]["results"]]
        if both:
            ca = sum(A[i]["results"][st]["correct"] for i in both)
            cb = sum(B[i]["results"][st]["correct"] for i in both)
            print(f"  {st}: n={len(both):>5} | {a}: {100 * ca / len(both):5.1f}% | {b}: {100 * cb / len(both):5.1f}%")
    print("\nlabel distribution by source (%):")
    for src in sorted({A[i]["source"] for i in common}):
        ids = [i for i in common if A[i]["source"] == src]
        for name, R in ((a, A), (b, B)):
            c = Counter(R[i]["label"] for i in ids)
            print(f"  {src:<14}{name:<12}" + "".join(f"{lab} {100 * c[lab] / len(ids):5.1f}%  " for lab in LABELS))
    return agree / len(common)


if __name__ == "__main__":
    main(*sys.argv[1:3])
