"""Step 4: the pre-specified tests, run once on runs_confirm/results.jsonl.

  python scripts/analyze_confirmatory.py [--margin 5] [--lang egy]

H1  final more accurate than standard RAG          exact McNemar test (two-sided), alpha 0.05, plus paired bootstrap CI
H2  final cheaper than always C2                   paired per-question cost difference, 95% bootstrap CI must lie below 0
H3  final not worse than always C2 by > margin     one-sided: lower end of the 95% paired-bootstrap CI of
                                                    (final - always_c2) must be above -margin points
Prints PASS / FAIL for each hypothesis and the numbers whatever the outcome. Also prints accuracy with Wilson CIs and
how often the final system escalated. Report all of it, including a failure.
"""
import argparse, json, random
from math import comb, sqrt
from pathlib import Path


def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def boot(vals, reps=5000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)], ms[int(0.05 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs_confirm/results.jsonl")
    ap.add_argument("--lang", default="egy")
    ap.add_argument("--margin", type=float, default=5.0)
    a = ap.parse_args()
    R = [json.loads(x) for x in Path(a.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    names = ["standard_rag", "always_c2", "final"]
    R = [r for r in R if all(f"{s}|{a.lang}" in r["systems"] for s in names)]
    n = len(R)
    print(f"{n} questions, input = {a.lang}")
    H = {s: [int(r["systems"][f"{s}|{a.lang}"]["correct"]) for r in R] for s in names}
    C = {s: [r["systems"][f"{s}|{a.lang}"]["usd"] for r in R] for s in names}
    for s in names:
        k = sum(H[s])
        lo, hi = wilson(k, n)
        print(f"  {s:13s} accuracy {100*k/n:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]   US$ per 1k questions {1000*sum(C[s])/n:.3f}")
    esc = sum(r["systems"][f"final|{a.lang}"].get("escalated", False) for r in R)
    print(f"  final escalated to C2 on {100*esc/n:.1f}% of questions")

    f, s_, c2 = H["final"], H["standard_rag"], H["always_c2"]
    b = sum(1 for x, y in zip(f, s_) if x and not y)
    c = sum(1 for x, y in zip(f, s_) if not x and y)
    d = [x - y for x, y in zip(f, s_)]
    lo, hi, _ = boot(d)
    p = mcnemar_p(b, c)
    print(f"\nH1 final vs standard RAG: {100*sum(d)/n:+.1f} points [{100*lo:+.1f}, {100*hi:+.1f}], McNemar p = {p:.4f} "
          f"(final-only correct {b}, standard-only correct {c})  -> {'PASS' if p < 0.05 and sum(d) > 0 else 'FAIL'}")
    dc = [x - y for x, y in zip(C["final"], C["always_c2"])]
    lo, hi, _ = boot(dc)
    print(f"H2 cost, final minus always C2: US$ {1000*sum(dc)/n:+.3f} per 1k [{1000*lo:+.3f}, {1000*hi:+.3f}]  -> {'PASS' if hi < 0 else 'FAIL'}")
    d3 = [x - y for x, y in zip(f, c2)]
    lo, hi, lo1 = boot(d3)
    print(f"H3 final vs always C2: {100*sum(d3)/n:+.1f} points, one-sided 95% lower bound {100*lo1:+.1f} "
          f"(margin -{a.margin:.0f})  -> {'PASS' if 100*lo1 > -a.margin else 'FAIL'}")
    print("\nReport all three lines whatever they show. Any change to the pipeline after the freeze voids this test.")


if __name__ == "__main__":
    main()
