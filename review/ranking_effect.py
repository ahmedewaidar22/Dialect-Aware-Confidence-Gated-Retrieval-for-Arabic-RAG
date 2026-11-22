"""Effect of judge errors on the reported results, using human labels on the multi-system sample.

Inputs: annotation_multi_key.csv (private key), the two filled annotator sheets, and optional adjudicated labels.
Final human label per item = annotator A if A == B, else the `--adjudicated` file (item,label) if given, else the item
is listed as unresolved and skipped.

  python ranking_effect.py --key annotation_multi_key.csv --a annotation_multi_A.csv --b annotation_multi_B.csv \
      [--adjudicated adjudicated.csv] [--reference final] --out ranking_effect

Reports (printed and written to ranking_effect_*.csv):
 1. Inter-annotator agreement: raw, Cohen's kappa (+ bootstrap 95% CI), PABAK.
 2. Per system, on the sample: judge accuracy, human accuracy, gap, false-accept rate (judge correct, human wrong) and
    false-reject rate (judge wrong, human correct), with bootstrap-over-questions 95% CIs on the gap.
 3. Ranking: systems sorted by judge accuracy and by human accuracy, Kendall tau, and every pair whose order flips.
 4. For the reference system vs each other system: paired accuracy difference under judge and under human labels
    (bootstrap-over-questions CI and exact McNemar p) so you can say whether each conclusion survives.
 5. Bias check: is the judge-human gap different between systems? (max gap difference with bootstrap CI)
Note: the sample is stratified (mixed questions over-represented), so sample accuracies are NOT the population numbers;
compare judge vs human on the SAME sample. Report the bias, do not replace the paper's accuracies with sample values.
"""
import argparse, csv, random
from collections import defaultdict
from math import comb


def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nid(x):
    """Item ids may come back as 71, 071 or 00071 after Excel; compare them as integers."""
    x = str(x).strip()
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def nlab(x):
    """Labels may come back as 1, 1.0, '1' or blank."""
    x = str(x).strip()
    try:
        v = float(x)
    except ValueError:
        return ""
    return str(int(v)) if v in (0.0, 1.0) else ""


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    p1x, p1y = sum(x) / n, sum(y) / n
    pe = p1x * p1y + (1 - p1x) * (1 - p1y)
    return po, (1.0 if pe == 1 else (po - pe) / (1 - pe)), 2 * po - 1


def mcnemar_p(b_, c_):
    n = b_ + c_
    if n == 0:
        return 1.0
    k = min(b_, c_)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def kendall(rank_a, rank_b):
    names = list(rank_a)
    c = d = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s = (rank_a[names[i]] - rank_a[names[j]]) * (rank_b[names[i]] - rank_b[names[j]])
            c += s > 0
            d += s < 0
    t = c + d
    return (c - d) / t if t else float("nan")


def ci(vals):
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--adjudicated", default="")
    ap.add_argument("--reference", default="final")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ranking_effect")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    A = {nid(r["item"]): nlab(r["label"]) for r in read(a.a)}
    B = {nid(r["item"]): nlab(r["label"]) for r in read(a.b)}
    adj = {nid(r["item"]): nlab(r["label"]) for r in read(a.adjudicated)} if a.adjudicated else {}
    adj = {k: v for k, v in adj.items() if v != ""}
    bad = [i for i in A if A[i] not in {"0", "1"} or B.get(i) not in {"0", "1"}]
    if bad:
        raise SystemExit(f"{len(bad)} items have a missing or non 0/1 label in A or B (e.g. {bad[:5]})")
    items = list(A)
    xa, xb = [int(A[i]) for i in items], [int(B[i]) for i in items]
    po, k, pabak = kappa(xa, xb)
    kb = []
    for _ in range(a.boot):
        idx = [rng.randrange(len(items)) for _ in items]
        kb.append(kappa([xa[i] for i in idx], [xb[i] for i in idx])[1])
    lo, hi = ci(kb)
    print(f"1. Annotators: n={len(items)} raw agreement {po:.3f}, kappa {k:.3f} [{lo:.3f}, {hi:.3f}], PABAK {pabak:.3f}")

    human, unresolved = {}, []
    for i in items:
        if A[i] == B[i]:
            human[i] = int(A[i])
        elif i in adj:
            human[i] = int(adj[i])
        else:
            unresolved.append(i)
    if unresolved:
        print(f"   {len(unresolved)} disagreements have no adjudicated label and are skipped (write them to --adjudicated)")

    key = [{**r, "item": nid(r["item"])} for r in read(a.key)]
    key = [r for r in key if r["item"] in human]
    systems = sorted({r["system"] for r in key})
    q_of = defaultdict(dict)
    for r in key:
        q_of[r["qid"]][r["system"]] = (int(r["judge"]), human[r["item"]])
    qids = [q for q, d in q_of.items() if len(d) == len(systems)]
    print(f"   {len(qids)} questions with every system labelled")

    def stats(qs):
        out = {}
        for s in systems:
            j = [q_of[q][s][0] for q in qs]
            h = [q_of[q][s][1] for q in qs]
            n = len(qs)
            out[s] = {"judge": sum(j) / n, "human": sum(h) / n,
                      "fa": sum(1 for x, y in zip(j, h) if x and not y) / max(1, sum(j)),
                      "fr": sum(1 for x, y in zip(j, h) if not x and y) / max(1, n - sum(j))}
        return out

    base = stats(qids)
    boots = [stats([qids[rng.randrange(len(qids))] for _ in qids]) for _ in range(a.boot)]
    print("\n2. Per system on the sample (judge vs human)")
    print(f"{'system':22s}{'judge':>8s}{'human':>8s}{'gap':>8s}{'gap 95% CI':>18s}{'false-acc':>10s}{'false-rej':>10s}")
    rows2 = []
    for s in systems:
        g = [bs[s]["judge"] - bs[s]["human"] for bs in boots]
        l, h_ = ci(g)
        d = base[s]
        print(f"{s:22s}{100*d['judge']:8.1f}{100*d['human']:8.1f}{100*(d['judge']-d['human']):8.1f}{f'[{100*l:.1f}, {100*h_:.1f}]':>18s}{100*d['fa']:10.1f}{100*d['fr']:10.1f}")
        rows2.append({"system": s, **{k_: round(v, 4) for k_, v in d.items()}, "gap_lo": round(l, 4), "gap_hi": round(h_, 4)})
    with open(f"{a.out}_per_system.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows2[0]))
        w.writeheader()
        w.writerows(rows2)

    rj = {s: r for r, s in enumerate(sorted(systems, key=lambda s: -base[s]["judge"]))}
    rh = {s: r for r, s in enumerate(sorted(systems, key=lambda s: -base[s]["human"]))}
    print(f"\n3. Ranking: Kendall tau (judge vs human order) = {kendall(rj, rh):.3f}")
    print("   judge order:", " > ".join(sorted(systems, key=lambda s: rj[s])))
    print("   human order:", " > ".join(sorted(systems, key=lambda s: rh[s])))
    flips = [(x, y) for x in systems for y in systems if x < y and (base[x]["judge"] - base[y]["judge"]) * (base[x]["human"] - base[y]["human"]) < 0]
    print("   pairs whose order flips:", flips if flips else "none")

    if a.reference in systems:
        print(f"\n4. {a.reference} vs each system (accuracy difference, percentage points)")
        print(f"{'vs':22s}{'judge diff':>12s}{'human diff':>12s}{'human 95% CI':>20s}{'human McNemar p':>17s}  conclusion")
        rows4 = []
        for s in systems:
            if s == a.reference:
                continue
            dj = sum(q_of[q][a.reference][0] - q_of[q][s][0] for q in qids) / len(qids)
            dh = sum(q_of[q][a.reference][1] - q_of[q][s][1] for q in qids) / len(qids)
            bs = []
            for _ in range(a.boot):
                qs = [qids[rng.randrange(len(qids))] for _ in qids]
                bs.append(sum(q_of[q][a.reference][1] - q_of[q][s][1] for q in qs) / len(qs))
            l, h_ = ci(bs)
            b_ = sum(1 for q in qids if q_of[q][a.reference][1] and not q_of[q][s][1])
            c_ = sum(1 for q in qids if not q_of[q][a.reference][1] and q_of[q][s][1])
            p = mcnemar_p(b_, c_)
            verdict = "tie under human labels" if dh == 0 else ("direction kept" if (dj > 0) == (dh > 0) else "DIRECTION CHANGES")
            print(f"{s:22s}{100*dj:12.1f}{100*dh:12.1f}{f'[{100*l:.1f}, {100*h_:.1f}]':>20s}{p:17.3f}  {verdict}")
            rows4.append({"vs": s, "judge_diff": round(dj, 4), "human_diff": round(dh, 4), "human_lo": round(l, 4), "human_hi": round(h_, 4), "mcnemar_p": round(p, 4)})
        with open(f"{a.out}_vs_reference.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, list(rows4[0]))
            w.writeheader()
            w.writerows(rows4)

    print("\n5. Is the judge's bias the same for every system?")
    gaps = {s: base[s]["judge"] - base[s]["human"] for s in systems}
    mx = max(gaps, key=gaps.get)
    mn = min(gaps, key=gaps.get)
    diffs = [bs[mx]["judge"] - bs[mx]["human"] - (bs[mn]["judge"] - bs[mn]["human"]) for bs in boots]
    l, h_ = ci(diffs)
    print(f"   largest gap {mx} {100*gaps[mx]:.1f} pts, smallest {mn} {100*gaps[mn]:.1f} pts; difference {100*(gaps[mx]-gaps[mn]):.1f} [{100*l:.1f}, {100*h_:.1f}]")
    print("   (an interval that excludes 0 means the judge favours some systems over others)")


if __name__ == "__main__":
    main()
