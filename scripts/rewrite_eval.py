"""Phase 6 (RQ4): how much of the Egyptian-vs-MSA retrieval gap does Egyptian->MSA rewriting close?

  python scripts/rewrite_eval.py --budget_usd 0.5      # LLM rewrites ~$0.10 (cached), retrieval local (~10 min GPU)

Every Egyptian test question (data/egy_test/egy_test.jsonl) is searched five ways:
  egy        the Egyptian question as is                       (what standard RAG sees)
  egy>llm    rewritten to MSA by the LLM                       (the rewriter)
  egy>rules  rewritten by simple word substitution             (free baseline)
  msa        the original MSA twin                             (best case)
  msa>llm    the MSA twin passed through the rewriter anyway   (cost of rewriting everything)
Writes runs_rewrite/rewrite_report.txt/.json, rewrites.jsonl, ranks.jsonl, rewrite_review.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.egy.rewrite import check_rewrite, rewrite_llm, rewrite_rules     # noqa: E402

CONDS = ["egy", "egy>llm", "egy>rules", "msa", "msa>llm"]
METHODS = ["bm25", "dense", "hybrid"]
KS = (1, 5, 10, 20)
SETS = ["ardqa", "pilot", "heldout", "ALL"]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: returns (only_a, only_b, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    n = x + y
    if n == 0:
        return x, y, 1.0
    tail = sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, 2 * tail)


def closed_ci(egy, rew, msa, reps=2000, seed=0):
    """Share of the gap closed = (rew - egy) / (msa - egy), with a paired bootstrap 95% CI."""
    egy, rew, msa = (np.asarray(v, float) for v in (egy, rew, msa))
    gap = msa.mean() - egy.mean()
    point = (rew.mean() - egy.mean()) / gap if abs(gap) > 1e-9 else float("nan")
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(reps):
        s = rng.randint(0, len(egy), len(egy))
        g = msa[s].mean() - egy[s].mean()
        if abs(g) > 1e-9:
            vals.append((rew[s].mean() - egy[s].mean()) / g)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) > 50 else (float("nan"), float("nan")))
    return float(point), float(lo), float(hi)


def first_hit(ranked, gold):
    for r, p in enumerate(ranked):
        if p in gold:
            return r + 1
    return 10 ** 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="runs_rewrite")
    ap.add_argument("--budget_usd", type=float, default=0.5)
    ap.add_argument("--price", action="append", default=["gpt-4.1-mini=0.40,1.60"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--n_review", type=int, default=50)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    from arroute.llm import Budget, complete_with_usage
    from label_outcomes import parse_prices
    model = os.environ.get("ARROUTE_LLM_MODEL")
    if not model:
        raise SystemExit("set ARROUTE_LLM_MODEL (e.g. gpt-4.1-mini)")
    budget = Budget(a.budget_usd, parse_prices(a.price))
    budget.price(model)
    gen = {"fn": complete_with_usage, "model": model, "budget": budget}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [it for it in items if "rejected" not in (it.get("review") or {}).values()]
    print(f"[1/3] rewriting {len(items)} Egyptian questions and their MSA twins (LLM, cached) ...")
    with ThreadPoolExecutor(a.workers) as pool:
        egy_llm = list(pool.map(lambda it: rewrite_llm(it["q_egy"], gen), items))
        msa_llm = list(pool.map(lambda it: rewrite_llm(it["q_msa"], gen), items))
    queries = {"egy": [it["q_egy"] for it in items], "egy>llm": egy_llm,
               "egy>rules": [rewrite_rules(it["q_egy"]) for it in items], "msa": [it["q_msa"] for it in items],
               "msa>llm": msa_llm}
    with (out / "rewrites.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": it["id"], "set": it["set"], **{c: queries[c][i] for c in CONDS},
                                "flags_llm": check_rewrite(it["q_egy"], egy_llm[i], it.get("answer")),
                                "flags_rules": check_rewrite(it["q_egy"], queries["egy>rules"][i], it.get("answer"))},
                               ensure_ascii=False) + "\n")
    print(f"      done ({time.time() - t0:.0f}s, ${budget.spent:.3f})")

    print("[2/3] retrieval: 5 versions x 3 retrievers ...")
    from arroute.retrieval.index import load_retriever
    retr = load_retriever(a.corpus, dense=True, device=a.device)
    ranks = {c: {m: np.zeros(len(items), int) for m in METHODS} for c in CONDS}
    for c in CONDS:
        for i, (q, it) in enumerate(zip(queries[c], items)):
            res = retr.all_methods(q, k=max(KS))
            for m in METHODS:
                ranks[c][m][i] = first_hit(res[m], set(it["gold_pids"]))
        print(f"      {c:<10} done ({time.time() - t0:.0f}s)")
    with (out / "ranks.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": it["id"], "set": it["set"],
                                **{f"{c}|{m}": int(ranks[c][m][i]) for c in CONDS for m in METHODS}}) + "\n")

    print("[3/3] report ...")
    sets = np.array([it["set"] for it in items])
    masks = {s: (sets == s) if s != "ALL" else np.ones(len(items), bool) for s in SETS}
    L = [f"PHASE 6 - EGYPTIAN -> MSA REWRITING (RQ4)   {len(items):,} questions, LLM spend ${budget.spent:.3f}",
         "versions: egy = as typed | egy>llm = rewritten by the LLM | egy>rules = word substitution | "
         "msa = original MSA twin | msa>llm = MSA passed through the rewriter", ""]
    js = {"recall": {}, "gap": {}, "harm": {}, "diagnostics": {}}
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        L.append(f"RECALL - {s} (n={int(m_.sum())})   R@1 / R@5 / R@10 / R@20")
        L.append(f"  {'version':<11}" + "".join(f"{m:>26}" for m in METHODS))
        for c in CONDS:
            cells = []
            for m in METHODS:
                r = ranks[c][m][m_]
                vals = [float((r <= k).mean()) for k in KS]
                js["recall"].setdefault(s, {}).setdefault(c, {})[m] = dict(zip(map(str, KS), vals))
                cells.append(" / ".join(f"{100 * v:4.1f}" for v in vals))
            L.append(f"  {c:<11}" + "".join(f"{x:>26}" for x in cells))
        L.append("")

    L.append("THE DIALECT GAP AND HOW MUCH THE REWRITER CLOSES (hit within top-k; k=5 is what C1 reads)")
    L.append(f"  {'set':<8} {'retr.':<7} {'k':>2} {'MSA':>6} {'EGY':>6} {'gap':>6} {'p':>8} | {'LLM':>6} {'gain':>6} {'p':>8} "
             f"{'closed [95% CI]':>22} | {'rules':>6} {'closed':>7} | {'left vs MSA p':>13}")
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        for m in METHODS:
            for k in (5, 10):
                e, l_, r_, ms = ((ranks[c][m][m_] <= k) for c in ("egy", "egy>llm", "egy>rules", "msa"))
                _, _, p_gap = mcnemar(ms, e)
                _, _, p_gain = mcnemar(l_, e)
                _, _, p_left = mcnemar(ms, l_)
                cl, lo, hi = closed_ci(e, l_, ms, a.reps)
                cr, _, _ = closed_ci(e, r_, ms, 200)
                js["gap"].setdefault(s, {}).setdefault(m, {})[str(k)] = {
                    "msa": float(ms.mean()), "egy": float(e.mean()), "llm": float(l_.mean()), "rules": float(r_.mean()),
                    "p_gap": p_gap, "p_gain": p_gain, "p_left": p_left, "closed": cl, "closed_ci": [lo, hi],
                    "closed_rules": cr}
                L.append(f"  {s:<8} {m:<7} {k:>2} {100 * ms.mean():5.1f}% {100 * e.mean():5.1f}% {100 * (ms.mean() - e.mean()):+5.1f} "
                         f"{p_gap:8.1g} | {100 * l_.mean():5.1f}% {100 * (l_.mean() - e.mean()):+5.1f} {p_gain:8.1g} "
                         f"{100 * cl:6.0f}% [{100 * lo:5.0f},{100 * hi:4.0f}] | {100 * r_.mean():5.1f}% {100 * cr:6.0f}% | {p_left:13.2g}")
    L.append("  closed = share of the MSA-vs-Egyptian gap recovered by rewriting (100% = as good as the MSA twin;")
    L.append("           > 100% = better than the original MSA). 'left vs MSA p' tests whether a gap remains.")

    L.append("\nDOES REWRITING HURT QUESTIONS THAT ARE ALREADY MSA?  (why a dialect head should decide when to rewrite)")
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        for m in METHODS:
            ms, ml = (ranks["msa"][m][m_] <= 5), (ranks["msa>llm"][m][m_] <= 5)
            x, y, p = mcnemar(ms, ml)
            js["harm"].setdefault(s, {})[m] = {"msa": float(ms.mean()), "msa_llm": float(ml.mean()), "p": p}
            L.append(f"  {s:<8} {m:<7} R@5  MSA {100 * ms.mean():5.1f}% -> rewritten {100 * ml.mean():5.1f}%  "
                     f"({100 * (ml.mean() - ms.mean()):+.1f} pts, lost {x} / gained {y}, p={p:.2g})")

    recs = [json.loads(x) for x in (out / "rewrites.jsonl").read_text(encoding="utf-8").splitlines()]
    from collections import Counter
    fl = Counter(p for r in recs for p in r["flags_llm"])
    fr = Counter(p for r in recs for p in r["flags_rules"])
    same_msa = np.mean([r["msa>llm"].strip() == r["msa"].strip() for r in recs])
    js["diagnostics"] = {"llm_flags": dict(fl), "rules_flags": dict(fr), "msa_unchanged_by_llm": float(same_msa)}
    L += ["", "REWRITE CHECKS (automatic)",
          f"  LLM rewrites:   {dict(fl) or 'no problems'}  (of {len(recs)})",
          f"  rule rewrites:  {dict(fr) or 'no problems'}",
          f"  MSA questions returned unchanged by the rewriter: {100 * same_msa:.1f}%"]

    rng = np.random.RandomState(5)
    pick = sorted(rng.choice(len(recs), min(a.n_review, len(recs)), replace=False))
    with (out / "rewrite_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "set", "egyptian", "rewritten_msa", "original_msa", "auto_flags", "ok", "notes"])
        for k, i in enumerate(pick, 1):
            r = recs[i]
            w.writerow([f"R{k:03d}", r["set"], r["egy"], r["egy>llm"], r["msa"], "; ".join(r["flags_llm"]), "", ""])
    L.append(f"\nspot-check file (optional, {len(pick)} rows): {out / 'rewrite_review.csv'}   total time {time.time() - t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (out / "rewrite_report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "rewrite_report.json").write_text(json.dumps(js, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
