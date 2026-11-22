"""Phase 6b: better ways to use the Egyptian->MSA rewrite in retrieval.

  python scripts/rewrite_variants.py --budget_usd 0.5     # minimal-edit rewrites ~$0.10 (cached); retrieval local

Needs Phase 6 (runs_rewrite/rewrites.jsonl). Compares, on the same questions:
  egy                 Egyptian as typed
  egy>llm             Phase 6 rewrite (free paraphrase into MSA)
  egy>min             NEW minimal-edit rewrite: only dialect words change, content words stay
  egy+llm (fused)     search BOTH the Egyptian question and the rewrite, merge the rankings (RRF)
  egy+min (fused)     same with the minimal-edit rewrite
  egy|min (one query) Egyptian and minimal rewrite joined into one query string
  msa                 MSA twin (best case)
  msa>min / msa+min   the minimal edit / fusion applied to questions that are ALREADY MSA (harm check)
Writes runs_rewrite/variants_report.txt/.json and variants_ranks.jsonl
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.egy.rewrite import P_REWRITE_MIN, check_rewrite, rewrite_llm   # noqa: E402
from rewrite_eval import closed_ci, first_hit, mcnemar                      # noqa: E402

METHODS = ["bm25", "dense", "hybrid"]
KS = (1, 5, 10, 20)
SETS = ["ardqa", "pilot", "heldout", "ALL"]
# variant -> list of query-version names whose rankings are fused (one name = plain search)
VARIANTS = {"egy": ["egy"], "egy>llm": ["egy>llm"], "egy>min": ["egy>min"],
            "egy+llm (fused)": ["egy", "egy>llm"], "egy+min (fused)": ["egy", "egy>min"],
            "egy|min (one query)": ["egy|min"], "msa": ["msa"], "msa>min": ["msa>min"],
            "msa+min (fused)": ["msa", "msa>min"]}
EGY_VARIANTS = [v for v in VARIANTS if v.startswith("egy")]


def fused_lists(per_query, depth=100):
    """per_query: list of all_methods() outputs. bm25/dense: RRF of that retriever's lists over the queries;
    hybrid: RRF of ALL bm25 + dense lists (exactly how single-query hybrid fuses its two lists)."""
    from arroute.retrieval.index import rrf
    if len(per_query) == 1:
        return per_query[0]
    out = {m: rrf([r[m] for r in per_query])[:depth] for m in ("bm25", "dense")}
    out["hybrid"] = rrf([r[m] for r in per_query for m in ("bm25", "dense")])[:depth]
    return out


def cached_encoder(encode, texts, batch=64):
    """Encode every distinct query once in big batches (fast on GPU), then serve searches from the cache."""
    from arroute.retrieval.text import normalize_ar
    uniq = sorted({normalize_ar(t) for t in texts})
    vecs = {}
    for i in range(0, len(uniq), batch):
        chunk = uniq[i:i + batch]
        for t, v in zip(chunk, encode(chunk)):
            vecs[t] = v

    def enc(ts):
        return np.stack([vecs[t] if t in vecs else encode([t])[0] for t in ts])
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--rewrites", default="runs_rewrite/rewrites.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="runs_rewrite")
    ap.add_argument("--budget_usd", type=float, default=0.5)
    ap.add_argument("--price", action="append", default=["gpt-4.1-mini=0.40,1.60"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    from arroute.llm import Budget, complete_with_usage
    from label_outcomes import parse_prices
    model = os.environ.get("ARROUTE_LLM_MODEL")
    if not model:
        raise SystemExit("set ARROUTE_LLM_MODEL (e.g. gpt-4.1-mini)")
    budget = Budget(a.budget_usd, parse_prices(a.price))
    budget.price(model)
    gen = {"fn": complete_with_usage, "model": model, "budget": budget, "prompt": P_REWRITE_MIN}
    out = Path(a.out)
    t0 = time.time()

    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [it for it in items if "rejected" not in (it.get("review") or {}).values()]
    prev = {}
    for x in Path(a.rewrites).read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            prev[r["id"]] = r
    missing = [it["id"] for it in items if it["id"] not in prev]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no Phase 6 rewrite - run run_phase6.ps1 first")

    print(f"[1/3] minimal-edit rewrites for {len(items)} Egyptian questions and their MSA twins ...")
    with ThreadPoolExecutor(a.workers) as pool:
        egy_min = list(pool.map(lambda it: rewrite_llm(it["q_egy"], gen), items))
        msa_min = list(pool.map(lambda it: rewrite_llm(it["q_msa"], gen), items))
    q = {"egy": [it["q_egy"] for it in items], "egy>llm": [prev[it["id"]]["egy>llm"] for it in items],
         "egy>min": egy_min, "msa": [it["q_msa"] for it in items], "msa>min": msa_min}
    q["egy|min"] = [f"{e} {m}" for e, m in zip(q["egy"], q["egy>min"])]
    print(f"      done ({time.time() - t0:.0f}s, new spend ${budget.spent:.3f})")

    print("[2/3] retrieval ...")
    from arroute.retrieval.index import load_retriever
    retr = load_retriever(a.corpus, dense=True, device=a.device)
    retr.dense.encode = cached_encoder(retr.dense.encode, [t for v in q.values() for t in v])
    print(f"      encoded all queries in one go ({time.time() - t0:.0f}s)")
    res = {name: [retr.all_methods(t, k=100) for t in texts] for name, texts in q.items()}
    print(f"      searched {sum(len(v) for v in res.values()):,} queries ({time.time() - t0:.0f}s)")
    ranks = {v: {m: np.zeros(len(items), int) for m in METHODS} for v in VARIANTS}
    for v, parts in VARIANTS.items():
        for i, it in enumerate(items):
            lists = fused_lists([res[p][i] for p in parts])
            for m in METHODS:
                ranks[v][m][i] = first_hit(lists[m], set(it["gold_pids"]))
    with (out / "variants_ranks.jsonl").open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": it["id"], "set": it["set"], "egy>min": q["egy>min"][i], "msa>min": q["msa>min"][i],
                                **{f"{v}|{m}": int(ranks[v][m][i]) for v in VARIANTS for m in METHODS}},
                               ensure_ascii=False) + "\n")

    print("[3/3] report ...")
    sets = np.array([it["set"] for it in items])
    masks = {s: (sets == s) if s != "ALL" else np.ones(len(items), bool) for s in SETS}
    js = {"recall": {}, "closed": {}, "vs_phase6": {}, "harm": {}}
    L = [f"PHASE 6b - BETTER USE OF THE REWRITE   {len(items):,} questions, new LLM spend ${budget.spent:.3f}", ""]
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        L.append(f"RECALL - {s} (n={int(m_.sum())})   R@1 / R@5 / R@10 / R@20")
        L.append(f"  {'variant':<20}" + "".join(f"{m:>26}" for m in METHODS))
        for v in VARIANTS:
            cells = []
            for m in METHODS:
                vals = [float((ranks[v][m][m_] <= k).mean()) for k in KS]
                js["recall"].setdefault(s, {}).setdefault(v, {})[m] = dict(zip(map(str, KS), vals))
                cells.append(" / ".join(f"{100 * x:4.1f}" for x in vals))
            L.append(f"  {v:<20}" + "".join(f"{x:>26}" for x in cells))
        L.append("")

    L.append("SHARE OF THE DIALECT GAP CLOSED, top-5 (what C1 reads)   [95% CI]   and: better than the Phase 6 rewrite?")
    L.append(f"  {'set':<8} {'retr.':<7} {'variant':<20} {'R@5':>6} {'closed':>8} {'95% CI':>14} | vs egy>llm: {'diff':>6} {'p':>8}")
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        for m in METHODS:
            e, ms, base = ((ranks[v][m][m_] <= 5) for v in ("egy", "msa", "egy>llm"))
            for v in EGY_VARIANTS[1:]:
                h = ranks[v][m][m_] <= 5
                cl, lo, hi = closed_ci(e, h, ms, a.reps)
                x, y, p = mcnemar(h, base)
                js["closed"].setdefault(s, {}).setdefault(m, {})[v] = {"r5": float(h.mean()), "closed": cl, "ci": [lo, hi]}
                js["vs_phase6"].setdefault(s, {}).setdefault(m, {})[v] = {"diff": float(h.mean() - base.mean()), "p": p}
                L.append(f"  {s:<8} {m:<7} {v:<20} {100 * h.mean():5.1f}% {100 * cl:7.0f}% [{100 * lo:5.0f},{100 * hi:5.0f}] | "
                         f"{'':11}{100 * (h.mean() - base.mean()):+5.1f} {p:8.2g}")
        L.append("")

    L.append("HARM ON QUESTIONS THAT ARE ALREADY MSA (top-5): plain MSA vs minimal edit vs fusion")
    for s in SETS:
        m_ = masks[s]
        if not m_.any():
            continue
        for m in METHODS:
            ms = ranks["msa"][m][m_] <= 5
            parts = []
            for v in ("msa>min", "msa+min (fused)"):
                h = ranks[v][m][m_] <= 5
                x, y, p = mcnemar(ms, h)
                js["harm"].setdefault(s, {}).setdefault(m, {})[v] = {"r5": float(h.mean()), "p": p}
                parts.append(f"{v} {100 * h.mean():5.1f}% ({100 * (h.mean() - ms.mean()):+.1f}, p={p:.2g})")
            L.append(f"  {s:<8} {m:<7} MSA {100 * ms.mean():5.1f}% | " + " | ".join(parts))

    same_msa = np.mean([a_.strip() == b_.strip() for a_, b_ in zip(q["msa"], q["msa>min"])])
    flags = [p for it, r in zip(items, egy_min) for p in check_rewrite(it["q_egy"], r, it.get("answer"))]
    from collections import Counter
    L += ["", f"minimal-edit rewriter: MSA questions returned unchanged {100 * same_msa:.1f}% "
          f"(Phase 6 rewriter: see rewrite_report) | automatic flags on Egyptian rewrites: {dict(Counter(flags)) or 'none'}"]
    best = max(EGY_VARIANTS[1:], key=lambda v: js["recall"]["ALL"][v]["hybrid"]["5"])
    L.append(f"best variant on ALL questions, hybrid R@5: {best}")
    js["best_hybrid_r5_all"] = best
    L.append(f"total time {time.time() - t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    (out / "variants_report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "variants_report.json").write_text(json.dumps(js, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
