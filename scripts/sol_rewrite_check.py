"""Phase 8 (optional): is the remaining dialect gap a limit of the METHOD or of the small REWRITER?

  python scripts/sol_rewrite_check.py --budget_usd 2            # ~500 native ArDQA questions, hard cap $2

Rewrites native Egyptian questions (ArDQA: the realistic set, where the gap is largest) with a stronger model
(default gpt-6.1-sol), using the SAME prompt as the Phase 6 rewriter, and compares retrieval (found in top 5):
  Egyptian as typed | rewritten by gpt-4.1-mini (Phase 6) | rewritten by Sol | the MSA question (upper bound)
plus the share of the gap each rewriter closes (paired bootstrap CI), McNemar Sol vs mini, and cost per 1,000
rewrites. Writes runs_phase8/sol_rewrite_report.txt and sol_rewrites.jsonl. Local search, no answering.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None, complete_fn=None, retriever=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="low", help="reasoning effort; 'off' = plain call with temperature 0")
    ap.add_argument("--max_out", type=int, default=2000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--budget_usd", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs_phase8")
    a = ap.parse_args(argv)

    from concurrent.futures import ThreadPoolExecutor

    from rewrite_eval import closed_ci, first_hit, mcnemar

    from arroute.egy.rewrite import check_rewrite, rewrite_llm_usage
    from arroute.llm import Budget, BudgetExceeded
    from arroute.router.final_eval import tok_cost
    if complete_fn is None:
        from arroute.llm import complete_with_usage as complete_fn
    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [it for it in items if it["set"] == "ardqa" and it.get("gold_pids")]
    random.Random(a.seed).shuffle(items)
    items = items[:a.n]
    budget = Budget(a.budget_usd)
    pin, pout = budget.price(a.model)
    params = {"temperature": 0, "max_output_tokens": 200} if a.effort == "off" else \
        {"reasoning": {"effort": a.effort}, "max_output_tokens": a.max_out}
    gen = {"fn": complete_fn, "model": a.model, "budget": budget, "params": params}
    gen_mini = {"fn": complete_fn, "model": "gpt-4.1-mini", "budget": budget}   # = the router's rewriter (cached)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} native Egyptian questions | rewriter {a.model} ({params}) | budget ${a.budget_usd:.2f}")

    # one call first: a wrong model name or parameter fails here, before any spending in parallel
    res, stopped = {items[0]["id"]: (rewrite_llm_usage(items[0]["q_egy"], gen),
                                     rewrite_llm_usage(items[0]["q_egy"], gen_mini))}, False

    def job(it):
        return it["id"], (rewrite_llm_usage(it["q_egy"], gen), rewrite_llm_usage(it["q_egy"], gen_mini))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for fut in [pool.submit(job, it) for it in items[1:]]:
            try:
                k, r = fut.result()
                res[k] = r
            except BudgetExceeded:
                stopped = True
            except Exception as ex:                                  # noqa: BLE001
                print(f"  skipped: {type(ex).__name__}: {str(ex)[:120]}")
    items = [it for it in items if it["id"] in res]
    print(f"rewritten: {len(items)}  | spend ${budget.spent:.4f}" + ("  (STOPPED by the budget)" if stopped else ""))

    if retriever is None:
        from arroute.retrieval.index import load_retriever
        retriever = load_retriever(a.corpus, dense=True)
    hits = {k: [] for k in ("typed", "mini", "sol", "msa")}
    rows, probs, cost_sol, cost_mini, incomplete = [], 0, [], [], 0
    t0 = time.time()
    for it in items:
        (sol_q, usage, _), (mini_q, usage_mini, _) = res[it["id"]]
        q = {"typed": it["q_egy"], "mini": mini_q, "sol": sol_q, "msa": it["q_msa"]}
        gold = set(it["gold_pids"])
        for k, text in q.items():
            hits[k].append(first_hit(retriever.search(text, k=5, method="hybrid"), gold) <= 5)
        p = check_rewrite(it["q_egy"], sol_q)
        probs += bool(p)
        incomplete += bool(usage.get("incomplete"))
        cost_sol.append(tok_cost(usage, pin, pout))
        cost_mini.append(tok_cost(usage_mini, *budget.price("gpt-4.1-mini")))
        rows.append({"id": it["id"], "q_egy": it["q_egy"], "q_msa": it["q_msa"], "mini": q["mini"], "sol": sol_q,
                     "hit": {k: bool(hits[k][-1]) for k in hits}, "problems": p, "usage": usage})
    H = {k: np.array(v, bool) for k, v in hits.items()}
    n = len(items)
    L = [f"PHASE 8 - STRONGER REWRITER CHECK ({n} native Egyptian ArDQA questions, hybrid search, found in top 5)",
         f"rewriter: {a.model} {params} vs gpt-4.1-mini (Phase 6), same prompt", ""]
    for k, name in (("typed", "Egyptian as typed"), ("mini", "rewritten by gpt-4.1-mini"),
                    ("sol", f"rewritten by {a.model}"), ("msa", "MSA question (upper bound)")):
        L.append(f"  {name:<34}{100 * H[k].mean():6.1f}%")
    L.append("")
    for k in ("mini", "sol"):
        pt, lo, hi = closed_ci(H["typed"], H[k], H["msa"])
        L.append(f"  gap closed by {k:<5} {100 * pt:5.0f}%  [{100 * lo:.0f}-{100 * hi:.0f}]")
    x, y, p = mcnemar(H["sol"], H["mini"])
    d = 100 * (H["sol"].mean() - H["mini"].mean())
    L += [f"  Sol vs mini: {d:+.1f} pts (Sol only {x}, mini only {y}), McNemar p={p:.2g}",
          "",
          f"  cost per 1,000 rewrites: {a.model} ${1000 * np.mean(cost_sol):.3f}"
          f" | gpt-4.1-mini ${1000 * np.mean(cost_mini):.3f}",
          f"  Sol rewrites flagged by the Phase 6 checks: {probs} | cut off (incomplete): {incomplete}",
          f"  spend ${budget.spent:.4f} of ${a.budget_usd:.2f}" + ("  (STOPPED by the budget)" if stopped else ""),
          f"  time {time.time() - t0:.0f}s"]
    text = "\n".join(L)
    print("\n" + text)
    (out / "sol_rewrite_report.txt").write_text(text + "\n", encoding="utf-8")
    with (out / "sol_rewrites.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
