"""Phase 7 (RQ5, retrieval level): does the DIALECT-AWARE query plan beat simpler plans on mixed traffic?

  python scripts/eval_query_plan.py        # local, no API (uses Phase 6 rewrites + the saved dialect head scores)

Every question arrives twice - as the Egyptian version and as the MSA version - so the stream is half dialect,
half MSA. Plans compared (hybrid search; also BM25 and dense):
  never rewrite        search what was typed                         (standard RAG)
  always rewrite       rewrite every question                        (no dialect head)
  always both          search typed + rewrite, merge                 (no dialect head, safe but 2x searches)
  dialect head (0.5)   rewrite if P(Egyptian) >= 0.5, else as typed
  dialect head + band  >= hi rewrite | <= lo as typed | unsure: both  (THE ROUTER)
  oracle dialect       rewrite exactly the Egyptian inputs           (upper bound for any dialect head)
Writes runs_router7/query_plan_report.txt/.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from rewrite_eval import first_hit, mcnemar          # noqa: E402
from rewrite_variants import cached_encoder           # noqa: E402

METHODS = ["bm25", "dense", "hybrid"]
PLANS = ["never rewrite", "always rewrite", "always both", "dialect head (0.5)", "dialect head + band", "oracle dialect"]


def lists_for(per_query):
    from arroute.retrieval.index import rrf
    if len(per_query) == 1:
        return per_query[0]
    out = {m: rrf([r[m] for r in per_query]) for m in ("bm25", "dense")}
    out["hybrid"] = rrf([r[m] for r in per_query for m in ("bm25", "dense")])
    return out


def plan_queries(plan, typed, rew, p, truly_egy, lo, hi):
    if plan == "never rewrite":
        return ["t"]
    if plan == "always rewrite":
        return ["r"]
    if plan == "always both":
        return ["t", "r"]
    if plan == "dialect head (0.5)":
        return ["r"] if p >= 0.5 else ["t"]
    if plan == "dialect head + band":
        return ["r"] if p >= hi else (["t"] if p <= lo else ["t", "r"])
    return ["r"] if truly_egy else ["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--rewrites", default="runs_rewrite/rewrites.jsonl")
    ap.add_argument("--dialect_scores", default="models/dialect_head/egy_test_scores.jsonl")
    ap.add_argument("--dialect_meta", default="models/dialect_head/meta.json")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--out", default="runs_router7")
    a = ap.parse_args()
    t0 = time.time()
    meta = json.loads(Path(a.dialect_meta).read_text(encoding="utf-8"))
    lo, hi = meta.get("lo", 0.2), meta.get("hi", 0.8)
    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [it for it in items if "rejected" not in (it.get("review") or {}).values()]
    rw = {json.loads(x)["id"]: json.loads(x) for x in Path(a.rewrites).read_text(encoding="utf-8").splitlines() if x.strip()}
    ds = {json.loads(x)["id"]: json.loads(x) for x in Path(a.dialect_scores).read_text(encoding="utf-8").splitlines() if x.strip()}
    # the stream: every item twice (Egyptian input, MSA input)
    stream = []
    for it in items:
        r, d = rw[it["id"]], ds[it["id"]]
        stream.append({"id": it["id"], "set": it["set"], "input": "egy", "t": it["q_egy"], "r": r["egy>llm"],
                       "p": d["p_egy_on_egy"], "gold": set(it["gold_pids"])})
        stream.append({"id": it["id"], "set": it["set"], "input": "msa", "t": it["q_msa"], "r": r["msa>llm"],
                       "p": d["p_egy_on_msa"], "gold": set(it["gold_pids"])})
    from arroute.retrieval.index import load_retriever
    retr = load_retriever(a.corpus, dense=True, device=a.device)
    texts = sorted({s[k] for s in stream for k in ("t", "r")})
    retr.dense.encode = cached_encoder(retr.dense.encode, texts)
    res = {t: retr.all_methods(t, k=100) for t in texts}
    print(f"searched {len(texts):,} distinct queries ({time.time() - t0:.0f}s)")

    hits, nsearch = {}, {}
    for plan in PLANS:
        h = {m: np.zeros(len(stream), bool) for m in METHODS}
        ns = np.zeros(len(stream))
        for i, s in enumerate(stream):
            qs = plan_queries(plan, s["t"], s["r"], s["p"], s["input"] == "egy", lo, hi)
            lists = lists_for([res[s[k]] for k in qs])
            ns[i] = len(qs)
            for m in METHODS:
                h[m][i] = first_hit(lists[m], s["gold"]) <= 5
        hits[plan], nsearch[plan] = h, ns
    inp = np.array([s["input"] for s in stream])
    sets = np.array([s["set"] for s in stream])
    L = [f"PHASE 7 - DIALECT-AWARE QUERY PLAN on mixed traffic ({len(items):,} questions x 2 inputs = {len(stream):,})",
         f"band: rewrite if P(Egyptian) >= {hi}, as typed if <= {lo}, otherwise search both and merge.  Found in top 5:", ""]
    js = {"lo": lo, "hi": hi, "plans": {}}
    for scope, mask in (("ALL inputs", np.ones(len(stream), bool)), ("Egyptian inputs", inp == "egy"),
                        ("MSA inputs", inp == "msa"), ("ArDQA (native), all inputs", sets == "ardqa")):
        L.append(f"{scope} (n={int(mask.sum())})")
        L.append(f"  {'plan':<22}" + "".join(f"{m:>9}" for m in METHODS) + f"{'searches/q':>12}   hybrid vs router (McNemar)")
        for plan in PLANS:
            row = "".join(f"{100 * hits[plan][m][mask].mean():8.1f}%" for m in METHODS)
            x, y, p = mcnemar(hits[plan]["hybrid"][mask], hits["dialect head + band"]["hybrid"][mask])
            cmp = "" if plan == "dialect head + band" else f"   {100 * (hits[plan]['hybrid'][mask].mean() - hits['dialect head + band']['hybrid'][mask].mean()):+5.1f} pts, p={p:.2g}"
            L.append(f"  {plan:<22}{row}{nsearch[plan][mask].mean():12.2f}{cmp}")
            js["plans"].setdefault(scope, {})[plan] = {m: float(hits[plan][m][mask].mean()) for m in METHODS}
            js["plans"][scope][plan]["searches"] = float(nsearch[plan][mask].mean())
        L.append("")
    pe = np.array([s["p"] for s in stream])
    L.append("WHAT THE DIALECT HEAD DECIDED")
    for name, mask in (("Egyptian inputs", inp == "egy"), ("MSA inputs", inp == "msa")):
        p = pe[mask]
        L.append(f"  {name:<16} rewrite {100 * (p >= hi).mean():5.1f}% | both (unsure) {100 * ((p > lo) & (p < hi)).mean():5.1f}% "
                 f"| as typed {100 * (p <= lo).mean():5.1f}%")
    L.append(f"\ntotal time {time.time() - t0:.0f}s")
    text = "\n".join(L)
    print("\n" + text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "query_plan_report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "query_plan_report.json").write_text(json.dumps(js, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
