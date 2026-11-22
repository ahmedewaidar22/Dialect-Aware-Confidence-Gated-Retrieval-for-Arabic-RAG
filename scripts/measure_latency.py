"""Latency per stage, GPU time and total cost under a stated deployment scenario, for the final system and the baselines.

  python scripts/measure_latency.py --n 200 --gpu_usd_per_hour 0.35,0.70,1.40 --budget_usd 1.0

What is measured, per question (Egyptian input, a seeded random sample of the 1,000 test questions):
  local stages (run --local_repeats times, after a warm-up; GPU work is synchronised before the clock stops)
    dialect_head      AraBERT P(Egyptian)                                   GPU
    bm25              BM25 search (per query; twice in the uncertain band)  CPU
    dense             BGE-M3 query encoding + dense search                  GPU
    rerank            bge-reranker-v2-m3 on the 20 candidates               GPU
  LLM stages (FRESH calls, never the cache: a private '_lat' key is added, which changes only the cache key)
    rewrite           Egyptian -> MSA rewrite (only when the plan rewrites)
    c1                C1 answer from the reranked passages (final) or the hybrid top 5 (standard RAG)
    c2                iterative retrieval, LLM time and its own searches (final: only when C1 says "I don't know")
Systems: final, standard_rag, always_c2. (C1 is timed with a plain call on the same prompt; the final system's own C1 call
also returns token log-probabilities, which adds little time. C2's first search here is plain hybrid search.) End-to-end latency = sum of the stages each system runs, in order.
Reported: mean, median and 90th percentile per stage and per system with bootstrap 95% intervals over questions,
the spread across local repeats, GPU-seconds per question, peak GPU memory, and cost per 1,000 questions =
generator tokens (list price) + GPU time under two deployment scenarios:
  busy-time  the GPU is billed only while it works (serverless-style): GPU-seconds x price
  dedicated  one GPU instance always on: price x 24 h / (questions per day), for 1k, 10k and 100k questions per day
The GPU prices are inputs: pass the hourly price of the instance you would actually use, and say which in the paper.
Spend: about US$0.002 per question for the fresh LLM calls (hard cap --budget_usd).
"""
import argparse
import csv
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed(fn, *args, gpu=False, **kw):
    if gpu:
        sync()
    t = time.perf_counter()
    out = fn(*args, **kw)
    if gpu:
        sync()
    return out, time.perf_counter() - t


def boot_ci(xs, f=st.mean, reps=2000, seed=0):
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    v = sorted(f([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(reps))
    return v[int(0.025 * reps)], v[int(0.975 * reps)]


def p90(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.9 * len(s)))]


def run_question(system, q, rewrite_fresh, c1_fresh, c2_fresh, std_c1_fresh, repeats, idk):
    """All timings for one question. LLM functions return (text, seconds, usd)."""
    from arroute.pipeline.final import fuse
    R = {}
    # dialect head (local, repeated)
    dh = []
    for _ in range(repeats):
        p, t = timed(system.dialect, [q], gpu=True)
        dh.append(t)
    p = float(p[0])
    R["dialect_head"] = st.median(dh)
    R["dialect_head_spread"] = max(dh) - min(dh)
    # plan + fresh rewrite
    if p >= system.hi or p > system.lo:
        mq, R["rewrite"], R["usd_rewrite"] = rewrite_fresh(q)
    else:
        mq, R["rewrite"], R["usd_rewrite"] = q, 0.0, 0.0
    searches = [mq] if (p >= system.hi or p <= system.lo) else [q, mq]
    R["plan"] = "egy" if p >= system.hi else "msa" if p <= system.lo else "unsure"
    # retrieval stages (local, repeated): bm25 and dense separately, per search query
    b, d, rr = [], [], []
    for _ in range(repeats):
        tb = td = 0.0
        per = []
        for s in searches:
            lex, t1 = timed(system.r.bm25.search, s, system.r.depth)
            den, t2 = timed(system.r.dense.search, s, system.r.depth, gpu=True)
            tb, td = tb + t1, td + t2
            from arroute.retrieval.index import rrf
            per.append({"bm25": lex[:100], "dense": den[:100], "hybrid": rrf([lex, den])[:100]})
        b.append(tb)
        d.append(td)
        cands = fuse(per)[:system.n_cands]
        scores, t3 = timed(system.reranker, [(mq, system.r.text(c)) for c in cands], gpu=True)
        rr.append(t3)
    R["bm25"], R["dense"], R["rerank"] = st.median(b), st.median(d), st.median(rr)
    R["retrieval_spread"] = (max(b) - min(b)) + (max(d) - min(d)) + (max(rr) - min(rr))
    order = sorted(range(len(cands)), key=lambda i: (-float(scores[i]), i))
    top = [cands[i] for i in order[:system.k]]
    # final: C1 on top-k, then C2 if "I don't know"
    ans, R["c1_final"], R["usd_c1_final"] = c1_fresh(mq, top)
    esc = idk in (ans or "") or not (ans or "").strip()
    R["escalated"] = int(esc)
    if esc:
        _, R["c2_final"], R["usd_c2_final"] = c2_fresh(mq)
    else:
        R["c2_final"], R["usd_c2_final"] = 0.0, 0.0
    # standard RAG: hybrid top-5 on the question as typed (local timing measured once, plus fresh C1)
    lex, t1 = timed(system.r.bm25.search, q, system.r.depth)
    den, t2 = timed(system.r.dense.search, q, system.r.depth, gpu=True)
    from arroute.retrieval.index import rrf
    top5 = rrf([lex, den])[:5]
    R["std_retrieval"] = t1 + t2
    _, R["c1_std"], R["usd_c1_std"] = std_c1_fresh(q, top5)
    # always C2 on the question as typed
    _, R["c2_always"], R["usd_c2_always"] = c2_fresh(q)
    R["gpu_final"] = R["dialect_head"] + R["dense"] + R["rerank"]
    R["gpu_std"] = t2
    R["gpu_c2"] = t2                       # first search; C2's later searches are inside its timed block
    R["final"] = R["dialect_head"] + R["rewrite"] + R["bm25"] + R["dense"] + R["rerank"] + R["c1_final"] + R["c2_final"]
    R["standard_rag"] = R["std_retrieval"] + R["c1_std"]
    R["always_c2"] = R["c2_always"]
    R["usd_final"] = R["usd_rewrite"] + R["usd_c1_final"] + R["usd_c2_final"]
    R["usd_std"], R["usd_c2"] = R["usd_c1_std"], R["usd_c2_always"]
    return R


def summarize(rows, prices, lines):
    stages = ["dialect_head", "rewrite", "bm25", "dense", "rerank", "c1_final", "c2_final", "std_retrieval", "c1_std", "c2_always"]
    lines.append(f"{'stage':16s}{'mean ms':>10s}{'95% CI':>20s}{'median':>9s}{'p90':>9s}")
    for s in stages:
        xs = [1000 * r[s] for r in rows]
        lo, hi = boot_ci(xs)
        lines.append(f"{s:16s}{st.mean(xs):10.1f}{f'[{lo:.1f}, {hi:.1f}]':>20s}{st.median(xs):9.1f}{p90(xs):9.1f}")
    lines.append("")
    lines.append(f"{'system':14s}{'mean s':>9s}{'95% CI':>18s}{'median':>8s}{'p90':>8s}{'GPU s/q':>9s}{'US$/1k tokens':>15s}")
    sysmap = {"final": ("final", "gpu_final", "usd_final"), "standard_rag": ("standard_rag", "gpu_std", "usd_std"),
              "always_c2": ("always_c2", "gpu_c2", "usd_c2")}
    for name, (lat, gpu, usd) in sysmap.items():
        xs = [r[lat] for r in rows]
        lo, hi = boot_ci(xs)
        g = st.mean(r[gpu] for r in rows)
        u = 1000 * st.mean(r[usd] for r in rows)
        lines.append(f"{name:14s}{st.mean(xs):9.2f}{f'[{lo:.2f}, {hi:.2f}]':>18s}{st.median(xs):8.2f}{p90(xs):8.2f}{g:9.3f}{u:15.3f}")
    lines.append("")
    lines.append("TOTAL COST per 1,000 questions = generator tokens + GPU (the GPU price is an input, not a measurement)")
    for price in prices:
        lines.append(f"  GPU at US${price:.2f}/h")
        for name, (lat, gpu, usd) in sysmap.items():
            u = 1000 * st.mean(r[usd] for r in rows)
            g = st.mean(r[gpu] for r in rows)
            busy = 1000 * g / 3600 * price
            ded = {q: 1000 * 24 * price / q for q in (1_000, 10_000, 100_000)}
            lines.append(f"    {name:13s} tokens {u:.3f} + GPU busy-time {busy:.4f} = {u + busy:.3f}   | dedicated instance: "
                         + ", ".join(f"{q // 1000}k q/day {u + ded[q]:.2f}" for q in ded))
    lines.append("")
    lines.append(f"escalated to C2: {100 * st.mean(r['escalated'] for r in rows):.1f}% of sampled questions")
    lines.append(f"local-stage spread across repeats (max - min, mean): dialect head {1000 * st.mean(r['dialect_head_spread'] for r in rows):.1f} ms,"
                 f" retrieval + rerank {1000 * st.mean(r['retrieval_spread'] for r in rows):.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--per_question", default="runs_final/per_question.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local_repeats", type=int, default=3)
    ap.add_argument("--gpu_usd_per_hour", default="0.35,0.70,1.40")
    ap.add_argument("--budget_usd", type=float, default=1.0)
    ap.add_argument("--out", default="runs_latency")
    a = ap.parse_args()

    from arroute.egy.rewrite import rewrite_llm_usage
    from arroute.llm import Budget
    from arroute.llm import complete_with_usage as complete_fn
    from arroute.pipeline.final import load_final
    from arroute.pipeline.strategies import P_C1, fmt_passages, run_c2, clean_answer
    from arroute.router.outcome import IDK
    from arroute.router.v2 import tok_cost
    from label_outcomes import Generator

    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    budget = Budget(a.budget_usd)
    system = load_final(a.models, a.corpus, rewrite=lambda q: q)
    nonce = f"lat-{time.time_ns()}"                      # one key per run: every LLM call below is a fresh API call
    counter = {"i": 0}

    def key():
        counter["i"] += 1
        return f"{nonce}-{counter['i']}"

    def rewrite_fresh(q):
        gen = {"fn": complete_fn, "model": model, "budget": budget, "extra": {"_lat": key()}}
        (text, usage, _), t = timed(rewrite_llm_usage, q, gen)
        return text, t, tok_cost(usage)

    def c1_fresh(q, pids):
        prompt = P_C1.format(q=q, passages=fmt_passages([system.r.text(p) for p in pids]))
        (text, usage, _), t = timed(complete_fn, prompt, model=model, budget=budget, temperature=0, max_output_tokens=300, _lat=key())
        return clean_answer(text), t, tok_cost(usage)

    def c2_fresh(q):
        gen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 300, "_lat": key()})
        res, t = timed(run_c2, q, gen, system.r)
        return res["answer"], t, tok_cost(res)

    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    if Path(a.per_question).exists():
        with open(a.per_question, encoding="utf-8-sig", newline="") as f:
            ids = {r["id"] for r in csv.DictReader(f)}
        items = [it for it in items if it["id"] in ids]
    random.Random(a.seed).shuffle(items)
    items = items[:a.n]
    for it in items[:3]:                                    # warm-up (not recorded): loads kernels and caches
        system.dialect([it["q_egy"]])
        system.r.dense.search(it["q_egy"], system.r.depth)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    rows = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, it in enumerate(items, 1):
        R = run_question(system, it["q_egy"], rewrite_fresh, c1_fresh, c2_fresh, c1_fresh, a.local_repeats, IDK)
        R["id"], R["set"] = it["id"], it["set"]
        rows.append(R)
        if n % 20 == 0:
            print(f"  {n}/{len(items)} | spend ${budget.spent:.3f}")
    prices = [float(x) for x in a.gpu_usd_per_hour.split(",")]
    import platform
    L = ["LATENCY AND TOTAL COST (Egyptian input)", f"questions: {len(rows)} (seed {a.seed}), local repeats {a.local_repeats}",
         f"machine: {platform.processor() or platform.machine()} | " + (f"GPU {torch.cuda.get_device_name(0)}, peak memory "
         f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if torch is not None and torch.cuda.is_available() else "no GPU"),
         f"generator {model} via API (latency includes network); fresh calls only; spend ${budget.spent:.3f}", ""]
    summarize(rows, prices, L)
    text = "\n".join(L)
    print(text)
    (out / "latency_report.txt").write_text(text + "\n", encoding="utf-8")
    with (out / "latency_per_question.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {out / 'latency_report.txt'} and {out / 'latency_per_question.csv'}")


if __name__ == "__main__":
    main()
