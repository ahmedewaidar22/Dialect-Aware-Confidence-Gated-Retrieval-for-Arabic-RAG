"""Per-stage latency and total operational cost of the final system.

The paper's cost column prices only the generator's tokens. This script measures the rest (dialect head, rewrite
call, BM25 + BGE-M3 search, reranker, C1 call, C2 calls) so that a total cost per 1,000 questions can be reported.

Use inside your own run (a few lines around the objects load_final() already builds):

    from latency_profile import StageTimer, report
    t = StageTimer()
    system.dialect  = t.wrap("dialect_head", system.dialect)
    system.rewrite  = t.wrap("rewrite_llm", system.rewrite)
    system.reranker = t.wrap("reranker", system.reranker)
    system.r.all_methods = t.wrap("search_bm25_bgem3", system.r.all_methods)
    ask_c1 = t.wrap("c1_llm", ask_c1);  gen.ask = t.wrap("c2_llm", gen.ask)
    for q in questions:                       # run on a sample of >= 200 questions, after one warm-up question
        with t.question(): system.answer(q, ask_c1, gen)
    report(t, llm_usd_per_1k=0.456, gpu_usd_per_hour=[0.0, 0.5, 1.0, 2.0])

Cached LLM calls return instantly and would understate latency: clear or bypass the cache for this run.
`python scripts/latency_profile.py --selftest` checks the timing logic on fake stages (no models needed).
"""
import argparse
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager

GPU_STAGES = ("dialect_head", "search_bm25_bgem3", "reranker")      # stages that occupy the local GPU


class StageTimer:
    def __init__(self):
        self.stage = defaultdict(list)       # per call
        self.per_q = []                      # per question: {stage: seconds}
        self._cur = None

    def wrap(self, name, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                dt = time.perf_counter() - t0
                self.stage[name].append(dt)
                if self._cur is not None:
                    self._cur[name] = self._cur.get(name, 0.0) + dt
        return inner

    @contextmanager
    def question(self):
        self._cur = {}
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._cur["total"] = time.perf_counter() - t0
            self.per_q.append(self._cur)
            self._cur = None


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def report(t, llm_usd_per_1k=None, gpu_usd_per_hour=(0.0, 1.0)):
    n = len(t.per_q)
    L = [f"Latency over {n} questions (seconds per question; stages a question did not use count as 0)"]
    names = sorted({k for q in t.per_q for k in q if k != "total"})
    for s in names:
        v = [q.get(s, 0.0) for q in t.per_q]
        L.append(f"  {s:<20} mean {statistics.mean(v):7.3f}   p50 {pct(v, 50):7.3f}   p95 {pct(v, 95):7.3f}")
    tot = [q["total"] for q in t.per_q]
    L.append(f"  {'TOTAL':<20} mean {statistics.mean(tot):7.3f}   p50 {pct(tot, 50):7.3f}   p95 {pct(tot, 95):7.3f}")
    gpu_s = statistics.mean(sum(q.get(s, 0.0) for s in GPU_STAGES) for q in t.per_q)
    L.append(f"Local GPU seconds per question: {gpu_s:.3f}  ->  {1000 * gpu_s / 3600:.3f} GPU-hours per 1,000 questions")
    if llm_usd_per_1k is not None:
        L.append("Total cost per 1,000 questions = generator tokens + GPU time x assumed hourly price")
        for p in gpu_usd_per_hour:
            L.append(f"  GPU at ${p:.2f}/h: ${llm_usd_per_1k:.3f} + ${1000 * gpu_s / 3600 * p:.3f} = "
                     f"${llm_usd_per_1k + 1000 * gpu_s / 3600 * p:.3f}")
    text = "\n".join(L)
    print(text)
    return text


def selftest():
    t = StageTimer()
    f = t.wrap("reranker", lambda: time.sleep(0.02))
    g = t.wrap("c1_llm", lambda: time.sleep(0.01))
    for i in range(10):
        with t.question():
            f()
            if i % 2:
                g()
    assert len(t.per_q) == 10 and len(t.stage["reranker"]) == 10 and len(t.stage["c1_llm"]) == 5
    out = report(t, llm_usd_per_1k=0.456, gpu_usd_per_hour=[1.0])
    assert "reranker" in out and "TOTAL" in out
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
