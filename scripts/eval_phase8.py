"""Phase 8: FINAL end-to-end comparison on Egyptian input - standard RAG vs Adaptive-RAG vs YOUR router.

  python scripts/eval_phase8.py --budget_usd 5                 # route (GPU) -> answer (API) -> stage C (GPU) -> report
  python scripts/eval_phase8.py --report_only                  # rebuild the report from saved results (free)
  python scripts/eval_phase8.py --limit 20 --budget_usd 0.2    # quick smoke test on 20 questions

Test questions: all 300 held-out + all 200 pilot + a random --n_ardqa of the native ArDQA questions
(never used to train or tune anything). Each question is answered with C0, C1 and C2 under four inputs:
Egyptian as typed, Egyptian through the dialect plan, MSA as typed, MSA through the dialect plan
(identical plans are answered once). Every system is then scored OFFLINE from the same answers
(arroute/router/final_eval.py), so the comparison is paired and costs one run.

Finished work is saved as it goes; a re-run (e.g. after the budget stopped it) continues where it stopped,
and repeated LLM calls come from the cache for free. Writes runs_phase8/.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.final_eval import VARIANTS, arrays, report, tok_cost   # noqa: E402

FORCE = {"egy_t": ("q_egy", "typed"), "egy_r": ("q_egy", None), "msa_t": ("q_msa", "typed"), "msa_r": ("q_msa", None)}
STATE_KEYS = ("p_egy", "dialect", "query", "search", "probs3", "p_c0", "cands", "passages", "rr20", "p_ok_pre",
              "x_pre", "p_need_c2")


def read_jsonl(path, key):
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:          # a line cut off by a hard stop: it is redone
                    continue
                out[d[key]] = d
    return out


def select_items(test_path, n_ardqa, seed, limit=None):
    items = [json.loads(x) for x in Path(test_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [it for it in items if it.get("refs") and "rejected" not in (it.get("review") or {}).values()]
    rng = random.Random(seed)
    ard = [it for it in items if it["set"] == "ardqa"]
    rng.shuffle(ard)
    pick = [it for it in items if it["set"] in ("heldout", "pilot")] + ard[:n_ardqa]
    rng.shuffle(pick)                       # random order: if the budget stops the run, the done part is a random sample
    return pick[:limit] if limit else pick


def akey(item_id, st):
    return f"{item_id}||{st['query']}||{' | '.join(st['search'])}"


class PlanRetriever:
    """The routed query gets the routed (possibly merged) top-20; C2's follow-up queries get normal hybrid search."""

    def __init__(self, retriever, lock, query, cands):
        self.r, self.lock, self.query, self.cands = retriever, lock, query, cands

    def search(self, q, k=10, method="hybrid"):
        if q == self.query and k <= len(self.cands):
            return self.cands[:k]
        with self.lock:
            return self.r.search(q, k=k, method=method)

    def text(self, pid):
        return self.r.text(pid)


# ---------------------------------------------------------------- stage 1: route (local models, GPU)
def route_all(router, items, states_f, rw_usage, log=print):
    from arroute.llm import BudgetExceeded
    done = read_jsonl(states_f, "key")
    todo = [(it, v) for it in items for v in VARIANTS if f"{it['id']}|{v}" not in done]
    log(f"[route] {len(done):,} saved, {len(todo):,} to go")
    t0 = time.time()
    with Path(states_f).open("a", encoding="utf-8") as f:
        for n, (it, v) in enumerate(todo, 1):
            field, force = FORCE[v]
            q = it[field]
            t1 = time.time()
            typed = done.get(f"{it['id']}|{v[:3]}_t")
            try:
                if force is None and typed is not None:
                    pl = router.plan(q)                               # same plan as 'typed'? reuse its scores
                    if pl["search"] == [q] and pl["query"] == q:
                        st = {**{k: typed[k] for k in STATE_KEYS}, **pl}
                    else:
                        st = router.route(q, force=force, full=True)
                else:
                    st = router.route(q, force=force, full=True)
            except BudgetExceeded:                                    # a rewrite call hit the cap
                log("  budget reached while rewriting: routing stops here")
                break
            rewrote = st["dialect"] in ("egy", "unsure")
            rec = {"key": f"{it['id']}|{v}", "id": it["id"], "variant": v, "question": q,
                   **{k: st.get(k) for k in STATE_KEYS}, "rewrote": int(rewrote),
                   "rewrite_usage": rw_usage.get(q, {}) if rewrote else {}, "sec": time.time() - t1}
            done[rec["key"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 200 == 0:
                f.flush()
                log(f"  {n:,}/{len(todo):,}  ({time.time() - t0:.0f}s)")
    return done


# ---------------------------------------------------------------- stage 2: answer (API, threads)
def answer_one(item, st, gen, jgen, retriever, lock):
    from arroute.pipeline.judge import judge
    from arroute.pipeline.strategies import run_c0, run_c1, run_c2
    pr = PlanRetriever(retriever, lock, st["query"], st["cands"])
    out = {}
    for name in ("C0", "C1", "C2"):
        r = run_c0(st["query"], gen) if name == "C0" else \
            (run_c1 if name == "C1" else run_c2)(st["query"], gen, pr)
        ok, how, _ = judge(item["q_msa"], r["answer"], item["refs"], jgen)    # same judge prompt for every input
        out[name] = {k: r[k] for k in ("answer", "calls", "searches", "input_tokens", "output_tokens", "incomplete")}
        out[name].update(correct=bool(ok), judge=how, passages=r["passages"][:10])
    return out


def answer_all(items, states, answers_f, gen, jgen, retriever, workers, log=print):
    from arroute.llm import BudgetExceeded
    done = read_jsonl(answers_f, "akey")
    by_id = {it["id"]: it for it in items}
    todo = {}
    for it in items:                                     # item order = random order; identical plans run once
        for v in VARIANTS:
            st = states.get(f"{it['id']}|{v}")
            if st is not None and akey(it["id"], st) not in done:
                todo.setdefault(akey(it["id"], st), st)
    log(f"[answer] {len(done):,} saved, {len(todo):,} input plans to answer (C0 + C1 + C2 each)")
    lock, stopped, failed, t0 = threading.Lock(), False, 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool, Path(answers_f).open("a", encoding="utf-8") as f:
        futs = {pool.submit(answer_one, by_id[st["id"]], st, gen, jgen, retriever, lock): k for k, st in todo.items()}
        for n, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                res = fut.result()
            except BudgetExceeded:
                if not stopped:
                    log("  budget reached: no new calls; finishing the calls in flight...")
                stopped = True
                pool.shutdown(wait=False, cancel_futures=True)
                continue
            except Exception as ex:                      # noqa: BLE001
                failed += 1
                log(f"  skipped {k[:60]}: {type(ex).__name__}: {str(ex)[:120]}  (a re-run retries it)")
                continue
            done[k] = {"akey": k, "results": res}
            f.write(json.dumps(done[k], ensure_ascii=False) + "\n")
            if n % 100 == 0:
                f.flush()
                log(f"  {n:,}/{len(todo):,}  ({time.time() - t0:.0f}s)")
    return done, stopped, failed


# ---------------------------------------------------------------- stage 3: stage C score after C1 (GPU)
def rescue_all(router, items, states, answers, rescue_f, log=print):
    done = read_jsonl(rescue_f, "akey")
    n0 = len(done)
    with Path(rescue_f).open("a", encoding="utf-8") as f:
        for it in items:
            for v in VARIANTS:
                st = states.get(f"{it['id']}|{v}")
                if st is None:
                    continue
                k = akey(it["id"], st)
                if k in done or k not in answers:
                    continue
                s = {**st, "mode": "quality"}
                router.after_c1(s, answers[k]["results"]["C1"]["answer"])
                done[k] = {"akey": k, "p_rescue": float(s["p_rescue"])}
                f.write(json.dumps(done[k]) + "\n")
    log(f"[stage C] {len(done) - n0:,} new scores ({len(done):,} total)")
    return done


# ---------------------------------------------------------------- stage 4: report
def adaptive_lambda(labels, run_dir, heldout_ids, tol=0.01):
    """Adaptive-RAG 'tuned' variant: the cautious threshold λ chosen on the TRAINING questions' out-of-fold
    probabilities (Phase 4a), like the tuned row of the Phase 4e table. None if the files are missing."""
    if not (Path(labels).exists() and (Path(run_dir) / "oof.npz").exists()):
        return None
    from arroute.router.outcome import load_outcomes, outcome_matrix
    from arroute.router.route_report import load_run
    from train_router import choose_lambda
    held = set(json.loads(Path(heldout_ids).read_text(encoding="utf-8"))) if Path(heldout_ids).exists() else set()
    recs = [r for r in load_outcomes(labels) if r["id"] not in held]
    run = load_run(run_dir, recs)
    correct, cost, calls = outcome_matrix(recs)
    return float(choose_lambda(run["probs"], correct, cost, calls, tol))


def unit(st, ans, resc, prices):
    R = ans["results"]
    pin, pout = prices
    return {"correct": [R[s]["correct"] for s in ("C0", "C1", "C2")],
            "cost": [tok_cost(R[s], pin, pout) for s in ("C0", "C1", "C2")],
            "calls": [R[s]["calls"] for s in ("C0", "C1", "C2")],
            "searches": [R[s]["searches"] for s in ("C0", "C1", "C2")],
            "probs3": st["probs3"], "p_need_c2": st["p_need_c2"],
            "p_rescue": resc["p_rescue"] if resc else float("nan"),
            "rewrite_cost": tok_cost(st.get("rewrite_usage"), pin, pout), "rewrote": st["rewrote"],
            "n_search": len(st["search"])}


def build_report(items, states, answers, rescue, policy, lam, prices, out, meta):
    keep, U = [], {v: [] for v in VARIANTS}
    for it in items:
        sts = [states.get(f"{it['id']}|{v}") for v in VARIANTS]
        if any(s is None or akey(it["id"], s) not in answers for s in sts):
            continue                                            # only questions with ALL inputs answered
        keep.append(it)
        for v, st in zip(VARIANTS, sts):
            k = akey(it["id"], st)
            U[v].append(unit(st, answers[k], rescue.get(k), prices))
    if not keep:
        raise SystemExit("no question has all four inputs answered yet")
    A = {v: arrays(U[v]) for v in VARIANTS}
    egy_r = [states[it["id"] + "|egy_r"] for it in keep]
    shares = " / ".join(f"{100 * np.mean([st['dialect'] == d for st in egy_r]):.1f}%" for d in ("egy", "unsure", "msa"))
    judged = [answers[akey(it["id"], states[it["id"] + "|" + v])]["results"][s]["judge"]
              for it in keep for v in VARIANTS for s in ("C0", "C1", "C2")]
    meta = {**meta, "questions with all inputs answered": len(keep),
            "local routing time per question (median, GPU)": f"{np.median([st['sec'] for st in egy_r]):.2f}s",
            "Egyptian inputs rewritten / unsure (both) / as typed": shares,
            "judge methods": json.dumps({m: judged.count(m) for m in ("rule", "llm", "llm_unparsed")})}
    text, js = report(A, policy["thresholds"], lam, [it["set"] for it in keep], meta)
    out = Path(out)
    (out / "final_report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "final_report.json").write_text(json.dumps(js, indent=1, ensure_ascii=False), encoding="utf-8")
    return text


def main(argv=None, router=None, complete_fn=None, retriever=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--n_ardqa", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--budget_usd", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--text_run", default="runs_router/bert-base-arabertv02_seed0")
    ap.add_argument("--heldout_ids", default="data/egy_test/heldout_ids.json")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs_phase8")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    from arroute.llm import PRICES, Budget
    prices = PRICES.get(model)
    if prices is None:
        raise SystemExit(f"no price for {model}: add it to arroute/llm.py PRICES")
    items = select_items(a.test, a.n_ardqa, a.seed, a.limit)
    files = {k: out / f"{k}.jsonl" for k in ("states", "answers", "rescue")}
    policy = json.loads((Path(a.models) / "complexity_router" / "policy.json").read_text(encoding="utf-8")) \
        if router is None else {"thresholds": router.th}
    print(f"{len(items):,} test questions | generator + judge: {model} | budget ${a.budget_usd:.2f}")
    t0 = time.time()
    stopped, failed = False, 0
    if a.report_only:
        states = read_jsonl(files["states"], "key")
        answers, rescue = read_jsonl(files["answers"], "akey"), read_jsonl(files["rescue"], "akey")
    else:
        if not os.environ.get("OPENAI_API_KEY") and complete_fn is None:
            raise SystemExit("set $env:OPENAI_API_KEY first")
        from arroute.egy.rewrite import rewrite_llm_usage
        from label_outcomes import Generator
        if complete_fn is None:
            from arroute.llm import complete_with_usage as complete_fn
        budget = Budget(a.budget_usd)
        gen_rw = {"fn": complete_fn, "model": model, "budget": budget}
        rw_usage = {}

        def rewrite(q):
            t, u, _ = rewrite_llm_usage(q, gen_rw)
            rw_usage[q] = u
            return t
        if router is None:
            from arroute.pipeline.arrouter import load_router
            router = load_router(a.models, a.corpus, rewrite=rewrite)
        else:
            router.rewrite = rewrite
        retriever = retriever or router.r
        states = route_all(router, items, files["states"], rw_usage)
        gen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 300})   # = labelling
        jgen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 60})
        answers, stopped, failed = answer_all(items, states, files["answers"], gen, jgen, retriever, a.workers)
        rescue = rescue_all(router, items, states, answers, files["rescue"])
        print(f"spend this run ${budget.spent:.4f} of ${a.budget_usd:.2f}  ({time.time() - t0:.0f}s)")
    lam = adaptive_lambda(a.labels, a.text_run, a.heldout_ids)
    meta = {"generator / judge": model, "seed": a.seed, "n_ardqa sampled": a.n_ardqa,
            "Adaptive-RAG tuned lambda (chosen on training questions)": lam,
            "complexity thresholds": "models/complexity_router/policy.json (chosen on training questions only)",
            "stopped by budget": stopped, "failed plans (re-run retries)": failed}
    text = build_report(items, states, answers, rescue, policy, lam, prices, out, meta)
    print("\n" + text)
    if stopped:
        print("\nThe budget stopped the run: the report covers the questions finished so far. Re-run with a higher "
              "-Budget to finish (finished answers are reused for free).")


if __name__ == "__main__":
    main()
