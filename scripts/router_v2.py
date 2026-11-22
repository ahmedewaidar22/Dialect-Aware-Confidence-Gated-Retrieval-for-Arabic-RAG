"""Router v2: confidence-aware cascade (C0 first -> C1 on RERANKED top-k -> C2), compared with Router v1 on the
SAME 1,000 Phase 8 test questions.

  python scripts/router_v2.py --stage prepare            # GPU, free: search + reranker for every question; choose k
  python scripts/router_v2.py --stage answer --budget_usd 6    # API: C0 + C1R with token log-probs (+ C2 for ArDQA calib)
  python scripts/router_v2.py --stage features           # GPU, free: reranker check of C1R's answer + answer checks
  python scripts/router_v2.py --stage report             # CPU, free: train detectors, thresholds, final report

Questions:
  train  the labelled training questions (Phase 3; the 300 held-out ones excluded)            -> fit
  calib  the 667 native Egyptian ArDQA questions NOT in the Phase 8 test sample              -> fit
  test   the 1,000 Phase 8 test questions, Egyptian and MSA input through the dialect plan   -> evaluation only
Detectors and thresholds are chosen ONLY on train + calib (out-of-fold). The test answers are never used to
choose anything. Every stage is resumable; LLM calls are cached. Writes runs_v2/.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute import free as _free                                  # noqa: E402
_free.register()                                                # $0 price for a free gateway model
import eval_phase8 as E                                         # noqa: E402

from arroute.router import v2 as V                              # noqa: E402

KEEP = ("p_egy", "dialect", "query", "search", "probs3", "cands", "rr20", "x_pre")


def jl_append(path, rec):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_items(a):
    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    test = E.select_items(a.test, a.n_ardqa, a.seed)
    test_ids = {it["id"] for it in test}
    calib = [it for it in items if it["set"] == "ardqa" and it["id"] not in test_ids and it.get("refs")
             and it.get("gold_pids") and "rejected" not in (it.get("review") or {}).values()]
    return test, calib


# ---------------------------------------------------------------- stage: prepare (GPU, free)
def stage_prepare(a, out, router=None, log=print):
    from arroute.router.outcome import load_outcomes
    rows_f = out / "rows.jsonl"
    done = E.read_jsonl(rows_f, "key")
    test, calib = load_items(a)
    held = set(json.loads(Path(a.heldout_ids).read_text(encoding="utf-8"))) if Path(a.heldout_ids).exists() else set()
    recs = [r for r in load_outcomes(a.labels) if r["id"] not in held]
    try:
        from build_egy_testset import gold_map
        gold = gold_map(a.processed, [r["id"] for r in recs])
    except Exception:                                           # noqa: BLE001
        gold = {}
    todo = [("train", r) for r in recs if f"train|{r['id']}" not in done] + \
           [("calib", it) for it in calib if f"calib|{it['id']}" not in done]
    log(f"[prepare] train {len(recs):,} | calib {len(calib):,} | test {len(test):,} x 2 inputs | {len(todo):,} to search")
    if todo and router is None:
        router = load_router_cached(a)
    t0 = time.time()
    for n, (split, x) in enumerate(todo, 1):
        if split == "train":
            st = router.route(x["question"], force="typed", full=True)
            row = {"key": f"train|{x['id']}", "split": "train", "id": x["id"], "set": x["source"], "judge_q": x["question"],
                   "refs": x["refs"], "gold": gold.get(x["id"], []), "rewrote": 0, "rewrite_usage": {}}
        else:
            st = router.route(x["q_egy"], full=True)
            row = {"key": f"calib|{x['id']}", "split": "calib", "id": x["id"], "set": "ardqa", "judge_q": x["q_msa"],
                   "refs": x["refs"], "gold": x["gold_pids"], "rewrote": int(st["dialect"] in ("egy", "unsure")),
                   "rewrite_usage": {}}
        row.update({k: st.get(k) for k in KEEP})
        done[row["key"]] = row
        jl_append(rows_f, row)
        if n % 200 == 0:
            log(f"  {n:,}/{len(todo):,} ({time.time() - t0:.0f}s)")
    # test rows come straight from Phase 8 (same search, same reranker scores)
    st8 = E.read_jsonl(Path(a.phase8) / "states.jsonl", "key")
    for it in test:
        for v in ("egy_r", "msa_r"):
            k = f"test|{it['id']}|{v}"
            s = st8.get(f"{it['id']}|{v}")
            if k in done or s is None:
                continue
            row = {"key": k, "split": "test", "id": it["id"], "variant": v, "set": it["set"], "judge_q": it["q_msa"],
                   "refs": it["refs"], "gold": it.get("gold_pids", []), "rewrote": s["rewrote"],
                   "rewrite_usage": s.get("rewrite_usage", {}), "akey8": E.akey(it["id"], s),
                   **{kk: s.get(kk) for kk in KEEP}}
            done[k] = row
            jl_append(rows_f, row)
    k = choose_k(done, out, a.k, log)
    return done, k


def choose_k(rows, out, k_forced, log=print):
    """How many RERANKED passages match hybrid top-5 at finding the gold passage? On train + calib only."""
    fit = [r for r in rows.values() if r["split"] in ("train", "calib") and r.get("gold") and r.get("rr20")]
    L = ["RERANKED top-k vs HYBRID top-5: gold passage found (train + calib questions only; test not used)",
         f"  questions with a known gold passage: {len(fit):,}",
         f"  {'k':>3}{'hybrid order':>15}{'reranked':>11}"]
    hyb5 = np.mean([bool(set(r["cands"][:5]) & set(r["gold"])) for r in fit]) if fit else 0.0
    best = None
    for k in range(1, 6):
        h = np.mean([bool(set(r["cands"][:k]) & set(r["gold"])) for r in fit]) if fit else 0.0
        rr = np.mean([bool(set(V.reranked(r["cands"], r["rr20"], k)) & set(r["gold"])) for r in fit]) if fit else 0.0
        L.append(f"  {k:>3}{100 * h:14.1f}%{100 * rr:10.1f}%")
        if best is None and k >= 2 and rr >= hyb5 - 0.005:
            best = k
    k = k_forced or best or 5
    L.append(f"\n  chosen k = {k}  (smallest k >= 2 whose reranked top-k finds the gold passage as often as hybrid top-5"
             + (", OVERRIDDEN by --k" if k_forced else "") + ")")
    (out / "k_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "v2_config.json").write_text(json.dumps({"k": k, "hybrid_top5": hyb5}, indent=1), encoding="utf-8")
    log("\n".join(L))
    return k


def load_router_cached(a):
    from arroute.egy.rewrite import rewrite_llm
    from arroute.llm import Budget, complete_with_usage
    from arroute.pipeline.arrouter import load_router
    budget = Budget(0.5)                         # rewrites are cached from Phase 6; this only guards surprises
    gen_rw = {"fn": complete_with_usage, "model": os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini"), "budget": budget}
    return load_router(a.models, a.corpus, rewrite=lambda q: rewrite_llm(q, gen_rw))


# ---------------------------------------------------------------- stage: answer (API)
def answer_row(row, k, ask, gen, jgen, retriever, lock):
    from arroute.pipeline.judge import judge
    from arroute.pipeline.strategies import run_c2
    pids = V.reranked(row["cands"], row["rr20"], k)
    texts = [retriever.text(p) for p in pids]
    out = {"C0L": V.run_c0_lp(row["query"], ask), "C1R": V.run_c1r_lp(row["query"], pids, texts, ask)}
    if row["split"] == "calib":                                  # train has C2 in the labels, test in Phase 8
        r = run_c2(row["query"], gen, E.PlanRetriever(retriever, lock, row["query"], row["cands"]))
        out["C2"] = {kk: r[kk] for kk in ("answer", "calls", "searches", "input_tokens", "output_tokens")}
    for name, r in out.items():
        ok, how, _ = judge(row["judge_q"], r["answer"], row["refs"], jgen)
        r.update(correct=bool(ok), judge=how)
    return out


def stage_answer(a, out, rows, k, lp_fn=None, complete_fn=None, retriever=None, log=print):
    from arroute.llm import Budget, BudgetExceeded
    from label_outcomes import Generator
    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    if lp_fn is None:
        from arroute.llm import complete_logprobs as lp_fn
    if complete_fn is None:
        from arroute.llm import complete_with_usage as complete_fn
    budget = Budget(a.budget_usd)
    ask = lambda p: lp_fn(p, model=model, budget=budget)                               # noqa: E731
    gen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 300})
    jgen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 60})
    if retriever is None:
        from arroute.retrieval.index import load_retriever
        retriever = load_retriever(a.corpus, dense=True)
    ans_f = out / "answers.jsonl"
    done = E.read_jsonl(ans_f, "key")
    order = {"test": 0, "calib": 1, "train": 2}                  # test first: the comparison matters most
    todo = sorted([r for r in rows.values() if r["key"] not in done], key=lambda r: order[r["split"]])
    log(f"[answer] {len(done):,} saved, {len(todo):,} to go | k={k} | budget ${a.budget_usd:.2f}")
    if todo:                                                     # fail fast: one call before going parallel
        from arroute.pipeline.strategies import P_C0
        try:
            ask(P_C0.format(q=todo[0]["query"]))
        except BudgetExceeded:
            raise
        except Exception as ex:                                  # noqa: BLE001
            raise SystemExit(f"the log-prob call failed: {type(ex).__name__}: {str(ex)[:300]}")
    lock, stopped, failed, t0 = threading.Lock(), False, 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(answer_row, r, k, ask, gen, jgen, retriever, lock): r["key"] for r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                res = fut.result()
            except BudgetExceeded:
                if not stopped:
                    log("  budget reached: no new calls; finishing the calls in flight...")
                stopped = True
                pool.shutdown(wait=False, cancel_futures=True)
                continue
            except Exception as ex:                              # noqa: BLE001
                failed += 1
                log(f"  skipped {key}: {type(ex).__name__}: {str(ex)[:150]}  (a re-run retries it)")
                if failed == 1 and n <= 3:
                    log("  (if every question fails like this, the API key/model or the logprobs call is the problem)")
                continue
            done[key] = {"key": key, "k": k, "results": res}
            jl_append(ans_f, done[key])
            if n % 200 == 0:
                log(f"  {n:,}/{len(todo):,} ({time.time() - t0:.0f}s)")
    log(f"[answer] spend ${budget.spent:.4f} of ${a.budget_usd:.2f}" + ("  (STOPPED by the budget)" if stopped else ""))
    return done, stopped


# ---------------------------------------------------------------- stage: features (GPU, free)
def stage_features(a, out, rows, answers, router=None, log=print):
    feats_f = out / "features.jsonl"
    done = E.read_jsonl(feats_f, "key")
    todo = [k for k in answers if k not in done and k in rows]
    log(f"[features] {len(todo):,} to compute")
    if todo and router is None:
        router = load_router_cached(a)
    for key in todo:
        row, res = rows[key], answers[key]["results"]
        pids = res["C1R"]["passages"]
        texts = [router.r.text(p) for p in pids]
        a1, a0 = res["C1R"]["answer"], res["C0L"]["answer"]
        rr_top = max(row["rr20"][row["cands"].index(p)] for p in pids) if pids else 0.0
        rr_ans = None
        if a1.strip() and V.IDK not in a1:
            rr_ans = [float(x) for x in router.reranker([(f"{row['query']} الجواب: {a1}", t) for t in texts])]
        done[key] = {"key": key, **V.answer_checks(a1, a0, texts, rr_ans, rr_top)}
        jl_append(feats_f, done[key])
    return done


# ---------------------------------------------------------------- stage: report (CPU, free)
def oof_text_probs(a, ids):
    """Training questions: the text head's OUT-OF-FOLD probabilities (Phase 4a), never its own training fit."""
    from arroute.router.outcome import load_outcomes
    from arroute.router.route_report import load_run
    if not (Path(a.text_run) / "oof.npz").exists():
        return {}
    recs = [r for r in load_outcomes(a.labels) if r["id"] in set(ids)]
    run = load_run(a.text_run, recs)
    return {r["id"]: [float(x) for x in p] for r, p in zip(recs, run["probs"])}


def build(rows, answers, feats, labels_c2, phase8_answers, oof_p):
    """-> per split: X0 rows (dict), outcome dicts. Skips rows that are not complete."""
    data = {}
    for key, row in rows.items():
        if key not in answers or key not in feats:
            continue
        res = answers[key]["results"]
        if row["split"] == "train":
            c2 = labels_c2.get(row["id"])
        elif row["split"] == "calib":
            c2 = res.get("C2")
        else:
            c2 = (phase8_answers.get(row.get("akey8")) or {}).get("results", {}).get("C2")
        if c2 is None:
            continue
        probs = oof_p.get(row["id"], row["probs3"]) if row["split"] == "train" else row["probs3"]
        f = V.pre_features(probs, row["x_pre"], row["rr20"])
        f.update({f"c0_{k}": v for k, v in res["C0L"]["conf"].items()})
        f.update({f"c1_{k}": v for k, v in res["C1R"]["conf"].items()})
        f.update({k: feats[key][k] for k in V.CHECKS})
        strat = [res["C0L"], res["C1R"], c2]
        data[key] = {"row": row, "f": f, "correct": [bool(s["correct"]) for s in strat],
                     "cost": [V.tok_cost(s) for s in strat], "calls": [s.get("calls", 1) for s in strat],
                     "searches": [s.get("searches", 0) for s in strat],
                     "rewrite_cost": V.tok_cost(row.get("rewrite_usage") or {}), "rewrote": row.get("rewrote", 0)}
    return data


def mats(ds, feats):
    X = np.array([[d["f"][k] for k in feats] for d in ds], float)
    OC = {k: np.array([d[k] for d in ds], float) for k in ("correct", "cost", "calls", "searches")}
    OC["correct"] = OC["correct"].astype(bool)
    return X, OC


def fit_variant(fit, feats0, feats1, seed=0):
    """OOF scores on the fit set -> thresholds per rule; final detectors on the whole fit set."""
    from arroute.router.v2 import _norm
    X0, OC = mats(fit, feats0)
    X1, _ = mats(fit, feats1)
    groups = np.unique([_norm(d["row"]["judge_q"]) for d in fit], return_inverse=True)[1]
    p0 = V.oof_scores(X0, OC["correct"][:, 0], groups, seed=seed)
    p1 = V.oof_scores(X1, OC["correct"][:, 1], groups, seed=seed)
    th = {rule: V.search(p0, p1, OC, rule) for rule in V.RULES}
    m0 = V.make_model(seed).fit(X0, OC["correct"][:, 0])
    m1 = V.make_model(seed).fit(X1, OC["correct"][:, 1])
    return {"m0": m0, "m1": m1, "th": th, "feats0": feats0, "feats1": feats1, "oof": (p0, p1, OC)}


def stage_report(a, out, rows, answers, feats, log=print):
    from arroute.router.final_eval import arrays, paired, run_policy
    from arroute.router.metrics import wilson
    from arroute.router.outcome import fixed_frontier, frontier_acc, load_outcomes
    from arroute.router.route_report import auc
    labels_c2 = {r["id"]: r["results"]["C2"] for r in load_outcomes(a.labels)}
    p8 = Path(a.phase8)
    st8, an8 = E.read_jsonl(p8 / "states.jsonl", "key"), E.read_jsonl(p8 / "answers.jsonl", "akey")
    rs8 = E.read_jsonl(p8 / "rescue.jsonl", "akey")
    oof_p = oof_text_probs(a, [r["id"] for r in rows.values() if r["split"] == "train"])
    data = build(rows, answers, feats, labels_c2, an8, oof_p)
    fit_all = [d for d in data.values() if d["row"]["split"] in ("train", "calib")]
    fit_train = [d for d in fit_all if d["row"]["split"] == "train"]
    cfg = json.loads((out / "v2_config.json").read_text(encoding="utf-8")) if (out / "v2_config.json").exists() else {}
    log(f"[report] fit: train {len(fit_train):,} + calib {len(fit_all) - len(fit_train):,} | text-head OOF probs for "
        f"{len(oof_p):,} training questions")
    variants = {"v2": fit_variant(fit_all, V.F0, V.F1),
                "no ArDQA calibration data": fit_variant(fit_train, V.F0, V.F1),
                "no LLM-confidence features": fit_variant(fit_all, V.NO_CONF(V.F0), V.NO_CONF(V.F1))}

    # ---- test: questions complete in Phase 8 AND v2 (both inputs)
    test, _ = load_items(a)
    ids = [it["id"] for it in test if all(f"test|{it['id']}|{v}" in data for v in ("egy_r", "msa_r"))
           and all(f"{it['id']}|{v}" in st8 and E.akey(it["id"], st8[f"{it['id']}|{v}"]) in an8 for v in E.VARIANTS)]
    sets = np.array([next(it["set"] for it in test if it["id"] == i) for i in ids])
    if not ids:
        raise SystemExit("no test question is complete yet: run the answer and features stages")
    n = len(ids)
    pol = json.loads((Path(a.models) / "complexity_router" / "policy.json").read_text(encoding="utf-8"))
    U = {v: [E.unit(st8[f"{i}|{v}"], an8[E.akey(i, st8[f"{i}|{v}"])], rs8.get(E.akey(i, st8[f"{i}|{v}"])),
                    (V.PRICE_IN, V.PRICE_OUT)) for i in ids] for v in E.VARIANTS}
    A = {v: arrays(U[v]) for v in E.VARIANTS}
    hull = {lang: fixed_frontier(A[f"{lang}_t"]["correct"], A[f"{lang}_t"]["cost"], A[f"{lang}_t"]["calls"])
            for lang in ("egy", "msa")}
    T = {v: [data[f"test|{i}|{v}"] for i in ids] for v in ("egy_r", "msa_r")}
    rows_out, js = [], {"n": n, "k": cfg.get("k"), "systems": [], "thresholds": {}}

    def add(sec, name, lang, hit, cost, calls, srch, route=None):
        rows_out.append((sec, name, lang, np.asarray(hit, float), np.asarray(cost, float), np.asarray(calls, float),
                         np.asarray(srch, float), route))

    for lang in ("egy", "msa"):
        sec = "EGYPTIAN INPUT" if lang == "egy" else "MSA INPUT (same questions in MSA: must not get worse)"
        t, r = A[f"{lang}_t"], A[f"{lang}_r"]
        for s, nm in ((1, "Standard RAG (C1, hybrid top-5)"), (2, "Iterative RAG (always C2)")):
            add(sec, nm, lang, t["correct"][:, s], t["cost"][:, s], t["calls"][:, s], t["searches"][:, s])
        ch = t["probs3"].argmax(1)
        ix = np.arange(n)
        add(sec, "Adaptive-RAG (text classifier)", lang, t["correct"][ix, ch], t["cost"][ix, ch], t["calls"][ix, ch],
            t["searches"][ix, ch])
        for m in ("match_c1", "quality"):
            p = run_policy(r, "router", thresholds=pol["thresholds"][m])
            add(sec, f"Router v1 - {m}", lang, p["hit"], p["cost"], p["calls"], p["searches"], p["route"])
        X, OC = mats(T[f"{lang}_r"], V.F0)
        rw = np.array([d["rewrite_cost"] for d in T[f"{lang}_r"]])
        rwc = np.array([d["rewrote"] for d in T[f"{lang}_r"]], float)
        idk1 = np.array([d["f"]["c1_idk"] for d in T[f"{lang}_r"]], bool)
        add(sec, f"rewrite + C1R (reranked top-{cfg.get('k')})", lang, OC["correct"][:, 1], OC["cost"][:, 1] + rw,
            OC["calls"][:, 1] + rwc, OC["searches"][:, 1])
        add(sec, "rewrite + C1R, 'I don't know' -> C2", lang, np.where(idk1, OC["correct"][:, 2], OC["correct"][:, 1]),
            OC["cost"][:, 1] + idk1 * OC["cost"][:, 2] + rw, OC["calls"][:, 1] + idk1 * OC["calls"][:, 2] + rwc,
            OC["searches"][:, 1] + idk1 * OC["searches"][:, 2])
        for vname, var in variants.items():
            X0, _ = mats(T[f"{lang}_r"], var["feats0"])
            X1, _ = mats(T[f"{lang}_r"], var["feats1"])
            p0, p1 = var["m0"].predict_proba(X0)[:, 1], var["m1"].predict_proba(X1)[:, 1]
            if vname == "v2":
                pv1 = A[f"{lang}_r"]["p_c0"]
                js.setdefault("auc", {})[lang] = {
                    s: {"det0 (C0 right)": auc(p0[m], OC["correct"][m, 0]),
                        "v1 text head P(C0)": auc(pv1[m], OC["correct"][m, 0]),
                        "det1 (C1R right)": auc(p1[m], OC["correct"][m, 1]),
                        "'I don't know' only": auc(-idk1[m].astype(float), OC["correct"][m, 1])}
                    for s, m in (("all", np.ones(n, bool)), *[(x, sets == x) for x in ("ardqa", "pilot", "heldout")])}
            rules = V.RULES if vname == "v2" else ("balanced", "quality")
            for rule in rules:
                acc0, esc = V.decide(p0, p1, *var["th"][rule])
                hit, cost, calls, srch, route = V.outcome(OC, acc0, esc)
                nm = f"ROUTER V2 - {rule}" if vname == "v2" else f"  v2 ablation: {vname} - {rule}"
                add(sec, nm, lang, hit, cost + rw, calls + rwc, srch, route)
                js["thresholds"][f"{vname} | {rule}"] = var["th"][rule]
        orc = np.where(OC["correct"].any(1), OC["correct"].argmax(1), 0)
        add(sec, "Oracle (perfect cascade, upper bound)", lang, OC["correct"][ix, orc], np.array(
            [OC["cost"][i, :orc[i] + 1].sum() for i in range(n)]) + rw, orc + 1 + rwc, (orc > 0) * 1.0)

    # ---- text
    L = [f"ROUTER V2 vs ROUTER V1 - same {n:,} Phase 8 test questions (" + ", ".join(
        f"{s} {int((sets == s).sum())}" for s in ("ardqa", "pilot", "heldout")) + ")",
         f"C1R = C1 on the top-{cfg.get('k')} RERANKED passages (k chosen on training questions, see k_report.txt)",
         "cost = generator tokens (+ rewrite) per 1,000 questions; v2 always pays C0 first (it is ~$0.04/1k)",
         "'vs mix' = accuracy minus the best random mix of always-C0/C1/C2 (Phase 8, as typed) at the same cost",
         "p = exact McNemar (paired); detectors and thresholds were chosen on training + calibration questions only", ""]
    sec = None
    for s_, name, lang, hit, cost, calls, srch, route in rows_out:
        if s_ != sec:
            sec = s_
            L += [s_, f"  {'system':<50}{'accuracy':>9}{'95% CI':>15}{'$/1k':>8}{'calls/q':>9}{'search/q':>9}{'vs mix':>8}"
                      f"   {'vs standard RAG':<20}{'vs always C2':<20}"]
            base = next(x for x in rows_out if x[0] == s_ and x[1].startswith("Standard RAG"))
            c2 = next(x for x in rows_out if x[0] == s_ and x[1].startswith("Iterative RAG"))
        acc = hit.mean()
        lo, hi = wilson(int(hit.sum()), n)
        vm = acc - frontier_acc(hull[lang], 1000 * cost.mean())
        cmp = []
        for b in (base, c2):
            if b[1] == name:
                cmp.append(f"{'-':<20}")
            else:
                d, _, _, p = paired(hit.astype(bool), b[3].astype(bool))
                cmp.append(f"{100 * d:+5.1f} p={p:<9.2g}".ljust(20))
        L.append(f"  {name:<50}{100 * acc:8.1f}%  [{100 * lo:4.1f}-{100 * hi:4.1f}]{1000 * cost.mean():8.3f}"
                 f"{calls.mean():9.2f}{srch.mean():9.2f}{100 * vm:+7.1f}   " + "".join(cmp))
        js["systems"].append({"section": s_, "system": name, "accuracy": float(acc), "ci": [lo, hi],
                              "usd_per_1k": float(1000 * cost.mean()), "calls": float(calls.mean()),
                              "searches": float(srch.mean()), "vs_mix": float(vm),
                              "by_set": {s: float(hit[sets == s].mean()) for s in ("ardqa", "pilot", "heldout")},
                              "usd_by_set": {s: float(1000 * cost[sets == s].mean()) for s in ("ardqa", "pilot", "heldout")},
                              "route": None if route is None else {x: float((route == x).mean()) for x in np.unique(route)}})
    L += ["", "BY TEST SET (accuracy %  /  $ per 1k)   ardqa = native Egyptian | pilot = native | heldout = translated"]
    for x in js["systems"]:
        if x["section"].startswith("EGYPTIAN") and not x["system"].startswith("  v2 ablation"):
            L.append(f"  {x['system']:<50}" + "".join(f"{s:>8} {100 * x['by_set'][s]:5.1f}% / {x['usd_by_set'][s]:.3f}"
                                                       for s in ("ardqa", "pilot", "heldout")))
    L += ["", "DETECTOR QUALITY ON THE TEST SET (AUC; 0.5 = guessing)"]
    for lang in ("egy", "msa"):
        for s, d in js.get("auc", {}).get(lang, {}).items():
            L.append(f"  {lang} {s:<8}" + "  ".join(f"{k} {v:.2f}" for k, v in d.items()))
    L += ["", "WHERE ROUTER V2 SENDS EGYPTIAN QUESTIONS"]
    for x in js["systems"]:
        if x["section"].startswith("EGYPTIAN") and x["route"] and (x["system"].startswith("ROUTER V2") or "v1" in x["system"]):
            L.append(f"  {x['system']:<32}" + " | ".join(f"{k} {100 * v:5.1f}%" for k, v in sorted(x["route"].items())))
    L += ["", "THRESHOLDS (chosen on out-of-fold scores of the fit questions)"]
    L += [f"  {k}: accept C0 if P >= {v[0]:.3f}; escalate C1R->C2 if P < {v[1]:.3f}" for k, v in js["thresholds"].items()]
    text = "\n".join(L)
    (out / "v2_report.txt").write_text(text + "\n", encoding="utf-8")
    (out / "v2_report.json").write_text(json.dumps(js, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    log("\n" + text)
    return js


def main(argv=None, router=None, lp_fn=None, complete_fn=None, retriever=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prepare", "answer", "features", "report", "all"])
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--heldout_ids", default="data/egy_test/heldout_ids.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--phase8", default="runs_phase8")
    ap.add_argument("--text_run", default="runs_router/bert-base-arabertv02_seed0")
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--n_ardqa", type=int, default=500, help="must match Phase 8 (it defines the test sample)")
    ap.add_argument("--seed", type=int, default=42, help="must match Phase 8")
    ap.add_argument("--k", type=int, default=None, help="force the number of reranked passages")
    ap.add_argument("--budget_usd", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs_v2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "answer", "features", "report"] if a.stage == "all" else [a.stage]
    rows = E.read_jsonl(out / "rows.jsonl", "key")
    cfg = out / "v2_config.json"
    k = json.loads(cfg.read_text(encoding="utf-8"))["k"] if cfg.exists() else None
    if "prepare" in stages:
        rows, k = stage_prepare(a, out, router)
    if k is None:
        raise SystemExit("run --stage prepare first")
    answers = E.read_jsonl(out / "answers.jsonl", "key")
    if "answer" in stages:
        from arroute.llm import api_key, describe
        if not api_key() and lp_fn is None:
            raise SystemExit("set $env:ARROUTE_LLM_API_KEY (and $env:ARROUTE_LLM_BASE_URL for your own URL) first")
        print("LLM calls:", describe())
        answers, _ = stage_answer(a, out, rows, k, lp_fn, complete_fn, retriever or (router.r if router else None))
    feats = E.read_jsonl(out / "features.jsonl", "key")
    if "features" in stages:
        feats = stage_features(a, out, rows, answers, router)
    if "report" in stages:
        stage_report(a, out, rows, answers, feats)


if __name__ == "__main__":
    main()
