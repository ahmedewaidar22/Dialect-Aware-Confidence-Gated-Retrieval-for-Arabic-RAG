"""Phase 8: the final end-to-end comparison on Egyptian input (torch-free: pure analysis of stored outcomes).

Every test question is answered with ALL three strategies (C0, C1, C2) under each INPUT VARIANT:
  egy_t  Egyptian, searched as typed           (what standard RAG / Adaptive-RAG see)
  egy_r  Egyptian, through the dialect plan    (rewrite / unsure-both / as typed: what the router sees)
  msa_t  MSA, as typed                         (upper bound: the same question asked in MSA)
  msa_r  MSA, through the dialect plan         (the router on MSA traffic: must not get worse)
So every system - fixed strategies, Adaptive-RAG, the router at each operating point, the oracle - is scored
offline from the same answers, paired question by question. Cost = generator tokens at list price, PLUS the
rewrite call when the plan rewrote (judge calls are evaluation, not system cost).
"""
import numpy as np

from .metrics import wilson
from .outcome import PRICE_IN, PRICE_OUT, STRATS, choose_cautious, fixed_frontier, frontier_acc, mcnemar_p, oracle_choice
from .policy import decide, outcome

VARIANTS = ("egy_t", "egy_r", "msa_t", "msa_r")
MODES = ("budget", "match_c1", "balanced", "quality")


def tok_cost(usage, pin=PRICE_IN, pout=PRICE_OUT):
    usage = usage or {}
    return (usage.get("input_tokens", 0) * pin + usage.get("output_tokens", 0) * pout) / 1e6


def arrays(units):
    """units: list (one per question, same order for every variant) of dicts with
    correct[3], cost[3], calls[3], searches[3], probs3, p_need_c2, p_rescue, rewrite_cost, rewrote, n_search."""
    g = lambda k: np.array([u[k] for u in units], float)                    # noqa: E731
    a = {k: g(k) for k in ("correct", "cost", "calls", "searches", "probs3")}
    a["correct"] = a["correct"].astype(bool)
    a["p_c0"] = a["probs3"][:, 0]
    a["p_need_c2"] = g("p_need_c2")
    a["p_rescue"] = np.nan_to_num(g("p_rescue"), nan=-1.0)                 # no score (C1 failed to run) = never
    a["rewrite_cost"] = g("rewrite_cost")
    a["rewrote"] = g("rewrote")
    a["extra_search"] = g("n_search") - 1                                  # 'unsure' band searches twice
    return a


def _thr(t):
    return tuple(np.inf if t.get(k) is None else float(t[k]) for k in ("ta", "tb", "tc"))


def run_policy(a, kind, thresholds=None, lam=None, s=None):
    """-> dict hit, cost, calls, searches (per question), route (C0/C1/C1>C2/C2 per question)."""
    n = len(a["correct"])
    idx = np.arange(n)
    if kind in ("fixed", "argmax", "lambda", "oracle"):
        ch = {"fixed": lambda: np.full(n, s, int), "argmax": lambda: a["probs3"].argmax(1),
              "lambda": lambda: choose_cautious(a["probs3"], lam), "oracle": lambda: oracle_choice(a["correct"])}[kind]()
        hit, cost = a["correct"][idx, ch].astype(float), a["cost"][idx, ch]
        calls, srch = a["calls"][idx, ch], a["searches"][idx, ch]
        route = np.array(STRATS)[ch]
        uses_ret = ch > 0
    elif kind == "router":
        skip, c2, esc = decide(a["p_c0"], a["p_need_c2"], a["p_rescue"], _thr(thresholds))
        hit, cost, calls = outcome(a["correct"], a["cost"], a["calls"], skip, c2, esc)
        srch = np.where(skip, 0, np.where(c2, a["searches"][:, 2], a["searches"][:, 1] + esc * a["searches"][:, 2]))
        route = np.where(skip, "C0", np.where(c2, "C2", np.where(esc, "C1>C2", "C1")))
        uses_ret = ~skip
    else:
        raise ValueError(kind)
    return {"hit": hit, "cost": cost + a["rewrite_cost"], "calls": calls + a["rewrote"],
            "searches": srch + uses_ret * a["extra_search"], "route": route}


def paired(x, y, reps=2000, seed=0):
    """x, y: per-question hits (paired). -> (diff, ci_lo, ci_hi, mcnemar p)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    p = mcnemar_p(int((x & ~y).sum()), int((~x & y).sum()))
    rng = np.random.RandomState(seed)
    d = x.astype(float) - y.astype(float)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(reps)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p


def systems(A, thresholds, lam):
    """The rows of the final tables. A: variant -> arrays(). -> list of (section, name, variant, result)."""
    rows = []

    def add(sec, name, v, **kw):
        if v in A:
            rows.append((sec, name, v, run_policy(A[v], **kw)))

    E = "EGYPTIAN INPUT - main comparison"
    add(E, "No retrieval (always C0)", "egy_t", kind="fixed", s=0)
    add(E, "Standard RAG (always C1)", "egy_t", kind="fixed", s=1)
    add(E, "Iterative RAG (always C2)", "egy_t", kind="fixed", s=2)
    add(E, "Adaptive-RAG (text classifier, argmax)", "egy_t", kind="argmax")
    if lam is not None:
        add(E, "Adaptive-RAG (text classifier, tuned)", "egy_t", kind="lambda", lam=lam)
    for m in MODES:
        add(E, f"YOUR ROUTER - {m}", "egy_r", kind="router", thresholds=thresholds[m])
    add(E, "Oracle (perfect routing, upper bound)", "egy_r", kind="oracle")
    B = "EGYPTIAN INPUT - ablations (which part helps?)"
    for m in MODES:
        add(B, f"complexity router, NO rewrite - {m}", "egy_t", kind="router", thresholds=thresholds[m])
    add(B, "dialect plan + standard RAG (C1)", "egy_r", kind="fixed", s=1)
    add(B, "dialect plan + iterative RAG (C2)", "egy_r", kind="fixed", s=2)
    add(B, "dialect plan + Adaptive-RAG (argmax)", "egy_r", kind="argmax")
    M = "MSA INPUT - same questions asked in MSA (upper bound; the router must not hurt MSA users)"
    add(M, "Standard RAG (always C1)", "msa_t", kind="fixed", s=1)
    add(M, "Iterative RAG (always C2)", "msa_t", kind="fixed", s=2)
    add(M, "Adaptive-RAG (text classifier, argmax)", "msa_t", kind="argmax")
    for m in MODES:
        add(M, f"YOUR ROUTER - {m}", "msa_r", kind="router", thresholds=thresholds[m])
    return rows


def _find(rows, sec_start, name):
    return next((r[3] for r in rows if r[0].startswith(sec_start) and r[1] == name), None)


def report(A, thresholds, lam, sets, meta=None):
    """-> (text, json-able dict). sets: per-question test-set name ('ardqa', 'pilot', 'heldout')."""
    rows = systems(A, thresholds, lam)
    sets = np.asarray(sets)
    n = len(sets)
    std = _find(rows, "EGYPTIAN INPUT - main", "Standard RAG (always C1)")
    ada = _find(rows, "EGYPTIAN INPUT - main", "Adaptive-RAG (text classifier, argmax)")
    hulls = {lang: fixed_frontier(A[f"{lang}_t"]["correct"], A[f"{lang}_t"]["cost"], A[f"{lang}_t"]["calls"])
             for lang in ("egy", "msa") if f"{lang}_t" in A}
    bases = {"egy": (std, ada), "msa": (_find(rows, "MSA INPUT", "Standard RAG (always C1)"),
                                        _find(rows, "MSA INPUT", "Adaptive-RAG (text classifier, argmax)"))}
    L = [f"PHASE 8 - FINAL COMPARISON ({n:,} questions: " + ", ".join(f"{s} {int((sets == s).sum())}" for s in
                                                                     ("ardqa", "pilot", "heldout")) + ")",
         "accuracy = judged correct answers; $ per 1,000 questions = generator tokens (+ the rewrite call) at "
         f"${PRICE_IN}/${PRICE_OUT} per 1M tokens",
         "'vs mix' = accuracy minus the best RANDOM mix of always-C0/C1/C2 (same input language, as typed) at the same cost",
         "'vs standard RAG' / 'vs Adaptive-RAG' = against those systems on the SAME input language (Egyptian rows vs "
         "Egyptian, MSA rows vs MSA)", "p = exact McNemar (paired, same questions); [..] = 95% CI", ""]
    js = {"n": n, "sets": {s: int((sets == s).sum()) for s in set(sets.tolist())}, "lam": lam, "rows": []}
    sec = None
    for s_, name, v, r in rows:
        if s_ != sec:
            sec = s_
            L += [s_, f"  {'system':<44}{'input':>7}{'accuracy':>10}{'95% CI':>15}{'$/1k':>8}{'calls/q':>9}"
                      f"{'search/q':>9}{'vs mix':>8}   {'vs standard RAG':<22}{'vs Adaptive-RAG':<22}"]
        acc = r["hit"].mean()
        lo, hi = wilson(int(r["hit"].sum()), n)
        c1k = 1000 * r["cost"].mean()
        lang = v[:3]
        vm = acc - frontier_acc(hulls[lang], c1k)
        cmp = []
        for base in bases[lang]:
            if base is None or base is r:
                cmp.append(f"{'-':<22}")
            else:
                d, _, _, p = paired(r["hit"], base["hit"])
                cmp.append(f"{100 * d:+5.1f} p={p:<8.2g}".ljust(22))
        L.append(f"  {name:<44}{v:>7}{100 * acc:9.1f}%  [{100 * lo:4.1f}-{100 * hi:4.1f}]{c1k:8.3f}"
                 f"{r['calls'].mean():9.2f}{r['searches'].mean():9.2f}{100 * vm:+7.1f}   " + "".join(cmp))
        js["rows"].append({"section": s_, "system": name, "input": v, "accuracy": float(acc), "ci": [lo, hi],
                           "usd_per_1k": float(c1k), "calls": float(r["calls"].mean()),
                           "searches": float(r["searches"].mean()), "vs_mix": float(vm),
                           "route_share": {k: float((r["route"] == k).mean()) for k in ("C0", "C1", "C1>C2", "C2")}})
    L.append("")

    # paired CIs for the headline comparisons
    L.append("HEADLINE DIFFERENCES (paired bootstrap 95% CI)")
    for m in MODES:
        rr = _find(rows, "EGYPTIAN INPUT - main", f"YOUR ROUTER - {m}")
        for bname, base in (("standard RAG", std), ("Adaptive-RAG", ada)):
            if rr is None or base is None:
                continue
            d, lo, hi, p = paired(rr["hit"], base["hit"])
            cr = rr["cost"].mean() / max(base["cost"].mean(), 1e-12)
            L.append(f"  router {m:<9} vs {bname:<13} accuracy {100 * d:+5.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                     f"p={p:.2g} | cost x{cr:.2f}")
    L.append("")

    # per test set
    key = [("Standard RAG (always C1)", "EGYPTIAN INPUT - main"),
           ("Adaptive-RAG (text classifier, argmax)", "EGYPTIAN INPUT - main"),
           ("YOUR ROUTER - match_c1", "EGYPTIAN INPUT - main"), ("YOUR ROUTER - quality", "EGYPTIAN INPUT - main"),
           ("Standard RAG (always C1)", "MSA INPUT"), ("YOUR ROUTER - quality", "MSA INPUT")]
    L.append("BY TEST SET (accuracy %)   ardqa = native Egyptian (most realistic) | pilot = native | heldout = LLM-translated")
    L.append(f"  {'system':<44}{'input':>7}" + "".join(f"{s:>10}" for s in ("ardqa", "pilot", "heldout")))
    for name, sp in key:
        r = _find(rows, sp, name)
        if r is None:
            continue
        v = "msa" if sp.startswith("MSA") else "egy"
        L.append(f"  {name:<44}{v:>7}" + "".join(f"{100 * r['hit'][sets == s].mean():9.1f}%" if (sets == s).any()
                                                  else f"{'-':>10}" for s in ("ardqa", "pilot", "heldout")))
    L.append("")

    # dialect gap at the ANSWER level
    msa_std = _find(rows, "MSA INPUT", "Standard RAG (always C1)")
    rw_std = _find(rows, "EGYPTIAN INPUT - ablations", "dialect plan + standard RAG (C1)")
    if msa_std is not None and std is not None and rw_std is not None:
        gap = msa_std["hit"].mean() - std["hit"].mean()
        closed = (rw_std["hit"].mean() - std["hit"].mean()) / gap if abs(gap) > 1e-9 else float("nan")
        L += ["DIALECT GAP AT THE ANSWER LEVEL (standard RAG, C1)",
              f"  MSA {100 * msa_std['hit'].mean():.1f}% vs Egyptian {100 * std['hit'].mean():.1f}%  -> gap "
              f"{100 * gap:+.1f} pts (p={paired(msa_std['hit'], std['hit'])[3]:.2g})",
              f"  with the dialect plan: {100 * rw_std['hit'].mean():.1f}%  -> {100 * closed:.0f}% of the gap closed", ""]
        js["gap"] = {"msa": float(msa_std["hit"].mean()), "egy": float(std["hit"].mean()),
                     "egy_plan": float(rw_std["hit"].mean()), "closed": float(closed)}

    # where the router sends Egyptian questions
    L.append("WHERE YOUR ROUTER SENDS EGYPTIAN QUESTIONS")
    for m in MODES:
        r = _find(rows, "EGYPTIAN INPUT - main", f"YOUR ROUTER - {m}")
        if r is not None:
            rc = r["cost"] - A["egy_r"]["rewrite_cost"]
            L.append(f"  {m:<9}" + " | ".join(f"{k} {100 * (r['route'] == k).mean():5.1f}%" for k in ("C0", "C1", "C1>C2", "C2"))
                     + f"   | rewrite = {100 * A['egy_r']['rewrite_cost'].mean() / max(r['cost'].mean(), 1e-12):.0f}% "
                       f"of its cost (answering ${1000 * rc.mean():.3f}/1k)")
    if meta:
        L += ["", "RUN INFO"] + [f"  {k}: {v}" for k, v in meta.items()]
        js["meta"] = meta
    return "\n".join(L), js
