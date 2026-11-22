"""Turn out-of-fold router predictions into the RQ1 report (torch-free).

A 'run' directory holds oof.npz (ids, fold, logits, T, lam, lam_esc) + config.json, written by train_router.py.
A policy = (first strategy per question, escalate-on-'I don't know' yes/no).
"""
import json
import re
from pathlib import Path

import numpy as np

from .metrics import softmax, wilson
from .outcome import (bootstrap_vs, bootstrap_vs_mix, choose_argmax, choose_cautious, evaluate_routing, fixed_choice,
                      fixed_frontier, frontier_acc, idk_matrix, is_comparison, label_id, label_metrics,
                      mcnemar_p, oracle_choice, outcome_matrix, trainable)

LAM_GRID = [round(x, 2) for x in np.arange(0.05, 1.0001, 0.05)]


def load_run(run_dir, recs):
    d = np.load(Path(run_dir) / "oof.npz", allow_pickle=False)
    ids = [str(x) for x in d["ids"]]
    where = {i: k for k, i in enumerate(ids)}              # recs may be a SUBSET of the run (Phase 7 hold-out)
    missing = [r["id"] for r in recs if r["id"] not in where]
    if missing:
        raise ValueError(f"{run_dir}: {len(missing)} questions of the labels file are not in this run")
    src = np.array([where[r["id"]] for r in recs])
    logits, T, fold = d["logits"][src], d["T"][src], d["fold"][src]
    lam = d["lam"][src]
    lam_esc = (d["lam_esc"] if "lam_esc" in d.files else d["lam"])[src]
    probs = softmax(logits / T[:, None], 1.0)
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    return {"name": Path(run_dir).name, "probs": probs, "lam": lam, "lam_esc": lam_esc, "fold": fold, "config": cfg}


def auc(score, y):
    """Rank-based ROC AUC (ties averaged). nan if only one class is present."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    p, q = int(y.sum()), int((~y).sum())
    if p == 0 or q == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(score):                       # average ranks over ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - p * (p + 1) / 2) / (p * q))


def per_item_cautious(probs, lam):
    """λ chosen on each fold's inner dev set (no test information), applied item by item."""
    cum = np.cumsum(probs, axis=1)
    cum[:, -1] = 1.0
    return (cum >= lam[:, None] - 1e-12).argmax(axis=1)


def policies(recs, runs, correct):
    """name -> (choice, escalate?)."""
    n = len(recs)
    pol = {"always C0": (fixed_choice(n, 0), False),
           "always C1 (standard RAG)": (fixed_choice(n, 1), False),
           "always C2 (iterative)": (fixed_choice(n, 2), False),
           "cascade C1→C2 on IDK": (fixed_choice(n, 1), True),
           "cascade C0→C1→C2 on IDK": (fixed_choice(n, 0), True),
           "oracle (upper bound)": (oracle_choice(correct), False)}
    src = np.array([r["source"] for r in recs])
    if len(set(src.tolist())) > 1 and "hotpot_ctx_ar" in src:
        # reference only: uses the DATASET NAME, which a real system never sees. A router that merely
        # recognises dataset style can reach this, so a router must beat it to show real skill.
        pol["ref: dataset name → C1/C2"] = (np.where(src == "hotpot_ctx_ar", 2, 1), False)
    for run in runs:
        pol[f"{run['name']} | argmax"] = (choose_argmax(run["probs"]), False)
        pol[f"{run['name']} | λ dev"] = (per_item_cautious(run["probs"], run["lam"]), False)
        pol[f"{run['name']} | λ dev + IDK→up"] = (per_item_cautious(run["probs"], run["lam_esc"]), True)
    return pol


HEADER = (f"  {'policy':<40} {'accuracy':>8}  {'95% CI':>11}  {'$/1k q':>7}  {'calls':>5}  {'vs mix':>6}   "
          f"{'%C0':>5} {'%C1':>5} {'%C2':>5}")


def _row(name, ev, mix):
    lo, hi = wilson(round(ev["accuracy"] * ev["n"]), ev["n"])
    sh = ev["share"]
    return (f"  {name:<40} {100 * ev['accuracy']:7.1f}%  [{100 * lo:4.1f}-{100 * hi:4.1f}]  "
            f"${ev['cost_per_1k']:6.3f}  {ev['calls']:5.2f}  {100 * (ev['accuracy'] - mix):+5.1f}   "
            f"{100 * sh['C0']:5.1f} {100 * sh['C1']:5.1f} {100 * sh['C2']:5.1f}")


def _group_key(run):
    return re.sub(r"_seed\d+$", "", run["name"])


def build_report(recs, runs, none_as=None, reps=2000):
    correct, cost, calls = outcome_matrix(recs)
    idk = idk_matrix(recs)
    n = len(recs)
    out, js = [], {"n": n, "runs": {}}
    tr = np.array([trainable(r, none_as) for r in recs])
    comp = np.array([is_comparison(r) for r in recs])
    src = np.array([r["source"] for r in recs])
    y = np.array([label_id(r, none_as) if t else -1 for r, t in zip(recs, tr)])
    pol = policies(recs, runs, correct)

    out.append(f"ROUTER EVALUATION on {n:,} questions - out-of-fold: every prediction comes from a model "
               "that never saw that question.")
    out.append("cost = generator tokens only (gpt-4.1-mini prices) per 1,000 questions; calls = LLM calls per question")
    out.append("'vs mix' = accuracy minus the best RANDOM mix of fixed strategies that costs the same. A router is only")
    out.append("          useful if this is clearly > 0: otherwise a coin flip between C1 and C2 does as well.")
    out.append("'IDK→up' = if the chosen strategy answers 'لا أعرف', move up to the next one (and pay for both).\n")

    # ---- 1. label quality
    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    out.append(f"1) DOES THE HEAD PREDICT THE OUTCOME LABEL?  (trainable questions only: n={int(tr.sum())}; "
               "comparison and 'none' excluded)")
    out.append(f"   label mix: C0 {100 * prior[0]:.1f}%  C1 {100 * prior[1]:.1f}%  C2 {100 * prior[2]:.1f}%   "
               f"-> always guessing C1 = {100 * prior.max():.1f}% accuracy, "
               f"{100 * (2 * prior.max() / (1 + prior.max())) / 3:.1f}% macro-F1")
    out.append(f"   {'run':<36} {'acc':>6} {'macroF1':>8} {'bal.acc':>8}   recall C0/C1/C2")
    for run in runs:
        m = label_metrics(y[tr], run["probs"][tr].argmax(1))
        js["runs"].setdefault(run["name"], {})["label"] = m
        rc = m["recall"]
        out.append(f"   {run['name']:<36} {100 * m['accuracy']:5.1f}% {100 * m['macro_f1']:7.1f}% "
                   f"{100 * m['balanced_accuracy']:7.1f}%   {100 * rc['C0']:.0f}/{100 * rc['C1']:.0f}/{100 * rc['C2']:.0f}"
                   f"   confusion {m['confusion']}")

    # ---- 2. routing tables
    def table(mask, title, key):
        hull = fixed_frontier(correct[mask], cost[mask], calls[mask])
        out.append(f"\n{title}  (n={int(mask.sum())})")
        out.append(HEADER)
        res = {}
        for name, (ch, esc) in pol.items():
            ev = evaluate_routing(correct[mask], cost[mask], calls[mask], ch[mask], idk[mask] if esc else None)
            ev["vs_mix"] = ev["accuracy"] - frontier_acc(hull, ev["cost_per_1k"])
            res[name] = ev
            out.append(_row(name, ev, frontier_acc(hull, ev["cost_per_1k"])))
        js[key] = res

    table(np.ones(n, bool), "2) ROUTING - ALL questions (incl. 'none' and comparison)", "routing_all")
    for s in sorted(set(src.tolist())):
        table((src == s) & ~comp, f"   ROUTING - {s} (no comparison questions)", f"routing_{s}")
    if comp.any():
        table(comp, "   ROUTING - HotpotQA comparison questions only (never trained on)", "routing_comparison")

    # ---- 3. significance
    out.append(f"\n3) PAIRED TESTS on all questions (bootstrap {reps} reps for CIs, exact McNemar for accuracy)")
    refs = ["always C1 (standard RAG)", "always C2 (iterative)", "cascade C1→C2 on IDK"]
    for name, (ch, esc) in pol.items():
        if " | " not in name:
            continue
        vm = bootstrap_vs_mix(correct, cost, calls, ch, idk if esc else None, reps=max(200, reps // 2))
        js["runs"].setdefault(name.split(" | ")[0], {}).setdefault("vs_mix", {})[name] = vm
        sig = "BETTER than random mixing" if vm["ci"][0] > 0 else "not distinguishable from random mixing"
        out.append(f"   {name:<40} vs random mix at same cost: {100 * vm['vs_mix']:+5.1f} pts "
                   f"[{100 * vm['ci'][0]:+.1f},{100 * vm['ci'][1]:+.1f}]  -> {sig}")
        for bname in refs:
            bch, besc = pol[bname]
            b = bootstrap_vs(correct, cost, ch, bch, reps=reps, calls=calls,
                             idk_a=idk if esc else None, idk_b=idk if besc else None)
            p = mcnemar_p(b["b_only_a"], b["c_only_b"])
            js["runs"].setdefault(name.split(" | ")[0], {}).setdefault("vs", {})[f"{name} vs {bname}"] = {**b, "mcnemar_p": p}
            out.append(f"   {name:<40} vs {bname:<25}: acc {100 * b['acc_diff']:+5.1f} pts "
                       f"[{100 * b['acc_diff_ci'][0]:+.1f},{100 * b['acc_diff_ci'][1]:+.1f}] p={p:.2g} | "
                       f"cost x{b['cost_ratio']:.2f} [{b['cost_ratio_ci'][0]:.2f}-{b['cost_ratio_ci'][1]:.2f}]")

    # ---- 4. lambda sweep
    # ---- 3b. per-decision skill, overall and inside each dataset
    y0 = correct[:, 0]
    need2 = ~correct[:, 1] & correct[:, 2]
    c12 = np.array([r["label"] in ("C1", "C2") for r in recs]) & ~comp
    out.append("\n3b) WHICH DECISION DOES EACH ROUTER GET RIGHT?  AUC: 0.5 = coin flip, 1.0 = perfect")
    out.append("    skip?   = P(C0) ranks 'answering without retrieval is correct'")
    out.append("    C2?     = P(C2) ranks 'one search fails but iterative search succeeds'")
    out.append("    C1vsC2  = P(C2)/(P(C1)+P(C2)) among questions labelled C1 or C2 (the pure stage-B decision)")
    srcs = sorted(set(src.tolist()))
    head = "    " + f"{'run':<44}" + "".join(f"{x:>10}" for x in ["skip?", "C2?", "C1vsC2"])
    head += "".join(f"  {s[:9]:>9}:{'skip':>5}{'C2':>5}{'1v2':>5}" for s in srcs)
    out.append(head)
    js["auc"] = {}
    for run in runs:
        p = run["probs"]
        p12 = p[:, 2] / np.clip(p[:, 1] + p[:, 2], 1e-12, None)
        res = {"all": [auc(p[:, 0], y0), auc(p[:, 2], need2), auc(p12[c12], y[c12] == 2) if c12.any() else float("nan")]}
        for s_ in srcs:
            m = (src == s_) & ~comp
            res[s_] = [auc(p[m, 0], y0[m]), auc(p[m, 2], need2[m]), auc(p12[m & c12], y[m & c12] == 2)]
        js["auc"][run["name"]] = res
        line = f"    {run['name']:<44}" + "".join(f"{v:10.3f}" for v in res["all"])
        line += "".join(f"  {'':>9} {res[s_][0]:5.2f}{res[s_][1]:5.2f}{res[s_][2]:5.2f}" for s_ in srcs)
        out.append(line)

    out.append("\n3c) GAIN OVER RANDOM MIXING INSIDE EACH DATASET (removes any 'recognise the dataset' shortcut; no comparison q.)")
    for s_ in srcs:
        m = (src == s_) & ~comp
        for run in runs:
            for kind in ("argmax", "λ dev"):
                ch, esc = pol[f"{run['name']} | {kind}"]
                vm = bootstrap_vs_mix(correct[m], cost[m], calls[m], ch[m], None, reps=max(200, reps // 4))
                js.setdefault("vs_mix_within", {}).setdefault(s_, {})[f"{run['name']} | {kind}"] = vm
                out.append(f"    {s_:<14} {run['name'] + ' | ' + kind:<48} {100 * vm['vs_mix']:+5.1f} pts "
                           f"[{100 * vm['ci'][0]:+.1f},{100 * vm['ci'][1]:+.1f}]")

    out.append("\n4) ACCURACY vs COST as the caution knob λ moves (all questions). Descriptive curve only - the 'λ dev'")
    out.append("   rows above are the honest operating points (λ picked on inner-dev, never on the test fold).")
    hull = fixed_frontier(correct, cost, calls)
    js["sweep"] = {}
    for run in runs:
        out.append(f"   {run['name']}")
        out.append(f"     {'λ':>5}   {'plain: acc':>10} {'$/1k':>6} {'vs mix':>6}   {'IDK→up: acc':>11} {'$/1k':>6} {'vs mix':>6}")
        rows = []
        for lam in LAM_GRID:
            ch = choose_cautious(run["probs"], lam)
            e1 = evaluate_routing(correct, cost, calls, ch)
            e2 = evaluate_routing(correct, cost, calls, ch, idk)
            m1, m2 = frontier_acc(hull, e1["cost_per_1k"]), frontier_acc(hull, e2["cost_per_1k"])
            rows.append({"lam": lam, "plain": {**e1, "vs_mix": e1["accuracy"] - m1}, "idk_up": {**e2, "vs_mix": e2["accuracy"] - m2}})
            if round(lam * 20) % 2 == 0:
                out.append(f"     {lam:5.2f}   {100 * e1['accuracy']:9.1f}% {e1['cost_per_1k']:6.3f} {100 * (e1['accuracy'] - m1):+6.1f}"
                           f"   {100 * e2['accuracy']:10.1f}% {e2['cost_per_1k']:6.3f} {100 * (e2['accuracy'] - m2):+6.1f}")
        js["sweep"][run["name"]] = rows
        out.append(f"     λ picked on inner-dev per fold: plain {sorted(set(np.round(run['lam'], 2).tolist()))}, "
                   f"IDK→up {sorted(set(np.round(run['lam_esc'], 2).tolist()))}")

    # ---- 5. seeds
    groups = {}
    for run in runs:
        groups.setdefault(_group_key(run), []).append(run["name"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        out.append("\n5) SEED STABILITY (mean ± sd over seeds, all questions)")
        for key, names in multi.items():
            f1 = [js["runs"][nm]["label"]["macro_f1"] for nm in names]
            out.append(f"   {key} ({len(names)} seeds): label macro-F1 {100 * np.mean(f1):.1f} ± {100 * np.std(f1):.1f}%")
            for kind in ("argmax", "λ dev", "λ dev + IDK→up"):
                evs = [js["routing_all"][f"{nm} | {kind}"] for nm in names]
                a, c, v = ([e["accuracy"] for e in evs], [e["cost_per_1k"] for e in evs], [e["vs_mix"] for e in evs])
                out.append(f"     {kind:<15}: accuracy {100 * np.mean(a):.1f} ± {100 * np.std(a):.1f}%  "
                           f"$/1k {np.mean(c):.3f} ± {np.std(c):.3f}  vs mix {100 * np.mean(v):+.1f} ± {100 * np.std(v):.1f}")
    return "\n".join(out), js


def plot_curve(recs, runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    correct, cost, calls = outcome_matrix(recs)
    idk = idk_matrix(recs)
    n = len(recs)
    fig, ax = plt.subplots(figsize=(8, 5))
    hull = fixed_frontier(correct, cost, calls)
    ax.plot([h[0] for h in hull], [100 * h[1] for h in hull], "--", color="grey", lw=1, label="random mix of fixed strategies")
    fixed = [("always C0", fixed_choice(n, 0), None, "s"), ("always C1", fixed_choice(n, 1), None, "s"),
             ("always C2", fixed_choice(n, 2), None, "s"), ("cascade C1→C2", fixed_choice(n, 1), idk, "^"),
             ("oracle", oracle_choice(correct), None, "*")]
    for name, ch, ik, mk in fixed:
        ev = evaluate_routing(correct, cost, calls, ch, ik)
        ax.scatter(ev["cost_per_1k"], 100 * ev["accuracy"], marker=mk, s=80, color="black", zorder=3)
        ax.annotate(name, (ev["cost_per_1k"], 100 * ev["accuracy"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    for run in runs:
        pts = [evaluate_routing(correct, cost, calls, choose_cautious(run["probs"], lam), idk) for lam in LAM_GRID]
        line, = ax.plot([p["cost_per_1k"] for p in pts], [100 * p["accuracy"] for p in pts], "-o", ms=3,
                        label=f"{run['name']} (IDK→up)")
        ev = evaluate_routing(correct, cost, calls, per_item_cautious(run["probs"], run["lam_esc"]), idk)
        ax.scatter(ev["cost_per_1k"], 100 * ev["accuracy"], marker="D", s=50, color=line.get_color(), edgecolor="black", zorder=4)
    ax.set_xlabel("generator cost per 1,000 questions (USD)")
    ax.set_ylabel("answer accuracy (%)")
    ax.set_title("Router accuracy vs cost (out-of-fold; diamonds = λ picked on inner-dev)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
