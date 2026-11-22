"""Phase 4e: evaluate the FINAL 3-stage complexity router against every baseline (the RQ1 thesis table).

  python scripts/eval_final_router.py
Uses only stored results (no API calls): labels.jsonl, the Phase 4a text-head run, the Phase 4c/4d features and
reader runs, and (optional) the Phase 4c repeat run for a fresh-outcome check.
Writes runs_final/final_report.txt / .json and final_curve.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.escalate import ANSWER_FEATURES, bootstrap_items, escalate_outcome    # noqa: E402
from arroute.router.metrics import wilson                                                # noqa: E402
from arroute.router.outcome import (fixed_frontier, frontier_acc, load_outcomes, make_folds,  # noqa: E402
                                    oracle_choice, outcome_matrix, save_json, strategy_cost)
from arroute.router.policy import RULES, honest, outcome                                 # noqa: E402
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE                           # noqa: E402
from arroute.router.route_report import auc, load_run, per_item_cautious                 # noqa: E402
from eval_escalation import cv_lr, load_feats, load_reader                               # noqa: E402

VARIANTS = {"A: text head only": (True, False, False),
            "B: pre-answer detector only": (False, True, False),
            "A+B": (True, True, False),
            "A+B+C (full router)": (True, True, True)}


def build_scores(recs, fold, a):
    """sa, sb, sc out-of-fold, plus the names of the inputs used (for the report)."""
    text = load_run(a.text_run, recs)
    sa = text["probs"][:, 0]
    rescue = np.array([(not r["results"]["C1"]["correct"]) and r["results"]["C2"]["correct"] for r in recs]).astype(int)
    pre_reader, _ = load_reader(Path(a.readers) / a.pre_reader, recs)
    ans_reader, _ = load_reader(Path(a.readers) / a.ans_reader, recs)
    XR = load_feats(a.ret_feats, recs)[0]
    XRP = load_feats(a.rerank_feats, recs, RERANK_PRE)[0]
    XRA = load_feats(a.rerank_feats, recs, RERANK_POST)[0]
    XA = load_feats(a.esc_feats, recs, ANSWER_FEATURES)[0]
    PRE = np.hstack([pre_reader[:, None], XR, XRP])
    ALL = np.hstack([PRE, XA, XRA, ans_reader[:, None]])
    sb = cv_lr(PRE, rescue, fold, kind="hgb")
    sc = cv_lr(ALL, rescue, fold, kind="hgb")
    return sa, sb, sc, text, XA, {"PRE": PRE, "ALL": ALL, "rescue": rescue}


def fresh_outcomes(recs, path):
    """Replace C1/C2 outcomes by the repeat run's (where available). Returns (mask, correct, cost, calls)."""
    by = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["id"]] = r
    correct, cost, calls = outcome_matrix(recs)
    mask = np.array([r["id"] in by for r in recs])
    for i, r in enumerate(recs):
        if mask[i]:
            for j, s in ((1, "C1"), (2, "C2")):
                x = by[r["id"]][s]
                correct[i, j], cost[i, j], calls[i, j] = bool(x["correct"]), strategy_cost(x), x.get("calls", 1)
    return mask, correct, cost, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--text_run", default="runs_router/bert-base-arabertv02_seed0")
    ap.add_argument("--readers", default="runs_suff")
    ap.add_argument("--pre_reader", default="bert-base-arabertv02_c1_ok_v2_seed0")
    ap.add_argument("--ans_reader", default="bert-base-arabertv02_c1_ok_ans_v2_seed0")
    ap.add_argument("--esc_feats", default="data/router_feats/escalation_feats.jsonl")
    ap.add_argument("--ret_feats", default="data/router_feats/retrieval_feats.jsonl")
    ap.add_argument("--rerank_feats", default="data/router_feats/rerank_feats.jsonl")
    ap.add_argument("--repeat", default="runs_suff/repeat/repeat_temp0.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default="runs_final")
    a = ap.parse_args()

    recs = load_outcomes(a.labels)
    n = len(recs)
    correct, cost, calls = outcome_matrix(recs)
    fold = make_folds(recs, k=a.folds, seed=a.fold_seed)
    sa, sb, sc, text, XA, _ = build_scores(recs, fold, a)
    src = np.array([r["source"] for r in recs])
    rescue = ~correct[:, 1] & correct[:, 2]
    hull = fixed_frontier(correct, cost, calls)
    refs = {"always C2": (correct[:, 2].astype(float), cost[:, 2]),
            "always C1": (correct[:, 1].astype(float), cost[:, 1])}

    rows, js = [], {"n": n, "policies": {}, "inputs": vars(a)}

    def add(name, hit, c, k, shares=None, extra=None):
        b = bootstrap_items(hit, c, correct, cost, calls, refs, reps=a.reps)
        lo, hi = wilson(int(hit.sum()), n)
        js["policies"][name] = {"accuracy": float(hit.mean()), "ci": [lo, hi], "cost_per_1k": float(c.mean() * 1000),
                                "calls": float(k.mean()), "shares": shares, **b, **(extra or {})}
        rows.append((name, hit, c, k, b, lo, hi, shares))

    for j, s in enumerate(("C0", "C1", "C2")):
        add(f"always {s}", correct[:, j].astype(float), cost[:, j], calls[:, j])
    h, c = escalate_outcome(correct, cost, XA[:, 0] > 0)
    add("cascade: C1, then C2 if 'لا أعرف'", h, c, calls[:, 1] + (XA[:, 0] > 0) * calls[:, 2])
    ch = per_item_cautious(text["probs"], text["lam"])        # Phase 4a: Adaptive-RAG-style 3-class text router
    r_ = np.arange(n)
    add("Adaptive-RAG-style text router (4a)", correct[r_, ch].astype(float), cost[r_, ch], calls[r_, ch])
    och = oracle_choice(correct)
    add("oracle (upper bound)", correct[r_, och].astype(float), cost[r_, och], calls[r_, och])

    decisions = {}
    for vname, use in VARIANTS.items():
        for rule in RULES:
            skip, c2, esc, ts = honest(sa, sb, sc, fold, correct, cost, calls, rule, a.tol, use)
            hit, c, k = outcome(correct, cost, calls, skip, c2, esc)
            sh = {"C0": float(skip.mean()), "C2 direct": float(c2.mean()), "C1": float((~skip & ~c2).mean()),
                  "escalated": float(esc.mean())}
            add(f"{vname} | {rule}", hit, c, k, sh, {"thresholds_per_fold": ts})
            decisions[(vname, rule)] = (skip, c2, esc)

    L = [f"FINAL COMPLEXITY ROUTER - RQ1 table ({n:,} questions, out-of-fold, thresholds chosen on the other folds)",
         "stage A = AraBERT text head (skip retrieval) | stage B = pre-answer detector (C1 or C2) | stage C = escalate after C1",
         "budget = best accuracy at <= standard-RAG cost | match_c1 = cheapest within "
         f"{100 * a.tol:.0f} pt of standard RAG | balanced = best gain over random mixing | "
         f"quality = cheapest within {100 * a.tol:.0f} pt of always-C2",
         "", f"{'policy':<44} {'acc':>6} {'95% CI':>12} {'$/1k':>6} {'calls':>5} {'vs mix [95% CI]':>18} "
         f"{'vs always-C2':>24} {'vs always-C1':>24}   C0/C1/C2/esc %"]
    for name, hit, c, k, b, lo, hi, sh in rows:
        v2, v1 = b["always C2"], b["always C1"]
        shs = (f"{100 * sh['C0']:.0f}/{100 * sh['C1']:.0f}/{100 * sh['C2 direct']:.0f}/{100 * sh['escalated']:.0f}"
               if sh else "")
        L.append(f"{name:<44} {100 * hit.mean():5.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}] {c.mean() * 1000:6.3f} {k.mean():5.2f} "
                 f"{100 * b['vs_mix']:+5.1f} [{100 * b['vs_mix_ci'][0]:+.1f},{100 * b['vs_mix_ci'][1]:+.1f}] "
                 f"{100 * v2['acc_diff']:+5.1f}pt x{v2['cost_ratio']:.2f} [{v2['cost_ratio_ci'][0]:.2f}-{v2['cost_ratio_ci'][1]:.2f}] "
                 f"{100 * v1['acc_diff']:+5.1f}pt x{v1['cost_ratio']:.2f} [{v1['cost_ratio_ci'][0]:.2f}-{v1['cost_ratio_ci'][1]:.2f}]   {shs}")
        if name.endswith("quality") or name.startswith("oracle"):
            L.append("")

    L.append("\nSTAGE SKILL (AUC; 0.5 = coin flip):  "
             f"A: P(C0) vs 'C0 correct' {auc(sa, correct[:, 0]):.3f} | B: pre-answer score vs 'C2 rescues' {auc(sb, rescue):.3f}"
             f" | C: after-C1 score vs 'C2 rescues' {auc(sc, rescue):.3f}")

    L.append("\nFULL ROUTER PER DATASET (all questions of each dataset; same decisions as above)")
    L.append(f"  {'dataset':<16} {'rule':<9} {'acc':>6} {'$/1k':>6} {'always-C2 acc':>14} {'$/1k':>6}  {'vs mix':>6}")
    for s_ in sorted(set(src.tolist())):
        m = src == s_
        hl = fixed_frontier(correct[m], cost[m], calls[m])
        for rule in RULES:
            hit, c, _ = outcome(correct[m], cost[m], calls[m], *[d[m] for d in decisions[("A+B+C (full router)", rule)]])
            L.append(f"  {s_:<16} {rule:<9} {100 * hit.mean():5.1f}% {c.mean() * 1000:6.3f} {100 * correct[m, 2].mean():13.1f}% "
                     f"{cost[m, 2].mean() * 1000:6.3f}  {100 * (hit.mean() - frontier_acc(hl, c.mean() * 1000)):+5.1f}")

    if Path(a.repeat).exists():
        mask, fc, fcost, fcalls = fresh_outcomes(recs, a.repeat)
        L.append(f"\nFRESH-RUN CHECK on the {mask.sum()} re-run questions: the SAME decisions, scored with C1/C2 answers "
                 "generated again (Phase 4c repeat). If the router only memorised noise, it would fall apart here.")
        L.append(f"  {'policy':<36} {'original acc':>12} {'fresh acc':>10} {'fresh $/1k':>11}")
        for rule in RULES:
            skip, c2, esc = (d[mask] for d in decisions[("A+B+C (full router)", rule)])
            for lab, (cc, co, ca) in (("original", (correct[mask], cost[mask], calls[mask])),
                                      ("fresh", (fc[mask], fcost[mask], fcalls[mask]))):
                h, c, _ = outcome(cc, co, ca, skip, c2, esc)
                if lab == "original":
                    ho = h.mean()
                else:
                    L.append(f"  {'full router | ' + rule:<36} {100 * ho:11.1f}% {100 * h.mean():9.1f}% {c.mean() * 1000:11.3f}")
                    js.setdefault("fresh", {})[rule] = {"original": float(ho), "fresh": float(h.mean()), "cost": float(c.mean() * 1000)}
        for j, s in ((1, "C1"), (2, "C2")):
            L.append(f"  {'always ' + s:<36} {100 * correct[mask, j].mean():11.1f}% {100 * fc[mask, j].mean():9.1f}% "
                     f"{fcost[mask, j].mean() * 1000:11.3f}")

    text_out = "\n".join(L)
    print(text_out)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_report.txt").write_text(text_out + "\n", encoding="utf-8")
    save_json(js, out / "final_report.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot([h[0] for h in hull], [100 * h[1] for h in hull], "--", color="grey", lw=1, label="random mix of fixed strategies")
        marks = {"budget": "o", "match_c1": "v", "balanced": "D", "quality": "^"}
        for name, hit, c, k, b, lo, hi, sh in rows:
            x, y = c.mean() * 1000, 100 * hit.mean()
            if " | " in name:
                v, rule = name.split(" | ")
                ax.scatter(x, y, marker=marks[rule], s=45, label=name if rule == "balanced" else None,
                           color={"A: text head only": "C0", "B: pre-answer detector only": "C1", "A+B": "C2",
                                  "A+B+C (full router)": "C3"}[v], zorder=4)
            else:
                ax.scatter(x, y, marker="s", color="black", zorder=3)
                ax.annotate(name.split(" (")[0], (x, y), textcoords="offset points", xytext=(5, -10), fontsize=7)
        ax.set_xlabel("generator cost per 1,000 questions (USD)")
        ax.set_ylabel("answer accuracy (%)")
        ax.set_title("Final router: o budget, v match-C1, D balanced, ^ quality (out-of-fold)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "final_curve.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(f"\nreport: {out / 'final_report.txt'}")


if __name__ == "__main__":
    main()
