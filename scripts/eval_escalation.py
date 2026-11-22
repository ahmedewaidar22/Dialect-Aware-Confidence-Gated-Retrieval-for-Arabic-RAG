"""Phase 4c report: run C1, then escalate to C2 only where a detector says C1 is unreliable.

  python scripts/eval_escalation.py
Uses: labels.jsonl (stored outcomes - no API calls), data/router_feats/escalation_feats.jsonl,
data/router_feats/retrieval_feats.jsonl (optional), runs_suff/*/oof.npz (passage-reader runs, optional).
Writes <root>/escalation_report.txt / .json and escalation_curve.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.escalate import (ANSWER_FEATURES, C0_FEATURES, bootstrap_items,      # noqa: E402
                                     escalate_outcome, threshold_for, top_share)
from arroute.router.metrics import wilson                                                # noqa: E402
from arroute.router.outcome import (fixed_frontier, frontier_acc, load_outcomes, make_folds,  # noqa: E402
                                    outcome_matrix, save_json)
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE                           # noqa: E402
from arroute.router.route_report import auc                                              # noqa: E402

SHARES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_feats(path, recs, keys=None):
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r["feats"]
    miss = [r["id"] for r in recs if r["id"] not in by_id]
    if miss:
        raise SystemExit(f"{path}: {len(miss)} questions missing - re-run the script that writes it")
    keys = keys or list(by_id[recs[0]["id"]])
    return np.array([[by_id[r["id"]][k] for k in keys] for r in recs], float), keys


def load_reader(run_dir, recs):
    d = np.load(Path(run_dir) / "oof.npz", allow_pickle=False)
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    pos = {str(i): k for k, i in enumerate(d["ids"])}
    if any(r["id"] not in pos for r in recs):
        raise SystemExit(f"{run_dir} does not cover all questions")
    p = np.array([d["p"][pos[r["id"]]] for r in recs])
    return (1 - p) if cfg.get("target", "c1_ok") == "c1_ok" else p, cfg      # score: high = escalate


def cv_lr(X, y, fold, seed=0, kind="lr"):
    """Out-of-fold P(y=1) from a model trained on the other folds: 'lr' = standardised logistic
    regression, 'hgb' = small boosted trees (can use interactions between signals)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = np.zeros(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               l2_regularization=1.0, random_state=seed)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
        out[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def honest_policy(score, fold, correct, cost, calls, rule, tol, with_c0, pre):
    """Threshold picked on the OTHER folds (their out-of-fold scores), applied to the held-out fold."""
    esc = np.zeros(len(score), bool)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        t = threshold_for(score[tr], correct[tr], cost[tr], calls[tr], rule, tol, with_c0, pre)
        esc[te] = score[te] >= t
    return esc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--esc_feats", default="data/router_feats/escalation_feats.jsonl")
    ap.add_argument("--ret_feats", default="data/router_feats/retrieval_feats.jsonl")
    ap.add_argument("--rerank_feats", default="data/router_feats/rerank_feats.jsonl")
    ap.add_argument("--readers", default="runs_suff", help="folder with passage-reader runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default: <readers>/escalation_report")
    a = ap.parse_args()

    recs = load_outcomes(a.labels)
    n = len(recs)
    correct, cost, calls = outcome_matrix(recs)
    fold = make_folds(recs, k=a.folds, seed=a.fold_seed)
    c1_wrong = ~correct[:, 1]
    rescue = (~correct[:, 1] & correct[:, 2]).astype(int)
    XA, _ = load_feats(a.esc_feats, recs, ANSWER_FEATURES)
    XC0, _ = load_feats(a.esc_feats, recs, C0_FEATURES)
    has_ret = Path(a.ret_feats).exists()
    XR = load_feats(a.ret_feats, recs)[0] if has_ret else None
    has_rr = Path(a.rerank_feats).exists()
    XRP = load_feats(a.rerank_feats, recs, RERANK_PRE)[0] if has_rr else None
    XRA = load_feats(a.rerank_feats, recs, RERANK_POST)[0] if has_rr else None
    readers = {}
    for d in sorted(Path(a.readers).glob("*/oof.npz")):
        readers[d.parent.name] = load_reader(d.parent, recs)

    def pick(pred):
        """best reader run of a kind: prefer the fixed (v2) AraBERT runs, then any AraBERT, then TF-IDF."""
        names = [k for k in readers if pred(k, readers[k][1])]
        for pref in ("_v2", "arabert", ""):
            hit = [k for k in names if pref in k]
            if hit:
                return hit[0]
        return None
    pre_reader = pick(lambda k, c: c.get("target", "c1_ok") == "c1_ok" and not c.get("with_answer"))
    ans_reader = pick(lambda k, c: c.get("with_answer"))

    # detectors: name -> (score, needs C0 call, usable BEFORE C1 answers)
    det = {"C1 said 'لا أعرف'": (XA[:, 0] + 1e-3 * (1 - XA[:, 4]), False, False),
           "answer checks (LR)": (cv_lr(XA, rescue, fold), False, False),
           "answer checks + C0 (LR)": (cv_lr(np.hstack([XA, XC0]), rescue, fold), True, False)}
    if has_ret:
        det["retrieval signals (LR)"] = (cv_lr(XR, rescue, fold), False, True)
    if has_rr:
        det["reranker (LR)"] = (cv_lr(XRP, rescue, fold), False, True)
        det["reranker + answer support (LR)"] = (cv_lr(np.hstack([XRP, XRA]), rescue, fold), False, False)
    for name, (sc, cfg) in readers.items():
        det[f"reader {name}"] = (sc, False, not cfg.get("with_answer"))
    pre_parts = [x for x in ([readers[pre_reader][0][:, None]] if pre_reader else []) + [XR, XRP] if x is not None]
    post_parts = [XA] + ([XRA] if has_rr else []) + ([readers[ans_reader][0][:, None]] if ans_reader else [])
    if pre_parts:
        PRE = np.hstack(pre_parts)
        det["BEFORE C1: all signals (LR)"] = (cv_lr(PRE, rescue, fold), False, True)
        det["BEFORE C1: all signals (trees)"] = (cv_lr(PRE, rescue, fold, kind="hgb"), False, True)
        ALL = np.hstack([PRE] + post_parts)
    else:
        ALL = np.hstack(post_parts)
    det["ALL signals (LR)"] = (cv_lr(ALL, rescue, fold), False, False)
    det["ALL signals (trees)"] = (cv_lr(ALL, rescue, fold, kind="hgb"), False, False)
    det["ALL signals + C0 (trees)"] = (cv_lr(np.hstack([ALL, XC0]), rescue, fold, kind="hgb"), True, False)
    det["CEILING: perfect 'C1 fails'"] = (c1_wrong.astype(float), False, True)
    det["CEILING: perfect 'C2 rescues'"] = (rescue.astype(float), False, True)

    hull = fixed_frontier(correct, cost, calls)
    refs = {"always C2": (correct[:, 2].astype(float), cost[:, 2]),
            "always C1": (correct[:, 1].astype(float), cost[:, 1]),
            "IDK cascade": escalate_outcome(correct, cost, XA[:, 0] > 0)}
    L, js = [], {"n": n, "detectors": {}}
    L.append(f"ESCALATION: answer with C1, move to C2 only when a detector says C1 is unreliable  ({n:,} questions)")
    L.append("cost = generator $ per 1,000 questions; an escalated question pays C1 + C2 ('after C1'),")
    L.append("or only C1 or C2 when the detector runs before answering ('before C1'). Detectors marked +C0 also pay C0.")
    L.append("'vs mix' = accuracy minus the best random mix of always-C0/C1/C2 at the same cost (must be > 0 to be useful).")
    for nm, (h, c) in refs.items():
        L.append(f"  reference {nm:<12}: accuracy {100 * h.mean():.1f}%  ${c.mean() * 1000:.3f}/1k  "
                 f"vs mix {100 * (h.mean() - frontier_acc(hull, c.mean() * 1000)):+.1f}")
    L.append(f"  rescues available: {rescue.sum()} questions ({100 * rescue.mean():.1f}%) where C1 fails and C2 succeeds")

    L.append("\n1) HOW WELL DOES EACH DETECTOR RANK THE QUESTIONS?  AUC (0.5 = coin flip, 1.0 = perfect)")
    L.append(f"   {'detector':<44} {'C1 fails':>9} {'C2 rescues':>11}   {'arabicaqa':>10} {'hotpot':>8}  (rescue AUC per dataset)")
    src = np.array([r["source"] for r in recs])
    for nm, (s, _, _) in det.items():
        a1, a2 = auc(s, c1_wrong), auc(s, rescue)
        per = [auc(s[src == x], rescue[src == x]) for x in ("arabicaqa", "hotpot_ctx_ar")]
        js["detectors"].setdefault(nm, {})["auc"] = {"c1_fails": a1, "rescue": a2, "arabicaqa": per[0], "hotpot": per[1]}
        L.append(f"   {nm:<44} {a1:9.3f} {a2:11.3f}   {per[0]:10.3f} {per[1]:8.3f}")
    L.append("   (LR/trees detectors are trained to predict 'C2 rescues'; combined detectors stack the readers'")
    L.append("    out-of-fold scores - a standard, very mildly optimistic practice.)")
    L.append(f"   readers used in the combined detectors: before C1 = {pre_reader}, after C1 = {ans_reader}")

    L.append("\n2) OPERATING POINTS chosen on the OTHER folds only (honest):  'match C2' = cheapest share within "
             f"{100 * a.tol:.0f} pt of always-C2;  'best gain' = largest gain over random mixing")
    L.append(f"   {'detector':<40} {'when':<9} {'rule':<10} {'esc%':>5} {'accuracy':>8} {'95% CI':>12} {'$/1k':>6} "
             f"{'vs mix [95% CI]':>19}   vs always-C2: acc, cost")
    for nm, (s, with_c0, pre_ok) in det.items():
        for pre in ([False, True] if pre_ok else [False]):
            for rule in ("match_c2", "best_gain"):
                esc = honest_policy(s, fold, correct, cost, calls, rule, a.tol, with_c0, pre)
                hit, c = escalate_outcome(correct, cost, esc, with_c0, pre)
                b = bootstrap_items(hit, c, correct, cost, calls, refs, reps=a.reps)
                lo, hi = wilson(int(hit.sum()), n)
                v2 = b["always C2"]
                key = f"{'before' if pre else 'after'} C1 | {rule}"
                js["detectors"][nm][key] = {"escalated": float(esc.mean()), "accuracy": float(hit.mean()),
                                            "cost_per_1k": float(c.mean() * 1000), **b}
                L.append(f"   {nm[:40]:<40} {'before' if pre else 'after':<9} {rule:<10} {100 * esc.mean():5.1f} "
                         f"{100 * hit.mean():7.1f}% [{100 * lo:4.1f}-{100 * hi:4.1f}] {c.mean() * 1000:6.3f} "
                         f"{100 * b['vs_mix']:+5.1f} [{100 * b['vs_mix_ci'][0]:+.1f},{100 * b['vs_mix_ci'][1]:+.1f}]   "
                         f"{100 * v2['acc_diff']:+.1f} pts [{100 * v2['acc_diff_ci'][0]:+.1f},{100 * v2['acc_diff_ci'][1]:+.1f}], "
                         f"x{v2['cost_ratio']:.2f}" + (" +C0" if with_c0 else ""))

    L.append("\n3) THE WHOLE CURVE (escalating a fixed share of questions, 'after C1')  - descriptive")
    L.append(f"   {'detector':<40}" + "".join(f"{int(100 * x):>5}%" for x in SHARES) + "   <- escalated share; cells = accuracy / vs mix")
    js["sweep"] = {}
    for nm, (s, with_c0, _) in det.items():
        cells, rows = [], []
        for sh in SHARES:
            hit, c = escalate_outcome(correct, cost, top_share(s, sh), with_c0)
            g = hit.mean() - frontier_acc(hull, c.mean() * 1000)
            rows.append({"share": sh, "accuracy": float(hit.mean()), "cost_per_1k": float(c.mean() * 1000), "vs_mix": float(g)})
            cells.append(f"{100 * hit.mean():4.1f}/{100 * g:+.1f}")
        js["sweep"][nm] = rows
        L.append(f"   {nm[:40]:<40} " + " ".join(f"{x:>10}" for x in cells))

    text = "\n".join(L)
    print(text)
    stem = Path(a.out or Path(a.readers) / "escalation_report")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    save_json(js, stem.with_suffix(".json"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([h[0] for h in hull], [100 * h[1] for h in hull], "--", color="grey", lw=1, label="random mix of fixed strategies")
        for nm, (h, c) in refs.items():
            ax.scatter(c.mean() * 1000, 100 * h.mean(), marker="s", color="black", zorder=3)
            ax.annotate(nm, (c.mean() * 1000, 100 * h.mean()), textcoords="offset points", xytext=(5, -10), fontsize=8)
        for nm, rows in js["sweep"].items():
            style = ":" if nm.startswith("CEILING") else "-o"
            ax.plot([r["cost_per_1k"] for r in rows], [100 * r["accuracy"] for r in rows], style, ms=3, label=nm)
        ax.set_xlabel("generator cost per 1,000 questions (USD)")
        ax.set_ylabel("answer accuracy (%)")
        ax.set_title("Escalate from C1 to C2: accuracy vs cost")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")
        fig.tight_layout()
        fig.savefig(stem.with_name("escalation_curve.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(f"\nreport: {stem.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
