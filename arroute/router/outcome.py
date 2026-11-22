"""Phase 4a: complexity router trained on OUTCOME labels (torch-free core).

An outcome label is the cheapest strategy the judge accepted (C0 < C1 < C2), or 'none' if all failed.
Because labelling ran every strategy on every question ('full' mode), we know for each question
whether C0, C1 and C2 would have been correct and what each cost. So ANY routing policy can be
scored offline, for free: pick a strategy per question, look up whether it was correct and its cost.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATS = ["C0", "C1", "C2"]
PRICE_IN, PRICE_OUT = 0.40, 1.60          # gpt-4.1-mini, USD per 1M tokens (generator only)


# ---------------------------------------------------------------- data
def load_outcomes(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            missing = [s for s in STRATS if s not in r.get("results", {})]
            if missing:
                raise ValueError(f"{r.get('id')}: strategies {missing} missing - labels must come from --mode full")
            recs.append(r)
    if not recs:
        raise ValueError(f"no records in {path}")
    return recs


def is_comparison(rec):
    return rec.get("qtype") == "comparison"


def trainable(rec, none_as=None):
    """Router training excludes comparison questions (guessable) and, by default, 'none' questions."""
    if is_comparison(rec):
        return False
    if rec["label"] == "none":
        return none_as is not None
    return True


def label_id(rec, none_as=None):
    lab = rec["label"]
    if lab == "none":
        if none_as is None:
            raise ValueError("'none' label with none_as=None")
        lab = none_as
    return STRATS.index(lab)


def strategy_cost(res):
    return (res.get("input_tokens", 0) * PRICE_IN + res.get("output_tokens", 0) * PRICE_OUT) / 1e6


def outcome_matrix(recs):
    n = len(recs)
    correct, cost, calls = np.zeros((n, 3), bool), np.zeros((n, 3)), np.zeros((n, 3))
    for i, r in enumerate(recs):
        for j, s in enumerate(STRATS):
            res = r["results"][s]
            correct[i, j] = bool(res.get("correct"))
            cost[i, j] = strategy_cost(res)
            calls[i, j] = res.get("calls", 1)
    return correct, cost, calls


# ---------------------------------------------------------------- folds
def _norm_q(q):
    return re.sub(r"\s+", " ", q).strip()


def make_folds(recs, k=5, seed=42):
    """Stratified by (source, label, comparison). Identical question texts always share a fold."""
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[_norm_q(r["question"])].append(i)
    strata = defaultdict(list)
    for g, idx in groups.items():
        r = recs[idx[0]]
        strata[(r["source"], r["label"], is_comparison(r))].append(g)
    rng = np.random.RandomState(seed)
    fold = np.full(len(recs), -1, int)
    offset = 0
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        for n, g in enumerate(gs):
            for i in groups[g]:
                fold[i] = (n + offset) % k
        offset += len(gs)          # rotate start so small strata don't all land in fold 0
    assert (fold >= 0).all()
    return fold


def inner_dev_split(idx, recs, frac=0.15, seed=0):
    """Split training-fold indices into (fit, dev), keeping duplicate questions together."""
    idx = np.asarray(idx)
    texts = sorted({_norm_q(recs[i]["question"]) for i in idx})
    rng = np.random.RandomState(seed)
    rng.shuffle(texts)
    dev_t = set(texts[:max(1, int(round(frac * len(texts))))])
    is_dev = np.array([_norm_q(recs[i]["question"]) in dev_t for i in idx])
    return idx[~is_dev], idx[is_dev]


# ---------------------------------------------------------------- policies
def choose_argmax(probs):
    return np.asarray(probs).argmax(axis=1)


def choose_cautious(probs, lam):
    """Cheapest strategy s with P(label <= s) >= lam. lam -> 0: always C0; lam = 1: always C2.
    Labels are ordered (C0 < C1 < C2), so P(label <= s) is 'chance that s is already enough'."""
    cum = np.cumsum(np.asarray(probs), axis=1)
    cum[:, -1] = 1.0
    return (cum >= lam - 1e-12).argmax(axis=1)


def oracle_choice(correct):
    """Cheapest correct strategy; C0 when none is correct (cheapest way to fail)."""
    c = np.asarray(correct)
    return np.where(c.any(axis=1), c.argmax(axis=1), 0)


def fixed_choice(n, s):
    return np.full(n, STRATS.index(s) if isinstance(s, str) else s, int)


# ---------------------------------------------------------------- scoring
IDK = "لا أعرف"


def idk_matrix(recs):
    """True where the strategy answered 'I don't know' (C1/C2 prompts are told to do so without evidence)."""
    return np.array([[IDK in (r["results"][s].get("answer") or "") for s in STRATS] for r in recs], bool)


def policy_outcome(correct, cost, calls, choice, idk=None):
    """Per-question (hit, cost, calls, final strategy). With `idk`, a strategy that answers 'I don't know'
    escalates to the next one (C0 -> C1 -> C2), paying for every step taken."""
    n = len(choice)
    choice = np.asarray(choice)
    if idk is None:
        r = np.arange(n)
        return correct[r, choice].astype(float), cost[r, choice], calls[r, choice], choice
    hit, c, k, fin = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, int)
    for i in range(n):
        j = int(choice[i])
        while True:
            c[i] += cost[i, j]
            k[i] += calls[i, j]
            if not idk[i, j] or j == 2:
                break
            j += 1
        hit[i], fin[i] = correct[i, j], j
    return hit, c, k, fin


def evaluate_routing(correct, cost, calls, choice, idk=None):
    hit, c, k, fin = policy_outcome(correct, cost, calls, choice, idk)
    n = len(choice)
    out = {"n": int(n), "accuracy": float(hit.mean()), "cost_per_1k": float(c.sum() / n * 1000),
           "calls": float(k.mean()), "share": {s: float((np.asarray(choice) == j).mean()) for j, s in enumerate(STRATS)}}
    if idk is not None:
        out["escalated"] = float((fin != np.asarray(choice)).mean())
    return out


def fixed_frontier(correct, cost, calls):
    """Upper convex hull of the fixed strategies (always-C0/C1/C2) in (cost, accuracy). Any point on it is
    reachable WITHOUT a router, by sending a random share of questions to each strategy."""
    n = len(correct)
    pts = sorted((evaluate_routing(correct, cost, calls, fixed_choice(n, j))["cost_per_1k"],
                  evaluate_routing(correct, cost, calls, fixed_choice(n, j))["accuracy"]) for j in range(3))
    hull = []
    for p in pts:
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def frontier_acc(hull, c):
    """Accuracy of the best random mixture of fixed strategies that costs `c`."""
    if c <= hull[0][0]:
        return hull[0][1]
    for (c0, a0), (c1, a1) in zip(hull[:-1], hull[1:]):
        if c <= c1:
            return a0 + (a1 - a0) * (c - c0) / (c1 - c0)
    return hull[-1][1]


def bootstrap_vs(correct, cost, choice_a, choice_b, reps=2000, seed=0, calls=None, idk_a=None, idk_b=None):
    """Paired bootstrap over questions: accuracy(a) - accuracy(b) and cost(a) / cost(b), 95% CIs."""
    calls = np.ones_like(cost) if calls is None else calls
    ha, ca, _, _ = policy_outcome(correct, cost, calls, choice_a, idk_a)
    hb, cb, _, _ = policy_outcome(correct, cost, calls, choice_b, idk_b)
    n = len(ha)
    rng = np.random.RandomState(seed)
    d, q = np.empty(reps), np.empty(reps)
    for b in range(reps):
        s = rng.randint(0, n, n)
        d[b] = ha[s].mean() - hb[s].mean()
        q[b] = ca[s].sum() / max(cb[s].sum(), 1e-12)
    return {"acc_diff": float(ha.mean() - hb.mean()), "acc_diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "cost_ratio": float(ca.sum() / max(cb.sum(), 1e-12)),
            "cost_ratio_ci": [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))],
            "b_only_a": int(((ha == 1) & (hb == 0)).sum()), "c_only_b": int(((ha == 0) & (hb == 1)).sum())}


def bootstrap_vs_mix(correct, cost, calls, choice, idk=None, reps=1000, seed=0):
    """95% CI for 'accuracy minus the best random mix of fixed strategies at the same cost'.
    The mix frontier is re-estimated inside every bootstrap sample, so its noise is included."""
    hit, c, _, _ = policy_outcome(correct, cost, calls, choice, idk)
    n = len(hit)
    rng = np.random.RandomState(seed)

    def gap(s):
        hull = fixed_frontier(correct[s], cost[s], calls[s])
        return hit[s].mean() - frontier_acc(hull, c[s].sum() / len(s) * 1000)

    vals = np.array([gap(rng.randint(0, n, n)) for _ in range(reps)])
    return {"vs_mix": float(gap(np.arange(n))), "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def mcnemar_p(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def label_metrics(y, pred, k=3):
    y, pred = np.asarray(y), np.asarray(pred)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rec = [cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan") for i in range(k)]
    f1 = []
    for i in range(k):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"n": int(len(y)), "accuracy": float((y == pred).mean()), "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(np.nanmean(rec)),
            "recall": {s: float(v) for s, v in zip(STRATS, rec)}, "confusion": cm.tolist()}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
