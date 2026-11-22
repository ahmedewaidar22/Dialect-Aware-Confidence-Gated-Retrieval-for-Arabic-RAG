"""Phase 4e: the final complexity router as ONE policy with three stages (torch-free).

  stage A  'skip retrieval?'      score_a = P(C0 is enough)      from the AraBERT text head    -> C0
  stage B  'one search enough?'   score_b = P(C2 needed)         from signals before answering -> C2
  stage C  'escalate after C1?'   score_c = P(C2 would rescue)   from C1's answer + passages   -> C1, then C2
A question goes: skip (A) ? C0 : (B ? C2 : C1, then escalate if C).
Thresholds are chosen at fixed operating points:
  budget   - most accurate policy that costs no more than standard RAG (always-C1)
  match_c1 - cheapest policy within `tol` of standard RAG's accuracy ('same quality, cheaper')
  balanced - largest gain over the best random mix of fixed strategies at equal cost
  quality  - cheapest policy within `tol` of always-C2's accuracy (most accurate if none reaches it)
"""
import itertools

import numpy as np

from .outcome import fixed_frontier, frontier_acc

SHARES_A = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
SHARES_B = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2))
SHARES_C = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
RULES = ("budget", "match_c1", "balanced", "quality")


def decide(sa, sb, sc, t):
    """t = (ta, tb, tc) thresholds (np.inf disables a stage). Returns (skip, c2, esc) boolean arrays."""
    skip = np.asarray(sa) >= t[0]
    c2 = ~skip & (np.asarray(sb) >= t[1])
    esc = ~skip & ~c2 & (np.asarray(sc) >= t[2])
    return skip, c2, esc


def outcome(correct, cost, calls, skip, c2, esc):
    """Per question (hit, cost, calls) of the 3-stage policy."""
    hit = np.where(skip, correct[:, 0], np.where(c2 | esc, correct[:, 2], correct[:, 1])).astype(float)
    c = np.where(skip, cost[:, 0], np.where(c2, cost[:, 2], cost[:, 1] + esc * cost[:, 2]))
    k = np.where(skip, calls[:, 0], np.where(c2, calls[:, 2], calls[:, 1] + esc * calls[:, 2]))
    return hit, c, k


def _cands(score, shares):
    s = np.asarray(score, float)
    return sorted({np.inf, *[float(np.quantile(s, 1 - sh)) for sh in shares if sh > 0]}, reverse=True)


def search(sa, sb, sc, correct, cost, calls, rule, tol=0.01, use=(True, True, True)):
    """Best thresholds (ta, tb, tc) for `rule` on these questions. use=(A, B, C) switches stages on/off."""
    hull = fixed_frontier(correct, cost, calls)
    c1_cost, c1_acc, c2_acc = cost[:, 1].mean(), correct[:, 1].mean(), correct[:, 2].mean()
    ca = _cands(sa, SHARES_A) if use[0] else [np.inf]
    cb = _cands(sb, SHARES_B) if use[1] else [np.inf]
    cc = _cands(sc, SHARES_C) if use[2] else [np.inf]
    best = None
    for t in itertools.product(ca, cb, cc):
        hit, c, _ = outcome(correct, cost, calls, *decide(sa, sb, sc, t))
        acc, cm = hit.mean(), c.mean()
        if rule == "budget":
            key = (1, acc, -cm) if cm <= c1_cost + 1e-12 else (0, -cm, acc)
        elif rule == "match_c1":
            key = (1, -cm, acc) if acc >= c1_acc - tol - 1e-12 else (0, acc, -cm)
        elif rule == "quality":
            key = (1, -cm, acc) if acc >= c2_acc - tol - 1e-12 else (0, acc, -cm)
        else:
            key = (1, acc - frontier_acc(hull, cm * 1000), -cm)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def honest(sa, sb, sc, fold, correct, cost, calls, rule, tol=0.01, use=(True, True, True)):
    """Thresholds chosen on the other folds, applied to each held-out fold. Returns (skip, c2, esc, per-fold t)."""
    n = len(fold)
    skip, c2, esc = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    ts = {}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        t = search(sa[tr], sb[tr], sc[tr], correct[tr], cost[tr], calls[tr], rule, tol, use)
        skip[te], c2[te], esc[te] = decide(sa[te], sb[te], sc[te], t)
        ts[int(f)] = [float(x) for x in t]
    return skip, c2, esc, ts
