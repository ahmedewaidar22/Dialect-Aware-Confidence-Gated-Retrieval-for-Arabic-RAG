"""Router v2: a confidence-aware CASCADE (torch-free core).

  C0 always answers first (no search; ~$0.04 per 1k questions) WITH token log-probabilities
  detector 0  "is C0's answer right?"   <- C0 confidence + text head + search/reranker signals
      yes -> done
  C1R  answers from the top-k RERANKED passages (fewer, better passages: cheaper than C1's hybrid top-5)
  detector 1  "is C1R's answer right?"  <- C1R confidence + 'I don't know' + answer checks + all of the above
      no  -> C2 (iterative search)

Why (Phase 8): the text-only skip decision did not transfer to new question types (C0 was confidently wrong on
ArDQA), and simple rules sit on the C1-C2 line. Answer-level confidence (token log-probabilities) is the
signal recent work found most reliable and cheapest for 'do I need retrieval?' (Moskvoretskii et al., ACL 2025),
and looking at what retrieval found is what text-only routers miss.
"""
import itertools
import re

import numpy as np

from ..retrieval.text import normalize_ar, tokenize_ar
from .outcome import IDK, PRICE_IN, PRICE_OUT
from .rerankfeats import RERANK_PRE, rerank_features
from .retfeats import FEATURES as RET_FEATURES

CONF = ["lp_mean", "lp_min", "lp_first", "lp_sum", "ent_mean", "ent_max", "margin_mean", "margin_min", "n_tok", "idk"]
PRE = ["p_c0", "p_c1", "p_c2"] + RET_FEATURES + RERANK_PRE
F0 = PRE + [f"c0_{k}" for k in CONF]
CHECKS = ["c1_in_passages", "c1_tok_cover", "c1_in_top1", "c1_yesno", "c0c1_same", "c0c1_overlap", "rr_ans_max", "rr_ans_gain"]
F1 = F0 + [f"c1_{k}" for k in CONF] + CHECKS
NO_CONF = lambda feats: [f for f in feats if not re.match(r"c[01]_(lp_|ent_|margin_|n_tok)", f)]   # noqa: E731
RULES = ("budget", "balanced", "quality")
YES_NO = {"نعم", "لا", "yes", "no"}


# ---------------------------------------------------------------- confidence from token log-probabilities
def _answer_tokens(tokens):
    """Drop leading/trailing whitespace/quote tokens; keep the answer's own tokens."""
    keep = [t for t in tokens if t.get("t", "").strip().strip('"').strip()]
    return keep or tokens


def conf_features(tokens, answer):
    """tokens: [{"t", "lp", "top": [lp,...]}]. Mean/min log-prob, entropy (from the top alternatives) and the
    margin between the best and second-best token, over the answer's tokens."""
    tk = _answer_tokens(tokens or [])
    if not tk:
        return {"lp_mean": -5.0, "lp_min": -5.0, "lp_first": -5.0, "lp_sum": -5.0, "ent_mean": 2.0, "ent_max": 2.0,
                "margin_mean": 0.0, "margin_min": 0.0, "n_tok": 0.0, "idk": float(IDK in (answer or ""))}
    lps = np.array([t["lp"] for t in tk], float)
    ents, margins = [], []
    for t in tk:
        top = np.array(sorted(t.get("top") or [t["lp"]], reverse=True), float)
        p = np.exp(top)
        rest = max(0.0, 1.0 - p.sum())
        ents.append(float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() - (rest * np.log(rest) if rest > 1e-12 else 0.0)))
        margins.append(float(top[0] - top[1]) if len(top) > 1 else 10.0)
    return {"lp_mean": float(lps.mean()), "lp_min": float(lps.min()), "lp_first": float(lps[0]),
            "lp_sum": float(lps.sum()), "ent_mean": float(np.mean(ents)), "ent_max": float(np.max(ents)),
            "margin_mean": float(np.mean(margins)), "margin_min": float(np.min(margins)),
            "n_tok": float(np.log1p(len(tk))), "idk": float(IDK in (answer or ""))}


def clean(text):
    from ..pipeline.strategies import clean_answer
    return clean_answer(text)


def run_c0_lp(q, ask):
    """ask(prompt) -> (text, usage, cached, tokens)."""
    from ..pipeline.strategies import P_C0
    text, usage, cached, toks = ask(P_C0.format(q=q))
    a = clean(text)
    return {"answer": a, "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "calls": 1, "searches": 0, "cached": bool(cached), "conf": conf_features(toks, a)}


def run_c1r_lp(q, pids, texts, ask):
    from ..pipeline.strategies import P_C1, fmt_passages
    text, usage, cached, toks = ask(P_C1.format(q=q, passages=fmt_passages(texts)))
    a = clean(text)
    return {"answer": a, "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "calls": 1, "searches": 1, "cached": bool(cached), "passages": list(pids), "conf": conf_features(toks, a)}


def reranked(cands, rr20, k):
    """Top-k of the candidates by reranker score (ties keep retrieval order)."""
    order = sorted(range(len(cands)), key=lambda i: (-rr20[i], i))
    return [cands[i] for i in order[:k]]


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalize_ar(s or "").lower())).strip()


def answer_checks(a1, a0, passage_texts, rr_ans=None, rr_top=0.0):
    """Answer checks on C1R's own passages (same ideas as Phase 4c) + the reranker's view of the answer."""
    texts = [_norm(t) for t in passage_texts]
    joined = " ".join(texts)
    n1, idk1, idk0 = _norm(a1), IDK in (a1 or ""), IDK in (a0 or "")
    t1, t0 = set(tokenize_ar(a1 or "")), set(tokenize_ar(a0 or ""))
    ok = bool(n1) and not idk1
    f = {"c1_in_passages": float(ok and f" {n1} " in f" {joined} "),
         "c1_tok_cover": len(t1 & set(joined.split())) / len(t1) if t1 and not idk1 else 0.0,
         "c1_in_top1": float(ok and bool(texts) and f" {n1} " in f" {texts[0]} "),
         "c1_yesno": float(n1 in YES_NO), "c0c1_same": float(ok and _norm(a0) == n1),
         "c0c1_overlap": len(t0 & t1) / max(1, len(t1)) if not (idk0 or idk1) else 0.0}
    if rr_ans is None or not ok or len(rr_ans) == 0:
        f["rr_ans_max"], f["rr_ans_gain"] = float(rr_top), 0.0
    else:
        f["rr_ans_max"], f["rr_ans_gain"] = float(max(rr_ans)), float(max(rr_ans) - rr_top)
    return f


def pre_features(probs3, x_pre, rr20):
    """probs3: text head; x_pre: Phase 4e/7 pre-answer vector [reader_pre] + RET + RERANK_PRE (reader dropped)."""
    ret = list(x_pre[1:1 + len(RET_FEATURES)])
    rr = rerank_features(rr20)
    return dict(zip(PRE, list(probs3) + ret + [rr[k] for k in RERANK_PRE]))


def tok_cost(res, pin=PRICE_IN, pout=PRICE_OUT):
    return (res.get("input_tokens", 0) * pin + res.get("output_tokens", 0) * pout) / 1e6


# ---------------------------------------------------------------- the cascade policy
def decide(p0, p1, ta, tc):
    """accept C0 if p0 >= ta; otherwise C1R, and escalate to C2 if p1 < tc."""
    acc0 = np.asarray(p0) >= ta
    esc = ~acc0 & (np.asarray(p1) < tc)
    return acc0, esc


def outcome(OC, acc0, esc):
    """OC: dict with correct/cost/calls/searches arrays of shape (n, 3) for (C0L, C1R, C2)."""
    c, k = OC["correct"], OC["cost"]
    hit = np.where(acc0, c[:, 0], np.where(esc, c[:, 2], c[:, 1])).astype(float)
    cost = k[:, 0] + ~acc0 * (k[:, 1] + esc * k[:, 2])
    calls = OC["calls"][:, 0] + ~acc0 * (OC["calls"][:, 1] + esc * OC["calls"][:, 2])
    srch = ~acc0 * (OC["searches"][:, 1] + esc * OC["searches"][:, 2])
    route = np.where(acc0, "C0", np.where(esc, "C1R>C2", "C1R"))
    return hit, cost, calls, srch, route


def frontier(OC):
    from .outcome import fixed_frontier
    return fixed_frontier(OC["correct"], OC["cost"], OC["calls"])


def _cands(p, shares):
    return sorted({np.inf, -np.inf, *[float(np.quantile(p, 1 - s)) for s in shares if 0 < s < 1]})


def search(p0, p1, OC, rule, tol=0.01):
    """Thresholds (ta, tc) for `rule`, chosen on THESE questions (pass out-of-fold scores of training data).
    budget   - most accurate cascade that costs no more than always-C1R
    balanced - largest gain over the best random mix of always-C0/C1R/C2 at equal cost
    quality  - cheapest cascade within `tol` of always-C2's accuracy (most accurate if none reaches it)"""
    from .outcome import frontier_acc
    hull = frontier(OC)
    c1r_cost = OC["cost"][:, 1].mean()
    c2_acc = OC["correct"][:, 2].mean()
    ta_c = _cands(p0, np.arange(0.05, 0.71, 0.05))
    tc_c = _cands(p1, np.arange(0.05, 1.0, 0.05))
    best = None
    for ta, tc in itertools.product(ta_c, tc_c):
        hit, cost, *_ = outcome(OC, *decide(p0, p1, ta, tc))
        acc, cm = hit.mean(), cost.mean()
        if rule == "budget":
            key = (1, acc, -cm) if cm <= c1r_cost + 1e-12 else (0, -cm, acc)
        elif rule == "quality":
            key = (1, -cm, acc) if acc >= c2_acc - tol - 1e-12 else (0, acc, -cm)
        else:
            key = (1, acc - frontier_acc(hull, cm * 1000), -cm)
        if best is None or key > best[0]:
            best = (key, (float(ta), float(tc)))
    return best[1]


# ---------------------------------------------------------------- detectors
class _Const:
    """Fallback when a training set has only one class (tiny data): always predicts that rate."""

    def __init__(self, rate):
        self.rate = float(rate)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


class _Model:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            self.m = _Const(y.mean() if len(y) else 0.5)
        else:
            self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                                    min_samples_leaf=25, l2_regularization=1.0,
                                                    random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


def make_model(seed=0):
    return _Model(seed)


def oof_scores(X, y, groups=None, k=5, seed=0):
    """Out-of-fold P(correct): every training question is scored by a model that never saw it."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    out = np.zeros(len(y))
    y = np.asarray(y).astype(int)
    if groups is not None and len(np.unique(groups)) >= k:
        splits = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        kk = max(2, min(k, int(np.bincount(y, minlength=2).min()) or 2))
        splits = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed).split(X, y) if np.bincount(
            y, minlength=2).min() >= 2 else [(np.arange(len(y)), np.arange(len(y)))]
    for tr, te in splits:
        m = make_model(seed).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out
