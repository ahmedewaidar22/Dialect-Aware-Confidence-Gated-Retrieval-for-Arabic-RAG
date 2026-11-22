"""Phase 4c: decide AFTER C1 has answered whether to escalate to C2 (torch-free).

Two kinds of evidence:
  * answer checks  - is C1's answer backed by the passages it read? did it say 'I don't know'? does it
                     agree with what the model says without retrieval (C0)?  (cheap, no extra LLM call
                     except the optional C0 one)
  * passage reader - an AraBERT cross-encoder reads the question + C1's passages and predicts whether
                     they are enough (trained by scripts/train_sufficiency.py). It can run BEFORE C1.
Escalating costs C1 + C2 for that question (C2 restarts; its first search equals C1's anyway), which is
exactly what the stored outcomes let us price - so every policy here is evaluated without new API calls.
"""
import re

import numpy as np

from ..retrieval.text import normalize_ar, tokenize_ar
from .outcome import IDK, fixed_frontier, frontier_acc

ANSWER_FEATURES = ["c1_idk", "c1_len", "c1_yesno", "c1_in_passages", "c1_tok_cover", "c1_in_top1"]
C0_FEATURES = ["c0_idk", "c0c1_overlap", "c0c1_same"]          # need a C0 call at run time (cheap)
YES_NO = {"نعم", "لا", "yes", "no"}


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalize_ar(s or "").lower())).strip()


def answer_features(rec, passage_text):
    """rec: one labels.jsonl record; passage_text: id -> text. Uses only what exists at run time."""
    a1 = rec["results"]["C1"].get("answer") or ""
    a0 = rec["results"]["C0"].get("answer") or ""
    pids = rec["results"]["C1"].get("passages") or []
    texts = [_norm(passage_text.get(p, "")) for p in pids]
    joined = " ".join(texts)
    n1 = _norm(a1)
    t1, t0 = set(tokenize_ar(a1)), set(tokenize_ar(a0))
    idk1, idk0 = IDK in a1, IDK in a0
    found = bool(n1) and not idk1 and f" {n1} " in f" {joined} "
    cover = len(t1 & set(joined.split())) / len(t1) if t1 and not idk1 else 0.0
    f = {"c1_idk": float(idk1), "c1_len": float(np.log1p(len(t1))), "c1_yesno": float(n1 in YES_NO),
         "c1_in_passages": float(found), "c1_tok_cover": float(cover),
         "c1_in_top1": float(bool(texts) and bool(n1) and not idk1 and f" {n1} " in f" {texts[0]} "),
         "c0_idk": float(idk0),
         "c0c1_overlap": len(t0 & t1) / max(1, len(t1)) if not (idk0 or idk1) else 0.0,
         "c0c1_same": float(bool(n1) and not idk1 and _norm(a0) == n1)}
    return f


def reader_input(question, passages, words=60):
    """Text for the passage reader: question first (never truncated away), then each passage cut to `words`."""
    parts = [" ".join((p or "").split()[:words]) for p in passages]
    return question + " [SEP] " + " [SEP] ".join(parts)


# ---------------------------------------------------------------- policies
def escalate_outcome(correct, cost, esc, with_c0=False, pre=False):
    """Per-question (hit, cost). post (default): run C1, then C2 for escalated ones (pay both).
    pre: decide before answering - run C1 OR C2.  with_c0: also pay the C0 call the detector needed."""
    esc = np.asarray(esc, bool)
    hit = np.where(esc, correct[:, 2], correct[:, 1]).astype(float)
    c = np.where(esc, cost[:, 2], cost[:, 1]) if pre else cost[:, 1] + esc * cost[:, 2]
    if with_c0:
        c = c + cost[:, 0]
    return hit, c


def top_share(score, share, tie_seed=0):
    """Escalate the `share` of questions with the highest score (ties broken at random, reproducibly)."""
    score = np.asarray(score, float)
    n = len(score)
    k = int(round(share * n))
    order = np.lexsort((np.random.RandomState(tie_seed).rand(n), -score))
    esc = np.zeros(n, bool)
    esc[order[:k]] = True
    return esc


def threshold_for(score, correct, cost, calls, rule, tol=0.01, with_c0=False, pre=False,
                  shares=tuple(np.round(np.arange(0.02, 1.001, 0.02), 2))):
    """Pick an escalation threshold on TRAINING-fold questions (escalate where score >= threshold).
    Candidates are score quantiles, evaluated with the exact '>=' rule, so tied scores are handled the
    same way here and when the threshold is applied to the held-out fold.
    rule 'match_c2': cheapest threshold whose accuracy is within tol of always-C2 (if none reaches it:
                     the most accurate one).   rule 'best_gain': largest gain over the random mix."""
    score = np.asarray(score, float)
    hull = fixed_frontier(correct, cost, calls)
    target = correct[:, 2].mean() - tol
    cands = sorted({np.inf, *np.quantile(score, 1 - np.asarray(shares)).tolist()}, reverse=True)
    best = None
    for t in cands:
        hit, c = escalate_outcome(correct, cost, score >= t, with_c0, pre)
        acc, cc = hit.mean(), c.mean() * 1000
        if rule == "match_c2":
            key = (1, -cc) if acc >= target - 1e-12 else (0, acc)
        else:
            key = (1, acc - frontier_acc(hull, cc))
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def bootstrap_items(hit, c, correct, cost, calls, refs, reps=1000, seed=0):
    """CIs for: gain over the random mix at equal cost (mix re-estimated per sample), and accuracy
    difference / cost ratio vs each reference policy refs[name] = (hit, cost)."""
    n = len(hit)
    rng = np.random.RandomState(seed)

    def gap(s):
        return hit[s].mean() - frontier_acc(fixed_frontier(correct[s], cost[s], calls[s]), c[s].mean() * 1000)

    g = np.empty(reps)
    d = {k: (np.empty(reps), np.empty(reps)) for k in refs}
    for b in range(reps):
        s = rng.randint(0, n, n)
        g[b] = gap(s)
        for k, (rh, rc) in refs.items():
            d[k][0][b] = hit[s].mean() - rh[s].mean()
            d[k][1][b] = c[s].sum() / max(rc[s].sum(), 1e-12)
    out = {"vs_mix": float(gap(np.arange(n))), "vs_mix_ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}
    for k, (rh, rc) in refs.items():
        out[k] = {"acc_diff": float(hit.mean() - rh.mean()),
                  "acc_diff_ci": [float(np.percentile(d[k][0], 2.5)), float(np.percentile(d[k][0], 97.5))],
                  "cost_ratio": float(c.sum() / max(rc.sum(), 1e-12)),
                  "cost_ratio_ci": [float(np.percentile(d[k][1], 2.5)), float(np.percentile(d[k][1], 97.5))],
                  "only_this": int(((hit == 1) & (rh == 0)).sum()), "only_ref": int(((hit == 0) & (rh == 1)).sum())}
    return out
