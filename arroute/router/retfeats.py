"""Phase 4b: retrieval-confidence signals for the router (torch-free; the encoder is passed in).

Idea: whether ONE search is enough (C1) or iterative search is needed (C2) depends on what the retriever
finds, which the question text alone cannot show. So we run the same first search C1 would run and
describe how confident it looks: how high and how peaked the scores are, whether BM25 and the dense
retriever agree, and how much of the question the top passages cover. No gold information is used.
"""
import numpy as np

from ..retrieval.index import rrf
from ..retrieval.text import tokenize_ar

FEATURES = [
    "q_tokens",
    "bm25_top1", "bm25_top1_per_tok", "bm25_gap12", "bm25_gap15",
    "dense_top1", "dense_gap12", "dense_gap15", "dense_mean10",
    "agree_top1", "overlap5", "overlap10", "dense_top1_rank_in_bm25",
    "hyb_dense_mean5", "hyb_coherence5", "cover_top1", "cover_top5",
]


def _rel_gap(a, b):
    return float((a - b) / (abs(a) + 1e-9))


def question_features(query, bm25_scores, dense_scores, ids, emb, tok_sets, depth=100, k=5):
    """bm25_scores / dense_scores: score of every passage for this query (same order as ids/emb).
    tok_sets: passage index -> set of tokens (for coverage). Returns (features dict, hybrid top-k ids)."""
    q_toks = set(tokenize_ar(query))
    if bm25_scores is None:                       # no Arabic/latin token: BM25 returns nothing, like Retriever
        b_order, bm25_scores = np.array([], int), np.zeros(len(ids))
    else:
        b_order = np.argsort(-bm25_scores)[:depth]
    d_order = np.argsort(-dense_scores)[:depth]
    b = np.concatenate([bm25_scores[b_order], np.zeros(5)])[:max(5, len(b_order))]
    d = dense_scores[d_order]
    pos = {ids[i]: i for i in np.concatenate([b_order, d_order])}
    hyb = rrf([[ids[i] for i in b_order], [ids[i] for i in d_order]])[:k]   # exactly what C1 retrieves
    h_idx = [pos[p] for p in hyb]
    b_rank = {int(i): r for r, i in enumerate(b_order)}
    hv = emb[h_idx]
    sims = hv @ hv.T
    m = len(h_idx)
    cover = [len(q_toks & tok_sets[i]) / max(1, len(q_toks)) for i in h_idx]
    union = set().union(*[tok_sets[i] for i in h_idx]) if h_idx else set()
    f = {
        "q_tokens": float(np.log1p(len(q_toks))),
        "bm25_top1": float(b[0]), "bm25_top1_per_tok": float(b[0] / max(1, len(q_toks))),
        "bm25_gap12": _rel_gap(b[0], b[1]), "bm25_gap15": _rel_gap(b[0], b[4]),
        "dense_top1": float(d[0]), "dense_gap12": float(d[0] - d[1]), "dense_gap15": float(d[0] - d[4]),
        "dense_mean10": float(d[:10].mean()),
        "agree_top1": float(len(b_order) > 0 and b_order[0] == d_order[0]),
        "overlap5": len(set(b_order[:5]) & set(d_order[:5])) / 5.0,
        "overlap10": len(set(b_order[:10]) & set(d_order[:10])) / 10.0,
        "dense_top1_rank_in_bm25": float(np.log1p(b_rank.get(int(d_order[0]), depth))),
        "hyb_dense_mean5": float(dense_scores[h_idx].mean()),
        "hyb_coherence5": float((sims.sum() - np.trace(sims)) / max(1, m * (m - 1))),
        "cover_top1": float(cover[0]) if cover else 0.0,
        "cover_top5": len(q_toks & union) / max(1, len(q_toks)),
    }
    assert list(f) == FEATURES
    return f, hyb


def compute_all(questions, ids, texts, emb, encode, bm25, depth=100, k=5, log=print):
    """questions: list of str. bm25: rank_bm25 object over `texts` (same order). encode: texts -> unit vectors."""
    from ..retrieval.text import normalize_ar
    tok_sets = [set(tokenize_ar(t)) for t in texts]
    qv = encode([normalize_ar(q) for q in questions])
    emb = np.asarray(emb, dtype=np.float32)
    feats, hybs = [], []
    for n, (q, v) in enumerate(zip(questions, qv)):
        toks = tokenize_ar(q)
        bs = np.asarray(bm25.get_scores(toks), dtype=np.float64) if toks else None
        f, h = question_features(q, bs, emb @ v, ids, emb, tok_sets, depth, k)
        feats.append(f)
        hybs.append(h)
        if (n + 1) % 200 == 0:
            log(f"  {n + 1}/{len(questions)}")
    return feats, hybs
