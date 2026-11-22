"""Phase 4d: signals from a pretrained cross-encoder reranker (BAAI/bge-reranker-v2-m3), torch-free part.

The reranker reads (question, passage) together and scores how well the passage answers the question.
We score the top-20 hybrid candidates (C1 uses the first 5 of exactly this list):
  * BEFORE C1 answers: are C1's 5 passages relevant? are better passages waiting deeper (ranks 6-20) that
    an iterative search (C2) could reach?
  * AFTER C1 answers: does adding C1's answer to the question make the passages MORE relevant
    (= the answer is supported), or not?
"""
import numpy as np

RERANK_PRE = ["rr_top5_max", "rr_top5_mean", "rr_top5_2nd", "rr_top5_npos", "rr_rank1",
              "rr_deep_max", "rr_deep_gain", "rr_deep_npos"]
RERANK_POST = ["rr_ans_max", "rr_ans_gain"]


def rerank_features(scores20, ans_scores5=None, c1_idk=False):
    """scores20: reranker logits for the top-20 hybrid passages, in retrieval order (C1 = first 5).
    ans_scores5: logits for (question + C1 answer, passage) on C1's 5 passages (None/IDK -> neutral)."""
    s = np.asarray(scores20, float)
    top5, deep = s[:5], s[5:]
    srt = np.sort(top5)[::-1]
    f = {"rr_top5_max": float(srt[0]), "rr_top5_mean": float(top5.mean()),
         "rr_top5_2nd": float(srt[1]) if len(srt) > 1 else float(srt[0]),
         "rr_top5_npos": float((top5 > 0).sum()), "rr_rank1": float(top5[0]),
         "rr_deep_max": float(deep.max()) if len(deep) else float(srt[0]),
         "rr_deep_gain": float(deep.max() - srt[0]) if len(deep) else 0.0,
         "rr_deep_npos": float((deep > 0).sum()) if len(deep) else 0.0}
    if ans_scores5 is None or c1_idk or len(ans_scores5) == 0:
        f["rr_ans_max"], f["rr_ans_gain"] = float(srt[0]), 0.0            # nothing to verify
    else:
        a = np.asarray(ans_scores5, float)
        f["rr_ans_max"], f["rr_ans_gain"] = float(a.max()), float(a.max() - srt[0])
    assert list(f) == RERANK_PRE + RERANK_POST
    return f
