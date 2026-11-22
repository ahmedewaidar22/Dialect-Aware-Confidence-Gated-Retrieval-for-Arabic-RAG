"""Phase 7: the FULL router - dialect-aware AND complexity-aware - for one question at a time.

  question
    -> dialect head: P(Egyptian)
         >= hi : rewrite to MSA, search the rewrite
         <= lo : search as typed
         else  : search BOTH and merge the rankings (safe either way, Phase 6b)
    -> stage A (text head):        P(C0 enough) >= ta ?            -> C0 (no retrieval)
    -> stage B (pre-answer signals): P(C2 needed) >= tb ?          -> C2
    -> otherwise C1; after C1 answers, stage C: P(C2 rescues) >= tc -> also run C2

Every model is passed in as a plain function, so the logic is testable without torch. load_router() wires
the real saved models (models/dialect_head, models/complexity_router) and the corpus.
Features are computed with the SAME code that produced the training features (retfeats, rerankfeats, escalate).
"""
import json
from pathlib import Path

import numpy as np

from ..retrieval.index import rrf
from ..retrieval.text import normalize_ar, tokenize_ar
from ..router.escalate import ANSWER_FEATURES, answer_features, reader_input
from ..router.outcome import IDK
from ..router.rerankfeats import RERANK_POST, RERANK_PRE, rerank_features
from ..router.retfeats import FEATURES as RET_FEATURES, question_features

MODES = ("budget", "match_c1", "balanced", "quality")


def fuse(per_query, depth=100):
    """Merge all_methods() outputs of several queries: RRF of all their BM25 + dense lists (= hybrid)."""
    if len(per_query) == 1:
        return per_query[0]["hybrid"][:depth]
    return rrf([r[m] for r in per_query for m in ("bm25", "dense")])[:depth]


class ARRouter:
    def __init__(self, retriever, thresholds, dialect, text_head, reader_pre, reader_ans, stage_b, stage_c,
                 reranker, rewrite, lo=0.2, hi=0.8, mode="quality", words=60, n_rerank=20):
        """dialect(texts)->P(egy); text_head(texts)->(n,3) probs; reader_*(texts)->P(C1 ok);
        stage_*(X)->P(positive); reranker(pairs)->scores; rewrite(q)->MSA text; thresholds: {mode: {ta,tb,tc}}."""
        self.r, self.th = retriever, thresholds
        self.dialect, self.text_head, self.reader_pre, self.reader_ans = dialect, text_head, reader_pre, reader_ans
        self.stage_b, self.stage_c, self.reranker, self.rewrite = stage_b, stage_c, reranker, rewrite
        self.lo, self.hi, self.mode, self.words, self.n_rerank = lo, hi, mode, words, n_rerank
        self._tok_sets = None

    # ------------------------------------------------------------ helpers
    def _t(self, key, mode=None):
        v = self.th[mode or self.mode].get(key)
        return np.inf if v is None else float(v)

    def _token_sets(self):
        if self._tok_sets is None:
            self._tok_sets = [set(tokenize_ar(self.r.text(p))) for p in self.r.bm25.ids]
        return self._tok_sets

    def retrieval_features(self, query):
        toks = tokenize_ar(query)
        bs = np.asarray(self.r.bm25.bm25.get_scores(toks), dtype=np.float64) if toks else None
        ds = self.r.dense.emb @ self.r.dense.encode([normalize_ar(query)])[0]
        f, _ = question_features(query, bs, ds, self.r.bm25.ids, self.r.dense.emb, self._token_sets())
        return [f[k] for k in RET_FEATURES]

    # ------------------------------------------------------------ dialect step
    def plan(self, question, force=None):
        """force=None: the dialect band decides. force='typed': search as typed (no-rewrite ablation)."""
        p = float(self.dialect([question])[0])
        if force == "typed":
            return {"p_egy": p, "dialect": "typed (forced)", "query": question, "search": [question]}
        if p >= self.hi:
            q = self.rewrite(question)
            return {"p_egy": p, "dialect": "egy", "query": q, "search": [q]}
        if p <= self.lo:
            return {"p_egy": p, "dialect": "msa", "query": question, "search": [question]}
        q = self.rewrite(question)
        return {"p_egy": p, "dialect": "unsure", "query": q, "search": [question, q]}

    # ------------------------------------------------------------ complexity step (before answering)
    def route(self, question, mode=None, force=None, full=False):
        """full=True computes EVERY score (no early exit), so all operating points can be evaluated offline."""
        st = {"question": question, "mode": mode or self.mode, **self.plan(question, force)}
        q = st["query"]
        st["probs3"] = [float(x) for x in self.text_head([q])[0]]
        st["p_c0"] = st["probs3"][0]
        cands = fuse([self.r.all_methods(s, k=100) for s in st["search"]])[:self.n_rerank]
        st["cands"], st["passages"] = cands, cands[:5]
        if st["p_c0"] >= self._t("ta", st["mode"]) and not full:
            st["strategy"] = "C0"
            return st
        texts = [self.r.text(p) for p in cands]
        st["rr20"] = [float(x) for x in self.reranker([(q, t) for t in texts])]
        rr = rerank_features(st["rr20"])
        st["p_ok_pre"] = float(self.reader_pre([reader_input(q, texts[:5], self.words)])[0])
        st["x_pre"] = [st["p_ok_pre"]] + self.retrieval_features(q) + [rr[k] for k in RERANK_PRE]
        st["p_need_c2"] = float(self.stage_b(np.array([st["x_pre"]]))[0])
        if st["p_c0"] >= self._t("ta", st["mode"]):
            st["strategy"] = "C0"
        else:
            st["strategy"] = "C2" if st["p_need_c2"] >= self._t("tb", st["mode"]) else "C1"
        return st

    # ------------------------------------------------------------ stage C (after C1 answered)
    def after_c1(self, st, c1_answer):
        q, texts = st["query"], [self.r.text(p) for p in st["passages"]]
        rec = {"results": {"C1": {"answer": c1_answer, "passages": st["passages"]}, "C0": {"answer": ""}}}
        af = answer_features(rec, dict(zip(st["passages"], texts)))
        idk = IDK in (c1_answer or "")
        ans5 = None if idk or not (c1_answer or "").strip() else \
            [float(x) for x in self.reranker([(f"{q} الجواب: {c1_answer}", t) for t in texts])]
        rr = rerank_features(st["rr20"], ans5, idk)
        p_ans = float(self.reader_ans([reader_input(f"{q} [SEP] الجواب: {(c1_answer or '').strip()}", texts,
                                                    self.words)])[0])
        x = st["x_pre"] + [af[k] for k in ANSWER_FEATURES] + [rr[k] for k in RERANK_POST] + [p_ans]
        st["p_rescue"] = float(self.stage_c(np.array([x]))[0])
        st["escalate"] = st["p_rescue"] >= self._t("tc", st["mode"])
        return st["escalate"]

    # ------------------------------------------------------------ full answer (uses the LLM)
    def answer(self, question, gen, mode=None):
        """Run the whole pipeline. gen: an object with .ask(prompt) (see label_outcomes.Generator)."""
        from .strategies import run_c0, run_c1, run_c2
        st = self.route(question, mode)
        wrapped = _RoutedRetriever(self.r, st)
        if st["strategy"] == "C0":
            res = run_c0(st["query"], gen)
        elif st["strategy"] == "C2":
            res = run_c2(st["query"], gen, wrapped)
        else:
            res = run_c1(st["query"], gen, wrapped)
            if self.after_c1(st, res["answer"]):
                c2 = run_c2(st["query"], gen, wrapped)
                c2["calls"] += res["calls"]
                c2["input_tokens"] += res["input_tokens"]
                c2["output_tokens"] += res["output_tokens"]
                c2["escalated_from_c1"] = res["answer"]
                res = c2
        st["result"] = res
        return st


class _RoutedRetriever:
    """For the routed query, return the routed (possibly merged) ranking; any other query (C2's follow-up
    searches) goes to the normal hybrid search."""

    def __init__(self, retriever, st):
        self.r, self.st = retriever, st

    def search(self, q, k=10, method="hybrid"):
        if q == self.st["query"]:
            return self.st["cands"][:k]
        return self.r.search(q, k=k, method=method)

    def text(self, pid):
        return self.r.text(pid)


# ---------------------------------------------------------------- loading the real models
class HFClassifier:
    """Saved Hugging Face sequence classifier -> calibrated probabilities (temperature T)."""

    def __init__(self, path, T=1.0, max_len=128, device=None, batch=32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.T, self.max_len, self.batch = torch, float(T), max_len, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()

    def probs(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(list(texts[i:i + self.batch]), padding=True, truncation=True, max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                lg = self.model(**enc).logits.float().cpu().numpy() / self.T
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                out.append(e / e.sum(axis=1, keepdims=True))
        return np.vstack(out) if out else np.zeros((0, 2))


def load_router(models="models", corpus="data/corpus", device=None, rewrite=None, mode="quality",
                complexity_dir="complexity_router"):
    """Wire the saved models. rewrite: function(question)->MSA (e.g. a cached LLM rewriter)."""
    import joblib
    from ..retrieval.index import load_retriever
    from ..retrieval.rerank import Reranker
    m, c = Path(models), Path(models) / complexity_dir
    dmeta = json.loads((m / "dialect_head" / "meta.json").read_text(encoding="utf-8"))
    pol = json.loads((c / "policy.json").read_text(encoding="utf-8"))
    dia = HFClassifier(m / "dialect_head", dmeta["temperature"], dmeta.get("max_len", 64), device)
    th = HFClassifier(c / "text_head", pol["text_head"]["temperature"], 128, device)
    rp = HFClassifier(c / "reader_pre", pol["reader_pre"]["temperature"], 512, device, batch=8)
    ra = HFClassifier(c / "reader_ans", pol["reader_ans"]["temperature"], 512, device, batch=8)
    sb, sc = joblib.load(c / "stage_b.joblib"), joblib.load(c / "stage_c.joblib")
    rer = Reranker(pol["models"].get("reranker", "BAAI/bge-reranker-v2-m3"), device)
    retr = load_retriever(corpus, dense=True, device=device)
    if rewrite is None:
        raise ValueError("pass rewrite=... (an Egyptian->MSA function), e.g. from arroute.egy.rewrite.rewrite_llm")
    return ARRouter(retr, pol["thresholds"], lambda t: dia.probs(t)[:, 1], th.probs, lambda t: rp.probs(t)[:, 1],
                    lambda t: ra.probs(t)[:, 1], lambda X: sb.predict_proba(X)[:, 1], lambda X: sc.predict_proba(X)[:, 1],
                    rer.score, rewrite, dmeta.get("lo", 0.2), dmeta.get("hi", 0.8), mode)
