"""Phase 10: the FINAL system (the thesis result), one question at a time.

  question
    -> dialect head: P(Egyptian)
         >= hi : rewrite to MSA          <= lo : search as typed          else : search both, merge
    -> hybrid search (BM25 + BGE-M3), top-20 candidates
    -> reranker (bge-reranker-v2-m3) -> the top-K (2) passages
    -> C1 answers from those K passages
    -> answer is "I don't know"?  -> C2 (iterative search) answers instead

Measured on the 1,000 Phase 8 test questions (Phase 9): Egyptian input 78.7% at $0.456 per 1k questions, vs
standard RAG 69.4% at $0.534 and always-C2 77.4% at $0.936. Every model is passed in as a plain function, so the
logic is testable without torch; load_final() wires the saved models.
"""
import json
from pathlib import Path

from ..retrieval.index import rrf
from ..router.outcome import IDK


def fuse(per_query, depth=100):
    if len(per_query) == 1:
        return per_query[0]["hybrid"][:depth]
    return rrf([r[m] for r in per_query for m in ("bm25", "dense")])[:depth]


class FinalSystem:
    def __init__(self, retriever, dialect, reranker, rewrite, lo=0.2, hi=0.8, k=2, n_cands=20):
        """dialect(texts)->P(egy); reranker(pairs)->scores; rewrite(q)->MSA."""
        self.r, self.dialect, self.reranker, self.rewrite = retriever, dialect, reranker, rewrite
        self.lo, self.hi, self.k, self.n_cands = lo, hi, k, n_cands

    def plan(self, question):
        p = float(self.dialect([question])[0])
        if p >= self.hi:
            q = self.rewrite(question)
            return {"p_egy": p, "dialect": "egy", "query": q, "search": [q]}
        if p <= self.lo:
            return {"p_egy": p, "dialect": "msa", "query": question, "search": [question]}
        q = self.rewrite(question)
        return {"p_egy": p, "dialect": "unsure", "query": q, "search": [question, q]}

    def retrieve(self, question):
        """-> plan + top-20 candidates + reranker scores + the top-K passages C1 reads."""
        st = {"question": question, **self.plan(question)}
        cands = fuse([self.r.all_methods(s, k=100) for s in st["search"]])[:self.n_cands]
        scores = [float(x) for x in self.reranker([(st["query"], self.r.text(p)) for p in cands])] if cands else []
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
        st.update(cands=cands, rr=scores, passages=[cands[i] for i in order[:self.k]])
        return st

    def answer(self, question, ask_c1, gen):
        """ask_c1(prompt) -> (text, usage, cached, tokens)  (the Phase 9 call: Chat Completions with log-probs);
        gen: object with .ask(prompt) -> (text, usage, cached) for C2 (the Phase 8 call)."""
        from ..router.v2 import run_c1r_lp
        from .strategies import run_c2
        st = self.retrieve(question)
        c1 = run_c1r_lp(st["query"], st["passages"], [self.r.text(p) for p in st["passages"]], ask_c1)
        st["c1"] = c1
        st["escalated"] = IDK in (c1["answer"] or "") or not (c1["answer"] or "").strip()
        if st["escalated"]:
            c2 = run_c2(st["query"], gen, _Routed(self.r, st["query"], st["cands"]))
            st["c2"] = {k: c2[k] for k in ("answer", "calls", "searches", "input_tokens", "output_tokens")}
            st["answer"] = c2["answer"]
        else:
            st["answer"] = c1["answer"]
        st["calls"] = 1 + (st["c2"]["calls"] if st["escalated"] else 0)
        st["input_tokens"] = c1["input_tokens"] + (st["c2"]["input_tokens"] if st["escalated"] else 0)
        st["output_tokens"] = c1["output_tokens"] + (st["c2"]["output_tokens"] if st["escalated"] else 0)
        return st


class _Routed:
    """C2's first search = the routed candidates (as in Phase 8); its follow-up searches = normal hybrid search."""

    def __init__(self, retriever, query, cands):
        self.r, self.query, self.cands = retriever, query, cands

    def search(self, q, k=10, method="hybrid"):
        if q == self.query and k <= len(self.cands):
            return self.cands[:k]
        return self.r.search(q, k=k, method=method)

    def text(self, pid):
        return self.r.text(pid)


def load_final(models="models", corpus="data/corpus", rewrite=None, device=None, k=None):
    """Only what the final system needs: dialect head, reranker, retriever (no complexity models)."""
    from ..retrieval.index import load_retriever
    from ..retrieval.rerank import Reranker
    from .arrouter import HFClassifier
    m = Path(models)
    dmeta = json.loads((m / "dialect_head" / "meta.json").read_text(encoding="utf-8"))
    dia = HFClassifier(m / "dialect_head", dmeta["temperature"], dmeta.get("max_len", 64), device)
    if k is None:
        cfg = Path("runs_v2") / "v2_config.json"
        k = json.loads(cfg.read_text(encoding="utf-8"))["k"] if cfg.exists() else 2
    if rewrite is None:
        raise ValueError("pass rewrite=... (an Egyptian->MSA function)")
    return FinalSystem(load_retriever(corpus, dense=True, device=device), lambda t: dia.probs(t)[:, 1],
                       Reranker("BAAI/bge-reranker-v2-m3", device).score, rewrite,
                       dmeta.get("lo", 0.2), dmeta.get("hi", 0.8), k)
