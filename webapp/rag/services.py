"""Singleton wrapper around arroute.pipeline.final.FinalSystem (same wiring as scripts/final_demo.py)."""
import threading
import time

from django.conf import settings

_lock = threading.Lock()          # the GPU models and the BM25 index are shared: one question at a time
_state = {"system": None, "error": None}


class _Gen:
    def __init__(self, fn, model, budget, params):
        self.fn, self.model, self.budget, self.params = fn, model, budget, params

    def ask(self, prompt):
        return self.fn(prompt, model=self.model, budget=self.budget, **self.params)


def load():
    """Load the models once. Safe to call repeatedly."""
    with _lock:
        if _state["system"] is not None:
            return _state["system"]
        from arroute.egy.rewrite import rewrite_llm
        from arroute.llm import Budget, complete_logprobs, complete_with_usage
        from arroute.pipeline.final import load_final
        model, budget = settings.LLM_MODEL, Budget(settings.BUDGET_USD)
        gen_rw = {"fn": complete_with_usage, "model": model, "budget": budget}
        system = load_final(settings.MODELS_DIR, settings.CORPUS_DIR, rewrite=lambda q: rewrite_llm(q, gen_rw))
        system._web = {
            "ask": lambda p: complete_logprobs(p, model=model, budget=budget),
            "gen": _Gen(complete_with_usage, model, budget, {"temperature": 0, "max_output_tokens": 300}),
            "budget": budget,
        }
        _state["system"] = system
        return system


def ready():
    return _state["system"] is not None


def answer_question(question):
    from arroute.router.v2 import tok_cost
    system = load()
    w = system._web
    with _lock:
        t0 = time.time()
        st = system.answer(question, w["ask"], w["gen"])
        dt = time.time() - t0
        passages = [{"id": p, "text": system.r.text(p)} for p in st["passages"]]
    return {
        "answer": st["answer"], "dialect": st["dialect"], "p_egy": st["p_egy"], "query": st["query"],
        "passages": passages, "c1": st["c1"]["answer"], "escalated": st["escalated"],
        "calls": st["calls"], "usd": tok_cost(st), "seconds": round(dt, 2),
        "budget_spent": w["budget"].spent,
    }
