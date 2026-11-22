"""Use a FREE OpenAI-compatible gateway (freellmAPI, http://localhost:3001/v1) without touching llm.py.

register() gives the model in $env:ARROUTE_LLM_MODEL a price of $0 / $0 per 1M tokens so the budget guard accepts it
(llm.Budget refuses a model that has no price). It only acts when a non-OpenAI URL is set, or ARROUTE_FREE=1, and
never overrides a price that already exists (gpt-4.1-mini keeps its real price).
"""
import os


def is_free_endpoint():
    url = (os.environ.get("ARROUTE_LLM_BASE_URL") or "").strip().lower()
    if os.environ.get("ARROUTE_FREE", "").strip() in ("1", "true", "yes"):
        return True
    return bool(url) and "api.openai.com" not in url


def register(model=None):
    """-> True if a $0 price was registered for the model."""
    from arroute import llm
    model = model or os.environ.get("ARROUTE_LLM_MODEL")
    if not model or not is_free_endpoint():
        return False
    if llm.price_name(model) in llm.PRICES:
        return False
    llm.PRICES[model] = (0.0, 0.0)
    return True
