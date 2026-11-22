"""Cached LLM client (OpenAI Responses API) with token accounting and a HARD budget guard.

- Every successful call is cached on disk, keyed by (model, prompt, params): re-runs cost nothing.
- complete_with_usage() returns (text, usage, was_cached); complete() returns text only (old scripts).
- Params whose name starts with '_' (e.g. _repeat=1) only change the cache key and are never sent to the
  API: that is how a repeat run gets fresh answers to identical prompts.
- A Budget reserves the worst-case cost of a call BEFORE making it (prompt estimate + max_output_tokens)
  and refuses the call if that could cross the limit, so the limit is never exceeded, even with
  parallel workers. Actual costs (from the API's token counts) are written to data/llm_cache/ledger.jsonl.

Set:  PowerShell: $env:OPENAI_API_KEY="sk-..."   $env:ARROUTE_LLM_MODEL="<exact model ID>"
"""
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path

CACHE_DIR = Path(os.environ.get("ARROUTE_LLM_CACHE", "data/llm_cache"))
DEFAULT_MODEL = os.environ.get("ARROUTE_LLM_MODEL")

# USD per 1M tokens (input, output). gpt-6.1-sol: OpenAI list price, checked Oct 2026.
# gpt-4.1-mini: list price at its 2025 launch; verify on OpenAI's pricing page (override with --price).
PRICES = {"gpt-6.1-sol": (2.00, 10.00), "gpt-4.1-mini": (0.40, 1.60)}

RATE_LIMIT_RETRIES = 8         # extra patient retries on HTTP 429 (about 6 minutes in total)
_client = None                 # OpenAI (Responses API)
_local_clients = {}            # base_url -> OpenAI-compatible client (Ollama, LM Studio: Chat Completions API)
_ledger_lock = threading.Lock()
_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)    # reasoning blocks some local models emit


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, limit_usd, prices=None):
        self.limit = float(limit_usd)
        self.prices = {**PRICES, **(prices or {})}
        self.spent = 0.0            # includes reservations of calls still in flight
        self.calls = 0
        self.lock = threading.Lock()

    def price(self, model):
        if model not in self.prices:
            raise ValueError(f"no price for model '{model}': pass --price {model}=INPUT,OUTPUT (USD per 1M tokens)")
        return self.prices[model]

    def reserve(self, model, prompt, max_output_tokens):
        pin, pout = self.price(model)
        worst = (len(prompt) / 2) * pin / 1e6 + (max_output_tokens or 4000) * pout / 1e6   # ~2 chars/token: pessimistic
        with self.lock:
            if self.spent + worst > self.limit:
                raise BudgetExceeded(f"budget ${self.limit:.2f} reached (committed ${self.spent:.4f})")
            self.spent += worst
        return worst

    def settle(self, model, reserved, usage):
        pin, pout = self.price(model)
        actual = usage.get("input_tokens", 0) * pin / 1e6 + usage.get("output_tokens", 0) * pout / 1e6
        with self.lock:
            self.spent += actual - reserved
            self.calls += 1 if usage else 0
        return actual


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(max_retries=6, timeout=180)   # backoff on rate limits during parallel runs
    return _client


def _local_client(base_url):
    if base_url not in _local_clients:
        from openai import OpenAI
        _local_clients[base_url] = OpenAI(base_url=base_url, api_key=os.environ.get("ARROUTE_LOCAL_API_KEY", "local"),
                                          max_retries=2, timeout=600)
    return _local_clients[base_url]


def _call_local(base_url, model, prompt, params):
    """Local servers speak the Chat Completions API. Map our params; strip <think> blocks."""
    kw = {}
    if "max_output_tokens" in params:
        kw["max_tokens"] = params["max_output_tokens"]
    if "temperature" in params:
        kw["temperature"] = params["temperature"]
    resp = _local_client(base_url).chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **kw)
    choice = resp.choices[0]
    out = _THINK.sub("", choice.message.content or "").strip()
    u = getattr(resp, "usage", None)
    usage = {"input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
             "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
             "cached_tokens": 0, "reasoning_tokens": 0}
    return out, usage, choice.finish_reason == "length"


def _key(model, prompt, params):
    raw = json.dumps({"m": model, "p": prompt, "x": params}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    det_in = getattr(u, "input_tokens_details", None)
    det_out = getattr(u, "output_tokens_details", None)
    return {"input_tokens": int(getattr(u, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(det_in, "cached_tokens", 0) or 0) if det_in else 0,
            "reasoning_tokens": int(getattr(det_out, "reasoning_tokens", 0) or 0) if det_out else 0}


def complete_with_usage(prompt, model=None, budget=None, base_url=None, **params):
    """-> (text, usage dict, was_cached). Raises BudgetExceeded instead of overspending.
    base_url: an OpenAI-compatible LOCAL server (e.g. Ollama http://localhost:11434/v1); None = OpenAI."""
    model = model or DEFAULT_MODEL
    if not model:
        raise ValueError("Set ARROUTE_LLM_MODEL or pass model=...")
    safe_model = re.sub(r'[<>:"/\\|?*]', "_", model)          # Windows-illegal path chars
    key_params = {**params, "_base_url": base_url} if base_url else params
    path = CACHE_DIR / safe_model / f"{_key(model, prompt, key_params)}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["output"], d.get("usage", {}), True
    params = {k: v for k, v in params.items() if not k.startswith("_")}   # '_x' params: cache key only
    reserved = budget.reserve(model, prompt, params.get("max_output_tokens")) if budget else 0.0
    try:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                if base_url:
                    out, usage, incomplete = _call_local(base_url, model, prompt, params)
                else:
                    resp = _get_client().responses.create(model=model, input=prompt, **params)
                    usage, out = _usage(resp), resp.output_text or ""
                    incomplete = getattr(resp, "status", None) == "incomplete"
                break
            except Exception as e:                                # noqa: BLE001
                # tokens-per-minute limits reset each minute: wait (5, 10, 20, 40, 60, 60 ... s) and retry
                if type(e).__name__ != "RateLimitError" or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt) + random.uniform(0, 3))
    except Exception:
        if budget:
            budget.settle(model, reserved, {})                    # release the reservation
        raise
    usage["incomplete"] = incomplete
    cost = budget.settle(model, reserved, usage) if budget else None
    with _ledger_lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with (CACHE_DIR / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, **usage,
                                "cost_usd": cost}) + "\n")
    if not incomplete:                                            # incomplete answers are retried next time
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "params": key_params, "prompt": prompt, "output": out,
                                    "usage": usage}, ensure_ascii=False), encoding="utf-8")
    return out, usage, False


def complete_logprobs(prompt, model=None, budget=None, top_logprobs=5, max_tokens=300, temperature=0):
    """Router v2: one call through the Chat Completions API WITH token log-probabilities (the Responses API
    path above stays unchanged). -> (text, usage, was_cached, tokens); tokens = [{"t", "lp", "top": [lp, ...]}].
    Cached on disk like every other call (own cache key), and guarded by the same hard budget."""
    model = model or DEFAULT_MODEL
    if not model:
        raise ValueError("Set ARROUTE_LLM_MODEL or pass model=...")
    safe_model = re.sub(r'[<>:"/\\|?*]', "_", model)
    kp = {"_api": "chat_logprobs", "top_logprobs": top_logprobs, "max_tokens": max_tokens, "temperature": temperature}
    path = CACHE_DIR / safe_model / f"{_key(model, prompt, kp)}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["output"], d.get("usage", {}), True, d.get("tokens", [])
    reserved = budget.reserve(model, prompt, max_tokens) if budget else 0.0
    try:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                resp = _get_client().chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
                    max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs)
                break
            except Exception as e:                                # noqa: BLE001
                if type(e).__name__ != "RateLimitError" or attempt == RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt) + random.uniform(0, 3))
    except Exception:
        if budget:
            budget.settle(model, reserved, {})
        raise
    ch = resp.choices[0]
    out = ch.message.content or ""
    u = getattr(resp, "usage", None)
    usage = {"input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
             "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
             "incomplete": ch.finish_reason == "length"}
    content = getattr(getattr(ch, "logprobs", None), "content", None) or []
    tokens = [{"t": c.token, "lp": float(c.logprob), "top": [float(x.logprob) for x in (c.top_logprobs or [])]}
              for c in content]
    cost = budget.settle(model, reserved, usage) if budget else None
    with _ledger_lock:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with (CACHE_DIR / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, **usage,
                                "cost_usd": cost, "api": "chat_logprobs"}) + "\n")
    if not usage["incomplete"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": model, "params": kp, "prompt": prompt, "output": out, "usage": usage,
                                    "tokens": tokens}, ensure_ascii=False), encoding="utf-8")
    return out, usage, False, tokens


def complete(prompt, model=None, **params):
    """Text-only wrapper kept for the translation scripts."""
    return complete_with_usage(prompt, model=model, **params)[0]


def list_models():
    """Print model IDs your key can access, so you pin the exact ID in your thesis."""
    for m in sorted(x.id for x in _get_client().models.list().data):
        print(m)


if __name__ == "__main__":
    list_models()
