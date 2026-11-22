"""Pick ONE model on a free gateway (freellmAPI) for the whole run and save it to free_model.txt.

  python scripts/pick_free_model.py                 # list the gateway's models, test up to 12 candidates (2 tiny calls each)
  python scripts/pick_free_model.py --max 30 --match "llama|qwen|gemini|mistral|gpt"
  python scripts/pick_free_model.py --model some-model-id      # test just this one

A candidate must (1) answer a tiny Arabic question sensibly, (2) return token log-probs (Router v2 needs them).
The first one that passes is written to free_model.txt; run_phase9.ps1 then uses it as ARROUTE_LLM_MODEL.
Never use 'auto': the gateway would switch models between calls and the results would mix several models.
Uses $env:ARROUTE_LLM_BASE_URL (e.g. http://localhost:3001/v1) and $env:ARROUTE_LLM_API_KEY (freellmapi-...).
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

SKIP = re.compile(r"embed|whisper|tts|image|video|audio|rerank|moderat|guard|ocr|diffus|flux|speech|transcri", re.I)
REASONING = re.compile(r"(^|[-_/:])(r1|o1|o3|o4|qwq)|reason|think|gemini-(2\.5|[3-9])", re.I)   # Gemini 2.5+ think by default   # they spend the tiny token limits on hidden thinking
ANSWER = "ما هي عاصمة مصر؟ أجب بكلمة واحدة."


def _client():
    import os
    from openai import OpenAI
    key = (os.environ.get("ARROUTE_LLM_API_KEY") or "").strip()          # never fall back to an OpenAI key here
    url = (os.environ.get("ARROUTE_LLM_BASE_URL") or "").strip() or None
    if not key:
        raise SystemExit("set $env:ARROUTE_LLM_API_KEY first")
    if not url:
        raise SystemExit("set $env:ARROUTE_LLM_BASE_URL first (e.g. http://localhost:3001/v1)")
    if not key.startswith("freellmapi-"):
        print(f"WARNING: the key starts with '{key[:6]}...', but freellmAPI keys start with 'freellmapi-'. "
              "Copy the unified key from the gateway's Keys page.")
    return OpenAI(api_key=key, base_url=url, max_retries=2, timeout=90), url


def test_model(client, model):
    """Standalone (does not use arroute/llm.py): 1 normal call + 1 call with log-probs."""
    res = {"model": model, "ok": False, "lp": False, "note": ""}
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": ANSWER}],
                                           temperature=0, max_tokens=60)
        t = (r.choices[0].message.content or "").strip()
        if not t:
            res["note"] = "EMPTY completion (hidden thinking used up the 60-token limit the judge step uses)"
            return res
        if "القاهرة" not in t and "Cairo" not in t:
            res["note"] = f"odd answer: {t[:40]!r}"
            return res
        res["ok"] = True
    except Exception as e:                                                      # noqa: BLE001
        res["note"] = f"call failed: {type(e).__name__}: {str(e)[:90]}"
        return res
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": ANSWER + f" ({time.time():.0f})"}],
                                           temperature=0, max_tokens=60, logprobs=True, top_logprobs=5)
        c = getattr(getattr(r.choices[0], "logprobs", None), "content", None) or []
        res["lp"] = bool(c) and bool(c[0].top_logprobs)
        res["note"] = "ok" if res["lp"] else "no log-probs returned"
    except Exception as e:                                                      # noqa: BLE001
        res["note"] = f"log-prob call failed: {type(e).__name__}: {str(e)[:90]}"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="test only this model id")
    ap.add_argument("--match", default=None, help="regex: only consider model ids that match")
    ap.add_argument("--max", type=int, default=12, help="how many candidates to test")
    ap.add_argument("--allow_reasoning", action="store_true")
    ap.add_argument("--out", default="free_model.txt")
    a = ap.parse_args(argv)

    client, url = _client()
    print("gateway:", url)
    if a.model:
        cands = [a.model]
    else:
        ids = [m.id for m in client.models.list().data]
        print(f"gateway lists {len(ids)} models")
        cands = [m for m in ids if m.lower() != "auto" and not m.lower().startswith("auto:")
                 and not SKIP.search(m) and (a.allow_reasoning or not REASONING.search(m))
                 and (not a.match or re.search(a.match, m, re.I))]
        print(f"{len(cands)} look usable; testing the first {min(a.max, len(cands))}")
        cands = cands[:a.max]
    chosen = None
    for m in cands:
        r = test_model(client, m)
        flag = "PASS" if r["ok"] and r["lp"] else "fail"
        print(f"  {flag}  {m:60s} {r['note']}")
        if flag == "PASS":
            chosen = m
            break
    if not chosen:
        raise SystemExit("\nNo candidate passed. Try --match / a larger --max, or a gateway model that returns log-probs "
                         "(Router v2's confidence features need them).")
    Path(a.out).write_text(chosen + "\n", encoding="utf-8")
    print(f"\nChosen: {chosen}  -> saved to {a.out}. run_phase9.ps1 will use it.")


if __name__ == "__main__":
    main()