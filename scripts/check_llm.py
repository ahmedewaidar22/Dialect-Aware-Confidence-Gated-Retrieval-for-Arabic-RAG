"""Check the LLM endpoint before a long run: where calls go, one normal call, one call with token log-probs.

  python scripts/check_llm.py          # 2 tiny calls (a fraction of a cent), never cached

Uses $env:ARROUTE_LLM_BASE_URL (optional, any OpenAI-compatible URL), $env:ARROUTE_LLM_API_KEY, $env:ARROUTE_LLM_MODEL.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    import arroute.llm as L
    from arroute import free
    if free.register():
        print("free gateway: model priced at $0 (budget guard only counts tokens)")
    print("LLM calls:", L.describe(), "| model", L.DEFAULT_MODEL)
    if not L.DEFAULT_MODEL:
        raise SystemExit("set $env:ARROUTE_LLM_MODEL")
    b = L.Budget(0.05)
    ok = True
    q = "ما هي عاصمة مصر؟ أجب بكلمة واحدة."
    try:
        t, u, _ = L.complete_with_usage(q, budget=b, temperature=0, max_output_tokens=20, _ping=time.time())
        print(f"  normal call   OK: '{t}'  ({u.get('input_tokens')} in / {u.get('output_tokens')} out tokens)")
    except Exception as e:                                       # noqa: BLE001
        ok = False
        print(f"  normal call   FAILED: {type(e).__name__}: {str(e)[:300]}")
        print("  (if your URL does not support OpenAI's Responses API, set $env:ARROUTE_LLM_API = 'chat')")
    try:
        t, u, _, toks = L.complete_logprobs(q + f" ({time.time():.0f})", budget=b, max_tokens=20)
        if toks:
            print(f"  log-prob call OK: '{t}'  first token log-prob {toks[0]['lp']:.3f}, {len(toks[0]['top'])} alternatives")
        else:
            ok = False
            print(f"  log-prob call answered '{t}' but returned NO log-probs: Router v2 needs them (Phase 9)")
    except Exception as e:                                       # noqa: BLE001
        ok = False
        print(f"  log-prob call FAILED: {type(e).__name__}: {str(e)[:300]}")
    print(f"  spend ${b.spent:.5f}")
    if not ok:
        raise SystemExit(1)
    print("Endpoint OK.")


if __name__ == "__main__":
    main()
