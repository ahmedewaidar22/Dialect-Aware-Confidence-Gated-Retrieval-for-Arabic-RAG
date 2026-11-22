"""Phase 7 smoke test: load the FULL saved router and route real questions end to end.

  python scripts/router_demo.py --n 6                  # route 6 Egyptian + 6 MSA test questions (rewrites: LLM, cached)
  python scripts/router_demo.py --n 6 --answer         # also answer them with the LLM (a few cents, hard cap)
  python scripts/router_demo.py --question "مين اللي بنى السد العالي؟" --answer

Prints, per question: P(Egyptian) and the dialect decision, the search query, P(C0), P(C2 needed), the chosen
strategy, timing, and (with --answer) the answer, whether C1 escalated, and the judge's verdict.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--question", default=None)
    ap.add_argument("--mode", default="quality", choices=["budget", "match_c1", "balanced", "quality"])
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--budget_usd", type=float, default=0.2)
    ap.add_argument("--out", default="runs_router7/demo.jsonl")
    a = ap.parse_args()

    from arroute.egy.rewrite import rewrite_llm
    from arroute.llm import Budget, complete_with_usage
    from arroute.pipeline.arrouter import load_router
    from label_outcomes import Generator, parse_prices
    model = os.environ.get("ARROUTE_LLM_MODEL")
    if not model:
        raise SystemExit("set ARROUTE_LLM_MODEL (e.g. gpt-4.1-mini)")
    budget = Budget(a.budget_usd, parse_prices(["gpt-4.1-mini=0.40,1.60"]))
    budget.price(model)
    gen_rw = {"fn": complete_with_usage, "model": model, "budget": budget}
    t0 = time.time()
    router = load_router(a.models, a.corpus, rewrite=lambda q: rewrite_llm(q, gen_rw), mode=a.mode)
    print(f"router loaded in {time.time() - t0:.0f}s (mode: {a.mode})\n")
    gen = Generator(complete_with_usage, model, budget, {"temperature": 0, "max_output_tokens": 300})
    jgen = Generator(complete_with_usage, model, budget, {"temperature": 0, "max_output_tokens": 60})

    if a.question:
        todo = [{"q": a.question, "refs": None, "label": "your question"}]
    else:
        items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
        pick = [it for s in ("heldout", "ardqa", "pilot") for it in [i for i in items if i["set"] == s][:max(1, a.n // 3)]][:a.n]
        todo = [{"q": it[k], "refs": it["refs"], "label": f"{it['set']} / {v}"} for it in pick
                for k, v in (("q_egy", "Egyptian"), ("q_msa", "MSA"))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from arroute.pipeline.judge import judge
    with out.open("w", encoding="utf-8") as f:
        for t in todo:
            t1 = time.time()
            st = router.answer(t["q"], gen) if a.answer else router.route(t["q"])
            dt = time.time() - t1
            line = (f"[{t['label']}] {t['q']}\n   P(Egyptian) {st['p_egy']:.2f} -> {st['dialect']:<6} | search: {st['search']}\n"
                    f"   P(C0) {st['p_c0']:.2f} | P(C2 needed) {st.get('p_need_c2', float('nan')):.2f} -> {st['strategy']}")
            rec = {k: v for k, v in st.items() if k not in ("cands", "rr20", "x_pre", "result")}
            if a.answer:
                r = st["result"]
                ok = None
                if t["refs"]:
                    ok, _, _ = judge(t["q"], r["answer"], t["refs"], jgen)
                line += (f"{' -> escalated to C2' if st.get('escalate') else ''}\n   answer: {r['answer']}"
                         + (f"   | correct: {ok}" if ok is not None else "") + f"   | LLM calls {r['calls']}")
                rec.update(answer=r["answer"], correct=ok, calls=r["calls"])
            print(line + f"   | {dt:.1f}s\n")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"spend ${budget.spent:.4f} | saved {out}")


if __name__ == "__main__":
    main()
