"""Phase 10: run the FINAL system end to end on test questions or on your own question.

  python scripts/final_demo.py --n 6                                   # 6 Egyptian + 6 MSA test questions, judged
  python scripts/final_demo.py --question "مين اللي بنى السد العالي؟"

Prints per question: P(Egyptian) and the dialect decision, the search query, the 2 passages the reranker chose,
C1's answer, whether it escalated to C2 ("I don't know"), the final answer, cost, time, and (test questions)
the judge's verdict. Hard budget cap (default $0.20); answers already computed in Phases 8-9 come from the cache.
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


def main(argv=None, system=None, lp_fn=None, complete_fn=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--question", default=None)
    ap.add_argument("--budget_usd", type=float, default=0.2)
    ap.add_argument("--out", default="runs_final/demo.jsonl")
    a = ap.parse_args(argv)

    from arroute.egy.rewrite import rewrite_llm
    from arroute.llm import Budget
    from arroute.pipeline.judge import judge
    from arroute.router.v2 import tok_cost
    from label_outcomes import Generator
    if lp_fn is None:
        from arroute.llm import complete_logprobs as lp_fn
    if complete_fn is None:
        from arroute.llm import complete_with_usage as complete_fn
    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    budget = Budget(a.budget_usd)
    print("LLM model:", model)
    gen_rw = {"fn": complete_fn, "model": model, "budget": budget}
    t0 = time.time()
    if system is None:
        from arroute.pipeline.final import load_final
        system = load_final(a.models, a.corpus, rewrite=lambda q: rewrite_llm(q, gen_rw))
    else:
        system.rewrite = lambda q: rewrite_llm(q, gen_rw)
    print(f"final system loaded in {time.time() - t0:.0f}s (k = {system.k} reranked passages)\n")
    ask = lambda p: lp_fn(p, model=model, budget=budget)                       # noqa: E731
    gen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 300})
    jgen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 60})
    if a.question:
        todo = [{"q": a.question, "refs": None, "judge_q": a.question, "label": "your question"}]
    else:
        items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
        pick = [it for s in ("ardqa", "pilot", "heldout") for it in [i for i in items if i["set"] == s][:max(1, a.n // 3)]][:a.n]
        todo = [{"q": it[k], "refs": it["refs"], "judge_q": it["q_msa"], "label": f"{it['set']} / {v}"} for it in pick
                for k, v in (("q_egy", "Egyptian"), ("q_msa", "MSA"))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_judged = 0
    with out.open("w", encoding="utf-8") as f:
        for t in todo:
            t1 = time.time()
            st = system.answer(t["q"], ask, gen)
            dt = time.time() - t1
            ok = None
            if t["refs"]:
                ok, _, _ = judge(t["judge_q"], st["answer"], t["refs"], jgen)
                n_ok, n_judged = n_ok + bool(ok), n_judged + 1
            cost = tok_cost(st)
            print(f"[{t['label']}] {t['q']}\n   P(Egyptian) {st['p_egy']:.2f} -> {st['dialect']:<6} | search: {st['query']}\n"
                  f"   passages (reranked top-{system.k}): {st['passages']}\n   C1: {st['c1']['answer']}"
                  + ("   -> 'I don't know': escalated to C2" if st["escalated"] else "")
                  + f"\n   ANSWER: {st['answer']}" + (f"   | correct: {ok}" if ok is not None else "")
                  + f"   | LLM calls {st['calls']} | ${cost:.5f} | {dt:.1f}s\n")
            f.write(json.dumps({"question": t["q"], "label": t["label"], "p_egy": st["p_egy"], "dialect": st["dialect"],
                                "query": st["query"], "passages": st["passages"], "c1": st["c1"]["answer"],
                                "escalated": st["escalated"], "answer": st["answer"], "correct": ok,
                                "calls": st["calls"], "usd": cost, "seconds": dt}, ensure_ascii=False) + "\n")
    if n_judged:
        print(f"correct: {n_ok}/{n_judged}")
    print(f"spend ${budget.spent:.4f} | saved {out}")


if __name__ == "__main__":
    main()
