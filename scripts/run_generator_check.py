"""Second-generator check: the frozen pipeline with another LLM (default gpt-6.1-sol), graded by the SAME judge (gpt-4.1-mini).

  python scripts/run_generator_check.py --limit 10 --out runs_sol/smoke.jsonl     # 1. cost check on 10 questions
  python scripts/run_generator_check.py --n 300                                   # 2. the run (resumes if interrupted)
  python scripts/analyze_confirmatory.py --results runs_sol/results.jsonl         # 3. the same three tests, with Sol
  python scripts/analyze_confirmatory.py --results runs_sol/mini_same_questions.jsonl   # gpt-4.1-mini on the same questions

What changes and what does not:
  * Every LLM call INSIDE the systems uses --model: the Egyptian->MSA rewrite, C1 and C2 (final, standard RAG, always C2).
  * Nothing else changes: dialect head and its band, hybrid search, reranker, k, prompts, the "I don't know" rule.
  * The judge stays gpt-4.1-mini at temperature 0 (--judge_model), so accuracies are graded the same way as before.
  * C1 of the final system is a plain call with the identical prompt (the frozen pipeline also asks for token log-
    probabilities, which the final system does not use and reasoning models may not provide).
  * Reasoning models reject temperature: --model gets {"reasoning": {"effort": --effort}, "max_output_tokens": 2000};
    a non-reasoning model can be run with --effort none (then temperature 0, 300 output tokens, as for gpt-4.1-mini).
Questions: a seeded random sample of --n from the 1,027 confirmatory questions (ids saved to <out dir>/sample_ids.txt).
Two hard caps: --budget_usd for the system model (default 5.0) and --judge_budget_usd for the judge (default 1.0).
Output records have the same format as run_confirmatory.py, so analyze_confirmatory.py works unchanged, and
mini_same_questions.jsonl holds the earlier gpt-4.1-mini results on exactly the same questions.
"""
import argparse
import json
import random
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
    ap.add_argument("--test", default="data/confirm_arabicaqa/confirm_test.jsonl")
    ap.add_argument("--mini_results", default="runs_confirm2/results.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N sampled questions")
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="low", help="reasoning effort (low/medium/high), or 'none' for a non-reasoning model")
    ap.add_argument("--judge_model", default="gpt-4.1-mini")
    ap.add_argument("--price_in", type=float, default=2.00, help="US$ per 1M input tokens of --model")
    ap.add_argument("--price_out", type=float, default=10.00, help="US$ per 1M output tokens of --model")
    ap.add_argument("--budget_usd", type=float, default=5.0)
    ap.add_argument("--judge_budget_usd", type=float, default=1.0)
    ap.add_argument("--out", default="runs_sol/results.jsonl")
    a = ap.parse_args()

    from arroute.egy.rewrite import rewrite_llm_usage
    import arroute.llm as llm
    from arroute.llm import Budget
    from arroute.llm import complete_with_usage as complete_fn
    from arroute.pipeline.final import _Routed, load_final
    from arroute.pipeline.judge import judge
    from arroute.pipeline.strategies import P_C1, fmt_passages, run_c1, run_c2
    from arroute.router.outcome import IDK
    from arroute.router.v2 import clean
    from label_outcomes import Generator

    if hasattr(llm, "describe"):
        print(llm.describe())
    prices = dict(getattr(llm, "PRICES", {}))
    prices[a.model] = (a.price_in, a.price_out)
    if a.judge_model not in prices:
        prices[a.judge_model] = (0.40, 1.60)
    pin, pout = prices[a.model]
    cost = lambda r: (r.get("input_tokens", 0) * pin + r.get("output_tokens", 0) * pout) / 1e6   # noqa: E731
    params = ({"temperature": 0, "max_output_tokens": 300} if a.effort == "none"
              else {"reasoning": {"effort": a.effort}, "max_output_tokens": 2000})
    try:
        budget, jbudget = Budget(a.budget_usd, prices=prices), Budget(a.judge_budget_usd, prices=prices)
    except TypeError:                                   # older llm.py: Budget(limit) only
        budget, jbudget = Budget(a.budget_usd), Budget(a.judge_budget_usd)
    gen = Generator(complete_fn, a.model, budget, params)
    jgen = Generator(complete_fn, a.judge_model, jbudget, {"temperature": 0, "max_output_tokens": 60})
    last = {"usage": {}}

    def rewrite(q):
        text, usage, _ = rewrite_llm_usage(q, {"fn": complete_fn, "model": a.model, "budget": budget, "params": params})
        last["usage"] = usage
        return text

    system = load_final(a.models, a.corpus, rewrite=rewrite)
    print(f"system model {a.model} ({'effort ' + a.effort if a.effort != 'none' else 'temperature 0'}) | judge {a.judge_model} "
          f"| k = {system.k} | caps: system ${a.budget_usd}, judge ${a.judge_budget_usd}")

    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    items.sort(key=lambda it: it["id"])
    random.Random(a.seed).shuffle(items)
    items = items[:a.n]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "sample_ids.txt").write_text("\n".join(it["id"] for it in items) + "\n", encoding="utf-8")
    mr = Path(a.mini_results)
    if mr.exists():
        ids = {it["id"] for it in items}
        keep = [x for x in mr.read_text(encoding="utf-8").splitlines() if x.strip() and json.loads(x)["id"] in ids]
        (out.parent / "mini_same_questions.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
    if a.limit:
        items = items[:a.limit]
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}

    def final_answer(q):
        last["usage"] = {}
        st = system.retrieve(q)
        rw = cost(last["usage"])
        text, usage, _ = gen.ask(P_C1.format(q=st["query"], passages=fmt_passages([system.r.text(p) for p in st["passages"]])))
        ans = clean(text)
        c1 = {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
        esc = IDK in (ans or "") or not (ans or "").strip()
        usd, calls, reasoning = rw + cost(c1), 1 + int(rw > 0), usage.get("reasoning_tokens", 0)
        if esc:
            c2 = run_c2(st["query"], gen, _Routed(system.r, st["query"], st["cands"]))
            ans, usd, calls = c2["answer"], usd + cost(c2), calls + c2["calls"]
            reasoning += c2.get("reasoning_tokens", 0)
        return ans, usd, calls, esc, reasoning

    t0, n_new = time.time(), 0
    with out.open("a", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            q = it["q_egy"]
            rec = {"id": it["id"], "q_msa": it["q_msa"], "refs": it["refs"], "model": a.model, "effort": a.effort,
                   "systems": {}}
            fa, fusd, fcalls, esc, freas = final_answer(q)
            sr = run_c1(q, gen, system.r, k=5)
            c2 = run_c2(q, gen, system.r)
            for name, ans, usd, calls, reas in (("final", fa, fusd, fcalls, freas),
                                                ("standard_rag", sr["answer"], cost(sr), sr["calls"], sr.get("reasoning_tokens", 0)),
                                                ("always_c2", c2["answer"], cost(c2), c2["calls"], c2.get("reasoning_tokens", 0))):
                ok, method, _ = judge(it["q_msa"], ans, it["refs"], jgen)
                rec["systems"][f"{name}|egy"] = {"answer": ans, "correct": bool(ok), "judge": method, "usd": usd,
                                                 "calls": calls, "reasoning_tokens": reas}
            rec["systems"]["final|egy"]["escalated"] = bool(esc)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 10 == 0 or n == len(items):
                per = budget.spent / n_new
                print(f"{n}/{len(items)} | {a.model} spend ${budget.spent:.3f} (~${per:.4f}/question) | judge ${jbudget.spent:.3f}"
                      f" | {time.time() - t0:.0f}s")
    if n_new:
        per = budget.spent / n_new
        print(f"\ncost per question (3 systems, {a.model}): ${per:.4f} -> {a.n} questions would cost about ${per * a.n:.2f}"
              f" (cap ${a.budget_usd}); judge ${jbudget.spent / n_new:.4f}/question")
    print(f"results in {out}")


if __name__ == "__main__":
    main()
