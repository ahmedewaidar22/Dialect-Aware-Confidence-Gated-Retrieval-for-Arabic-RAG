"""Step 3: run the FROZEN pipeline and the two baselines on the new questions. No tuning, no changes to any file.

  python scripts/freeze_manifest.py --verify freeze_manifest.json      # must say FROZEN OK first
  python scripts/run_confirmatory.py --budget_usd 3.0
  python scripts/freeze_manifest.py --verify freeze_manifest.json      # and again afterwards

For every question in data/confirm/confirm_test.jsonl (Egyptian input) it runs
  final          the frozen final system (dialect head -> rewrite -> hybrid top-20 -> rerank top-k -> C1 -> IDK -> C2)
  standard_rag   C1, hybrid top-5, question searched as typed
  always_c2      C2, question searched as typed
grades each answer with the same judge (the question in MSA + the reference), and appends one JSON line per question to
runs_confirm/results.jsonl. It resumes if interrupted (finished questions are skipped). Cost is computed from tokens at the
list price, including the rewrite call, so a rerun from cache does not report zero. --also_msa adds the MSA input.
The run stops with an error if the budget cap would be exceeded.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/confirm/confirm_test.jsonl")
    ap.add_argument("--out", default="runs_confirm/results.jsonl")
    ap.add_argument("--budget_usd", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N questions")
    ap.add_argument("--also_msa", action="store_true")
    a = ap.parse_args()

    from arroute.egy.rewrite import rewrite_llm_usage
    from arroute.llm import Budget
    from arroute.llm import complete_logprobs as lp_fn
    from arroute.llm import complete_with_usage as complete_fn
    from arroute.pipeline.final import load_final
    from arroute.pipeline.judge import judge
    from arroute.pipeline.strategies import run_c1, run_c2
    from arroute.router.v2 import tok_cost
    from label_outcomes import Generator

    model = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    budget = Budget(a.budget_usd)
    print("LLM model:", model, "| budget cap $", a.budget_usd)
    gen_rw = {"fn": complete_fn, "model": model, "budget": budget}
    last = {"usage": {}}

    def rewrite(q):
        text, usage, _ = rewrite_llm_usage(q, gen_rw)
        last["usage"] = usage
        return text

    system = load_final(a.models, a.corpus, rewrite=rewrite)
    gen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 300})
    jgen = Generator(complete_fn, model, budget, {"temperature": 0, "max_output_tokens": 60})
    ask = lambda p: lp_fn(p, model=model, budget=budget)                       # noqa: E731
    print(f"final system loaded (k = {system.k})")

    items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    inputs = [("egy", "q_egy")] + ([("msa", "q_msa")] if a.also_msa else [])
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            rec = {"id": it["id"], "q_msa": it["q_msa"], "refs": it["refs"], "systems": {}}
            for lang, field in inputs:
                q = it[field]
                last["usage"] = {}
                st = system.answer(q, ask, gen)
                rw_cost = tok_cost(last["usage"])
                sr = run_c1(q, gen, system.r, k=5)
                c2 = run_c2(q, gen, system.r)
                for name, ans, cost, calls in (
                        ("final", st["answer"], tok_cost(st) + rw_cost, st["calls"] + (1 if rw_cost else 0)),
                        ("standard_rag", sr["answer"], tok_cost(sr), sr["calls"]),
                        ("always_c2", c2["answer"], tok_cost(c2), c2["calls"])):
                    ok, method, _ = judge(it["q_msa"], ans, it["refs"], jgen)
                    rec["systems"][f"{name}|{lang}"] = {"answer": ans, "correct": bool(ok), "judge": method,
                                                         "usd": cost, "calls": calls}
                rec["systems"][f"final|{lang}"]["escalated"] = bool(st["escalated"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{n}/{len(items)} questions | spend ${budget.spent:.3f} | {time.time() - t0:.0f}s")
    print(f"done. spend ${budget.spent:.3f} | results in {out}")


if __name__ == "__main__":
    main()
