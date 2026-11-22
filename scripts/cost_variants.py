"""Development check of cheaper variants of the final system (NOT a test: uses calibration questions, not test questions).

  python scripts/cost_variants.py --n 50 --budget_usd 3.0

Questions: a seeded sample of --n native Egyptian ArDQA CALIBRATION questions (in egy_test.jsonl but not among the
1,000 test questions of runs_final/per_question.csv). They were already used to choose k, so they are development data.
The 1,000 test questions and the 1,027 confirmatory questions are never touched.

Generator --model (default gpt-6.1-sol, reasoning effort --effort); judge gpt-4.1-mini, as in the earlier runs.
Variants, each run on the same questions:
  V0  frozen final system, every call with --model                         (reference)
  V1  rewrite with gpt-4.1-mini (frozen prompt and settings), rest --model   (option 1)
  V2  V1 + C1 and C2 at reasoning effort --cheap_effort                     (option 2; skipped if the API rejects it)
  V3  V1 + lighter C2 fallback: 2 searches, at most 6 passages              (option 3)
  STD standard RAG with --model                                              (reference baseline)
Reports per variant: accuracy (with a 95% interval; 50 questions is a COST check, accuracy differences are rough),
generator cost per 1,000 questions split into rewrite / C1 / C2, LLM calls, reasoning tokens, escalation rate, and how
many answers changed vs V0. Writes runs_costvar/results.jsonl (resumes) and runs_costvar/report.txt.
"""
import argparse
import csv
import json
import random
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--per_question", default="runs_final/per_question.csv")
    ap.add_argument("--n", type=int, default=50, help="number of questions; 0 = all")
    ap.add_argument("--set_file", default="", help="evaluate on this question file instead of calibration questions, "
                    "e.g. data/confirm_arabicaqa/confirm_test.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--cheap_effort", default="minimal")
    ap.add_argument("--price_in", type=float, default=2.00)
    ap.add_argument("--price_out", type=float, default=10.00)
    ap.add_argument("--cheap_model", default="gpt-4.1-mini")
    ap.add_argument("--budget_usd", type=float, default=3.0)
    ap.add_argument("--out", default="runs_costvar")
    a = ap.parse_args()

    import arroute.llm as llm
    from arroute.egy.rewrite import rewrite_llm_usage
    from arroute.llm import Budget
    from arroute.llm import complete_with_usage as complete_fn
    from arroute.pipeline.final import _Routed, load_final
    from arroute.pipeline.judge import judge
    from arroute.pipeline.strategies import P_C1, fmt_passages, run_c1, run_c2
    from arroute.router.outcome import IDK
    from arroute.router.v2 import clean
    from label_outcomes import Generator

    prices = dict(getattr(llm, "PRICES", {}))
    prices[a.model] = (a.price_in, a.price_out)
    prices.setdefault(a.cheap_model, (0.40, 1.60))
    prices.setdefault("gpt-4.1-mini", (0.40, 1.60))
    try:
        budget = Budget(a.budget_usd, prices=prices)
    except TypeError:
        budget = Budget(a.budget_usd)
    usd = lambda model, r: (r.get("input_tokens", 0) * prices[model][0] + r.get("output_tokens", 0) * prices[model][1]) / 1e6  # noqa: E731
    big = lambda effort: ({"temperature": 0, "max_output_tokens": 300} if effort == "none"   # noqa: E731
                          else {"reasoning": {"effort": effort}, "max_output_tokens": 2000})
    gen_main = Generator(complete_fn, a.model, budget, big(a.effort))
    gen_cheapeffort = Generator(complete_fn, a.model, budget, big(a.cheap_effort))
    jgen = Generator(complete_fn, "gpt-4.1-mini", budget, {"temperature": 0, "max_output_tokens": 60})
    rw_main = {"fn": complete_fn, "model": a.model, "budget": budget, "params": big(a.effort)}
    rw_cheap = {"fn": complete_fn, "model": a.cheap_model, "budget": budget}            # frozen mini rewrite settings
    last = {"usage": {}, "model": None, "rw": None}

    def rewrite(q):
        text, usage, _ = rewrite_llm_usage(q, last["rw"])
        last["usage"], last["model"] = usage, last["rw"]["model"]
        return text

    system = load_final(a.models, a.corpus, rewrite=rewrite)

    variants = {"V0": (rw_main, gen_main, {}), "V1": (rw_cheap, gen_main, {}),
                "V2": (rw_cheap, gen_cheapeffort, {}), "V3": (rw_cheap, gen_main, {"max_searches": 2, "max_passages": 6})}
    try:                                                    # does the API accept --cheap_effort?
        gen_cheapeffort.ask("Reply with the word: ok")
    except Exception as e:                                  # noqa: BLE001
        print(f"V2 skipped: {a.model} rejected reasoning effort '{a.cheap_effort}' ({type(e).__name__}: {str(e)[:120]})")
        variants.pop("V2")

    def final(q, rw, gen, c2kw):
        last["rw"], last["usage"], last["model"] = rw, {}, None
        st = system.retrieve(q)
        r_usd = usd(last["model"], last["usage"]) if last["model"] else 0.0
        text, u, _ = gen.ask(P_C1.format(q=st["query"], passages=fmt_passages([system.r.text(p) for p in st["passages"]])))
        ans = clean(text)
        c1_usd = usd(a.model, u)
        reas = u.get("reasoning_tokens", 0)
        esc = IDK in (ans or "") or not (ans or "").strip()
        c2_usd, calls = 0.0, 1 + int(bool(last["model"]))
        if esc:
            c2 = run_c2(st["query"], gen, _Routed(system.r, st["query"], st["cands"]), **c2kw)
            ans, c2_usd, calls, reas = c2["answer"], usd(a.model, c2), calls + c2["calls"], reas + c2.get("reasoning_tokens", 0)
        return {"answer": ans, "rewrite_usd": r_usd, "c1_usd": c1_usd, "c2_usd": c2_usd, "calls": calls,
                "reasoning_tokens": reas, "escalated": bool(esc)}

    if a.set_file:
        calib = [json.loads(x) for x in Path(a.set_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        calib.sort(key=lambda it: it["id"])
        label = f"questions from {a.set_file}"
    else:
        items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
        with open(a.per_question, encoding="utf-8-sig", newline="") as f:
            test_ids = {r["id"] for r in csv.DictReader(f)}
        calib = sorted((it for it in items if it["set"] == "ardqa" and it["id"] not in test_ids), key=lambda it: it["id"])
        label = "calibration questions (none from the test or confirmatory sets)"
    random.Random(a.seed).shuffle(calib)
    if a.n:
        calib = calib[:a.n]
    print(f"{len(calib)} {label} | variants {list(variants)} + STD")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rf = out / "results.jsonl"
    done = {json.loads(x)["id"]: json.loads(x) for x in rf.read_text(encoding="utf-8").splitlines() if x.strip()} if rf.exists() else {}
    with rf.open("a", encoding="utf-8") as f:
        for i, it in enumerate(calib, 1):
            if it["id"] in done:
                continue
            rec = {"id": it["id"], "res": {}}
            for name, (rw, gen, c2kw) in variants.items():
                rec["res"][name] = final(it["q_egy"], rw, gen, c2kw)
            sr = run_c1(it["q_egy"], gen_main, system.r, k=5)
            rec["res"]["STD"] = {"answer": sr["answer"], "rewrite_usd": 0.0, "c1_usd": usd(a.model, sr), "c2_usd": 0.0,
                                 "calls": sr["calls"], "reasoning_tokens": sr.get("reasoning_tokens", 0), "escalated": False}
            for name, r in rec["res"].items():
                ok, _, _ = judge(it["q_msa"], r["answer"], it["refs"], jgen)
                r["correct"] = bool(ok)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[it["id"]] = rec
            if i % 10 == 0:
                print(f"  {i}/{len(calib)} | spend ${budget.spent:.3f}")

    R = [done[it["id"]] for it in calib if it["id"] in done]
    n = len(R)
    names = [k for k in ("V0", "V1", "V2", "V3", "STD") if all(k in r["res"] for r in R)]
    L = [f"COST VARIANTS ({n} {label}, generator {a.model}, judge gpt-4.1-mini)", "",
         f"{'variant':8s}{'acc %':>7s}{'95% CI':>15s}{'US$/1k':>9s}{'rewrite':>9s}{'C1':>8s}{'C2':>8s}{'calls':>7s}{'reas.tok':>9s}{'esc %':>7s}{'changed':>9s}"]
    for k in names:
        rs = [r["res"][k] for r in R]
        c = sum(x["correct"] for x in rs)
        lo, hi = wilson(c, n)
        m = lambda f: 1000 * sum(x[f] for x in rs) / n                     # noqa: E731
        tot = m("rewrite_usd") + m("c1_usd") + m("c2_usd")
        changed = sum(r["res"][k]["correct"] != r["res"]["V0"]["correct"] for r in R) if k != "V0" else 0
        L.append(f"{k:8s}{100 * c / n:7.1f}{f'[{100 * lo:.0f}, {100 * hi:.0f}]':>15s}{tot:9.3f}{m('rewrite_usd'):9.3f}{m('c1_usd'):8.3f}"
                 f"{m('c2_usd'):8.3f}{sum(x['calls'] for x in rs) / n:7.2f}{sum(x['reasoning_tokens'] for x in rs) / n:9.1f}"
                 f"{100 * sum(x['escalated'] for x in rs) / n:7.1f}{changed:9d}")
    from math import comb
    mc = lambda b, c: 1.0 if b + c == 0 else min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))  # noqa: E731
    L.append("")
    for k in names:
        for base in ("V0", "STD"):
            if k == base or base not in names:
                continue
            b = sum(r["res"][k]["correct"] and not r["res"][base]["correct"] for r in R)
            c = sum(r["res"][base]["correct"] and not r["res"][k]["correct"] for r in R)
            L.append(f"  {k} vs {base}: {100 * (b - c) / n:+.1f} points (only {k} right {b}, only {base} right {c}), McNemar p = {mc(b, c):.4f}")
    L += ["", "changed = questions whose verdict (right/wrong) differs from V0. With 50 questions, read accuracy as a sanity",
          "check only; a variant worth keeping must be frozen and tested on NEW questions."]
    text = "\n".join(L)
    print(text)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"spend ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
