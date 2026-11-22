"""Phase 3b: OUTCOME labels. For each question run C0 -> C1 -> C2 (cheapest first) and label it with the
cheapest strategy whose answer the judge accepts ('none' if none does). --mode full runs all three.

  python scripts/label_outcomes.py --run trial_mini --n_per_source 10 --budget_usd 1
  python scripts/label_outcomes.py --run main_mini --budget_usd 10
  python scripts/label_outcomes.py --run sens_sol --gen_model gpt-6.1-sol --effort low --n_per_source 100 --budget_usd 5

Writes data/labels/<run>/labels.jsonl (one line per question, appended as they finish: a re-run resumes)
and summary.txt. The budget is a hard cap for THIS command; LLM answers are cached, so a re-run never
pays twice for the same call.
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.data import read_jsonl                               # noqa: E402
from arroute.llm import Budget, BudgetExceeded                    # noqa: E402
from arroute.pipeline.judge import judge                          # noqa: E402
from arroute.pipeline.strategies import STRATEGIES                # noqa: E402

ORDER = ["C0", "C1", "C2"]
DEFAULT_SOURCES = ["hotpot_ctx_ar", "arabicaqa"]     # MAFQA excluded: open-ended fatwa answers can't be judged reliably


class Generator:
    def __init__(self, complete_fn, model, budget, params):
        self.complete_fn, self.model, self.budget, self.params = complete_fn, model, budget, params

    def ask(self, prompt):
        return self.complete_fn(prompt, model=self.model, budget=self.budget, **self.params)


class LockedRetriever:
    """BM25 scoring and the GPU encoder are shared by all worker threads: serialise searches."""
    def __init__(self, r):
        self.r, self.lock = r, threading.Lock()

    def search(self, q, k=10, method="hybrid"):
        with self.lock:
            return self.r.search(q, k=k, method=method)

    def text(self, pid):
        return self.r.text(pid)


def select(processed, sources, n_per_source, cap_arabicaqa, seed):
    out = []
    for src in sources:
        f = Path(processed) / f"{src}.jsonl"
        if not f.exists():
            print(f"  {src}: {f.name} not found, skipping")
            continue
        exs = [e for e in read_jsonl(f) if e.answer and e.gold_passage_ids]
        Random(seed).shuffle(exs)                              # same seed -> same order in every run
        exs = [e for e in exs if usable_ref(e.answer)]          # filter AFTER shuffling: order stays comparable
        n = n_per_source or (cap_arabicaqa if src == "arabicaqa" else len(exs))
        out += exs[:n]
    return out


def usable_ref(ans):
    """Some ArabicaQA references are degenerate (e.g. '.'): a judge cannot grade against them."""
    return bool(ans) and len(re.findall(r"[\w\u0600-\u06FF]", str(ans))) >= 2


def parse_prices(items):
    prices = {}
    for it in items or []:
        model, vals = it.split("=")
        pin, pout = vals.split(",")
        prices[model] = (float(pin), float(pout))
    return prices


def process(e, gen, jgen, retriever, mode, k, method):
    refs = [e.answer] + ([e.extra["answer_en"]] if e.extra.get("answer_en") else [])
    gold = set(e.gold_passage_ids)
    results = {}
    for name in ORDER:
        fn = STRATEGIES[name]
        r = fn(e.question, gen) if name == "C0" else fn(e.question, gen, retriever, k=k, method=method)
        ok, how, jusage = judge(e.question, r["answer"], refs, jgen)
        r.update(correct=ok, judge=how, judge_input_tokens=jusage.get("input_tokens", 0),
                 gold_found=(len(gold & set(r["passages"])) / len(gold)) if r["passages"] else None)
        results[name] = r
        if ok and mode == "cascade":
            break
    label = next((n for n in ORDER if n in results and results[n]["correct"]), "none")
    yesno = str(e.answer).strip().lower() in ("نعم", "لا", "yes", "no")
    return {"id": e.id, "source": e.source, "split": e.split, "question": e.question, "answer": e.answer,
            "refs": refs, "n_gold": len(gold), "qtype": e.extra.get("type"), "yesno": yesno,
            "label": label, "results": results}


def summarize(records, budget, stopped, run_dir, args):
    where = f"local {args.gen_base_url}" if args.gen_base_url else f"effort {args.effort or '-'}"
    lines = [f"run {args.run} | generator {args.gen_model} ({where}) | judge {args.judge_model} "
             f"| mode {args.mode} | k={args.k} {args.method}",
             f"questions labelled: {len(records):,}" + ("   (STOPPED: budget reached)" if stopped else ""),
             f"spend this run: ${budget.spent:.4f} of ${budget.limit:.2f} (new API calls: {budget.calls})", ""]
    by_src = defaultdict(list)
    for r in records:
        by_src[r["source"]].append(r)
    lines.append(f"{'source':<16}{'n':>6}" + "".join(f"{lab:>9}" for lab in ORDER + ["none"]))
    for src, rs in sorted(by_src.items()):
        c = Counter(r["label"] for r in rs)
        lines.append(f"{src:<16}{len(rs):>6}" + "".join(f"{100 * c[lab] / len(rs):>8.1f}%" for lab in ORDER + ["none"]))
    lines += ["", "per strategy (over the questions where it ran):",
              f"  {'strategy':<9}{'ran':>6}{'correct':>9}{'avg calls':>11}{'avg in tok':>12}{'avg out tok':>13}"
              f"{'gold found':>12}"]
    for name in ORDER:
        rs = [r["results"][name] for r in records if name in r["results"]]
        if not rs:
            continue
        gf = [x["gold_found"] for x in rs if x["gold_found"] is not None]
        lines.append(f"  {name:<9}{len(rs):>6}{100 * sum(x['correct'] for x in rs) / len(rs):>8.1f}%"
                     f"{sum(x['calls'] for x in rs) / len(rs):>11.2f}{sum(x['input_tokens'] for x in rs) / len(rs):>12.0f}"
                     f"{sum(x['output_tokens'] for x in rs) / len(rs):>13.0f}"
                     + (f"{100 * sum(gf) / len(gf):>11.1f}%" if gf else f"{'-':>12}"))
    typed = [r for r in records if r.get("qtype")]
    if typed:
        lines += ["", "HotpotQA by question type (comparison questions with two options can be GUESSED without knowledge):"]
        for qt in sorted({r["qtype"] for r in typed}):
            rs = [r for r in typed if r["qtype"] == qt]
            c = Counter(r["label"] for r in rs)
            lines.append(f"  {qt:<12}{len(rs):>6}" + "".join(f"{lab} {100 * c[lab] / len(rs):5.1f}%  " for lab in ORDER + ["none"]))
        yn = [r for r in records if r.get("yesno")]
        if yn:
            lines.append(f"  yes/no questions: {len(yn)}, of which labelled C0: {sum(r['label'] == 'C0' for r in yn)}")
    jm = Counter(x["judge"] for r in records for x in r["results"].values())
    lines += ["", f"judge decisions: {dict(jm)}",
              "'gold found' = share of the gold passages that were among the passages the strategy used."]
    text = "\n".join(lines)
    (run_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


def main(argv=None, complete_fn=None, retriever=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gen_model", default=None, help="default: ARROUTE_LLM_MODEL")
    ap.add_argument("--judge_model", default=None, help="default: ARROUTE_LLM_MODEL")
    ap.add_argument("--effort", default=None, help="reasoning effort for reasoning models, e.g. low")
    ap.add_argument("--gen_base_url", default=None,
                    help="LOCAL generator: OpenAI-compatible server, e.g. http://localhost:11434/v1 (Ollama). Free.")
    ap.add_argument("--retriever_device", default=None, choices=["cpu", "cuda"],
                    help="where BGE-M3 runs; use cpu when a local LLM needs the GPU memory")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--n_per_source", type=int, default=None)
    ap.add_argument("--cap_arabicaqa", type=int, default=1000)
    ap.add_argument("--mode", choices=["cascade", "full"], default="cascade")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget_usd", type=float, required=True)
    ap.add_argument("--price", action="append", help="MODEL=IN,OUT in USD per 1M tokens (repeatable)")
    ap.add_argument("--max_out_gen", type=int, default=None, help="default: 300, or 3000 with --effort")
    ap.add_argument("--max_out_judge", type=int, default=60)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/labels")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    import os
    a.gen_model = a.gen_model or os.environ.get("ARROUTE_LLM_MODEL")
    a.judge_model = a.judge_model or os.environ.get("ARROUTE_LLM_MODEL")
    if not a.gen_model or not a.judge_model:
        raise SystemExit("set ARROUTE_LLM_MODEL or pass --gen_model and --judge_model")
    if complete_fn is None:
        from arroute.llm import complete_with_usage as complete_fn
    prices = parse_prices(a.price)
    if a.gen_base_url and a.gen_model not in prices:
        prices[a.gen_model] = (0.0, 0.0)                         # local model: no API cost
    budget = Budget(a.budget_usd, prices)
    budget.price(a.gen_model)                                    # fail fast if a price is missing
    budget.price(a.judge_model)
    if a.gen_base_url:
        gen_params = {"temperature": 0, "max_output_tokens": a.max_out_gen or 1024, "base_url": a.gen_base_url}
    elif a.effort:
        gen_params = {"reasoning": {"effort": a.effort}, "max_output_tokens": a.max_out_gen or 3000}
    else:
        gen_params = {"temperature": 0, "max_output_tokens": a.max_out_gen or 300}
    gen = Generator(complete_fn, a.gen_model, budget, gen_params)
    jgen = Generator(complete_fn, a.judge_model, budget, {"temperature": 0, "max_output_tokens": a.max_out_judge})
    if retriever is None:
        from arroute.retrieval.index import load_retriever
        retriever = load_retriever(a.corpus, dense=a.method != "bm25", device=a.retriever_device)
    retriever = LockedRetriever(retriever)

    run_dir = Path(a.out) / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_f = run_dir / "config.json"
    if cfg_f.exists():                                         # resuming: the settings must be identical
        old = json.loads(cfg_f.read_text(encoding="utf-8"))
        keys = ["mode", "sources", "gen_model", "judge_model", "effort", "gen_base_url", "k", "method"]
        diff = {k_: (old.get(k_), vars(a).get(k_)) for k_ in keys if old.get(k_) != vars(a).get(k_)}
        if diff:
            raise SystemExit(f"run '{a.run}' was made with different settings {diff}. Use a new --run name.")
    out_f = run_dir / "labels.jsonl"
    done = {}
    if out_f.exists():
        for line in out_f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:                     # a line cut off by a hard stop: redo it
                    continue
                done[r["id"]] = r
    todo = [e for e in select(a.processed, a.sources, a.n_per_source, a.cap_arabicaqa, a.seed) if e.id not in done]
    print(f"[{a.run}] {len(done):,} already labelled, {len(todo):,} to go | budget ${a.budget_usd:.2f}")
    cfg_f.write_text(json.dumps(vars(a), indent=1), encoding="utf-8")

    stopped, failed, t0 = False, 0, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool, out_f.open("a", encoding="utf-8") as fout:
        futs = {pool.submit(process, e, gen, jgen, retriever, a.mode, a.k, a.method): e for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except BudgetExceeded:
                if not stopped:
                    print("  budget reached: no new calls will be made; finishing...")
                stopped = True
                pool.shutdown(wait=False, cancel_futures=True)
                continue
            except Exception as ex:                             # noqa: BLE001  (API error, timeout, ...)
                failed += 1
                print(f"  skipped {futs[fut].id}: {type(ex).__name__}: {str(ex)[:120]}  (a re-run retries it)")
                continue
            done[rec["id"]] = rec
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0:
                print(f"  {i:,}/{len(todo):,} | ${budget.spent:.3f} | {time.time() - t0:.0f}s")
    if failed:
        print(f"\n{failed} question(s) failed and were skipped: run the same command again to finish them (free for cached calls).")
    summarize(list(done.values()), budget, stopped, run_dir, a)
    return list(done.values())


if __name__ == "__main__":
    main()
