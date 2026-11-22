"""Phase 4c: how REPEATABLE are the outcome labels?  (API, small hard cap)

Re-runs C1 and C2 on a stratified sample of already-labelled questions with fresh API calls (same prompts,
same settings; only the cache key differs), judges them the same way, and measures how often the result
repeats. If 'C1 fails but C2 succeeds' rarely happens twice, no router can predict it well - that sets the
ceiling for Phase 4.

  python scripts/repeat_check.py --n 300 --budget_usd 1.5                    # same settings as labelling (temperature 0)
  python scripts/repeat_check.py --n 300 --budget_usd 1.5 --temperature 0.7  # sampling noise, for comparison
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.llm import Budget, BudgetExceeded                      # noqa: E402
from arroute.pipeline.judge import judge                            # noqa: E402
from arroute.pipeline.strategies import STRATEGIES                  # noqa: E402
from arroute.router.outcome import IDK, load_outcomes               # noqa: E402


def sample(recs, n, seed):
    """Stratified by (source, label) so rare outcomes such as C2 are represented in proportion."""
    groups = defaultdict(list)
    for r in recs:
        groups[(r["source"], r["label"])].append(r)
    rng = np.random.RandomState(seed)
    out = []
    for key in sorted(groups):
        g = groups[key]
        k = max(1, int(round(n * len(g) / len(recs))))
        out += [g[i] for i in rng.choice(len(g), min(k, len(g)), replace=False)]
    return out


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def report(pairs, tag):
    o1 = {s: np.array([p[0]["results"][s]["correct"] for p in pairs], bool) for s in ("C1", "C2")}
    o2 = {s: np.array([p[1][s]["correct"] for p in pairs], bool) for s in ("C1", "C2")}
    same_ans = {s: np.mean([(p[0]["results"][s]["answer"] or "").strip() == (p[1][s]["answer"] or "").strip()
                            for p in pairs]) for s in ("C1", "C2")}
    L = [f"REPEATABILITY ({tag}) on {len(pairs)} questions", "",
         f"{'':<6}{'correct 1st':>12}{'correct 2nd':>12}{'same verdict':>13}{'kappa':>7}{'identical answer':>18}"]
    for s in ("C1", "C2"):
        L.append(f"{s:<6}{100 * o1[s].mean():11.1f}%{100 * o2[s].mean():11.1f}%{100 * (o1[s] == o2[s]).mean():12.1f}%"
                 f"{kappa(o1[s], o2[s]):7.2f}{100 * same_ans[s]:17.1f}%")
    r1 = ~o1["C1"] & o1["C2"]
    r2 = ~o2["C1"] & o2["C2"]
    L += ["", "'C2 rescues' (C1 wrong, C2 right) - the event the escalation detector must predict:",
          f"  1st run {r1.sum()} | 2nd run {r2.sum()} | in both {(r1 & r2).sum()}",
          f"  P(rescue again | rescue in 1st run) = {(r1 & r2).sum() / max(1, r1.sum()):.2f}",
          f"  kappa {kappa(r1, r2):.2f}"]
    # a detector that knew the 1st run perfectly, scored on the 2nd run: the realistic best case
    hit_oracle = np.where(r1, o2["C2"], o2["C1"]).mean()
    L += ["", "A detector that is PERFECT on the labels it was trained on, judged on a fresh run:",
          f"  accuracy {100 * hit_oracle:.1f}% (vs {100 * np.where(r2, o2['C2'], o2['C1']).mean():.1f}% if it could see the fresh run)",
          f"  always-C2 on the fresh run: {100 * o2['C2'].mean():.1f}%"]
    return "\n".join(L), {"n": len(pairs), "c1_same": float((o1["C1"] == o2["C1"]).mean()),
                          "c2_same": float((o1["C2"] == o2["C2"]).mean()), "rescue_1st": int(r1.sum()),
                          "rescue_2nd": int(r2.sum()), "rescue_both": int((r1 & r2).sum()),
                          "rescue_kappa": kappa(r1, r2), "oracle_fresh_acc": float(hit_oracle)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--budget_usd", type=float, required=True)
    ap.add_argument("--price", action="append", default=["gpt-4.1-mini=0.40,1.60"])
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs_suff/repeat")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from label_outcomes import Generator, LockedRetriever, parse_prices
    from arroute.llm import complete_with_usage
    from arroute.retrieval.index import load_retriever

    model = os.environ.get("ARROUTE_LLM_MODEL")
    if not model:
        raise SystemExit("set ARROUTE_LLM_MODEL (the model used for labelling, e.g. gpt-4.1-mini)")
    cfg_f = Path(a.labels).parent / "config.json"
    if cfg_f.exists():
        lab_cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
        if lab_cfg.get("gen_model") and lab_cfg["gen_model"] != model:
            raise SystemExit(f"labels were made with {lab_cfg['gen_model']}, ARROUTE_LLM_MODEL is {model}")
    budget = Budget(a.budget_usd, parse_prices(a.price))
    budget.price(model)
    tag = f"temp{a.temperature:g}"
    # '_repeat' only changes the cache key: identical prompts and settings, fresh answers
    gen = Generator(complete_with_usage, model, budget,
                    {"temperature": a.temperature, "max_output_tokens": 300, "_repeat": tag})
    jgen = Generator(complete_with_usage, model, budget, {"temperature": 0, "max_output_tokens": 60})
    retriever = LockedRetriever(load_retriever(a.corpus, dense=True))
    todo = sample(load_outcomes(a.labels), a.n, a.seed)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"repeat_{tag}.jsonl"
    done = {}
    if out_f.exists():
        for line in out_f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except json.JSONDecodeError:
                pass
    todo_left = [r for r in todo if r["id"] not in done]
    print(f"[repeat {tag}] {len(todo)} questions, {len(done)} done, {len(todo_left)} to go | cap ${a.budget_usd}")

    def one(rec):
        res = {}
        for s in ("C1", "C2"):
            r = STRATEGIES[s](rec["question"], gen, retriever, k=5, method="hybrid")
            ok, how, _ = judge(rec["question"], r["answer"], rec["refs"], jgen)
            res[s] = {"answer": r["answer"], "correct": ok, "judge": how, "calls": r["calls"],
                      "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}
        return {"id": rec["id"], **res}

    t0, stopped = time.time(), False
    with ThreadPoolExecutor(max_workers=a.workers) as pool, out_f.open("a", encoding="utf-8") as fout:
        futs = {pool.submit(one, r): r for r in todo_left}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except BudgetExceeded:
                stopped = True
                pool.shutdown(wait=False, cancel_futures=True)
                continue
            except Exception as ex:                                # noqa: BLE001
                print(f"  skipped {futs[fut]['id']}: {type(ex).__name__}: {str(ex)[:100]} (re-run retries it)")
                continue
            done[rec["id"]] = rec
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo_left)} | ${budget.spent:.3f} | {time.time() - t0:.0f}s")
    pairs = [(r, done[r["id"]]) for r in todo if r["id"] in done]
    text, js = report(pairs, tag)
    text += f"\n\nspend this run: ${budget.spent:.3f} of ${a.budget_usd}" + ("   (STOPPED at the cap)" if stopped else "")
    text += f"\nIDK answers 2nd run: C1 {np.mean([IDK in (p[1]['C1']['answer'] or '') for p in pairs]):.2f}"
    print("\n" + text)
    (out_dir / f"repeat_{tag}_report.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / f"repeat_{tag}_report.json").write_text(json.dumps(js, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
