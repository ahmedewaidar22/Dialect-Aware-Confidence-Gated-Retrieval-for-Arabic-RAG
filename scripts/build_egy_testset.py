"""Phase 5: build the Egyptian test set.

  python scripts/build_egy_testset.py --stage build --budget_usd 1         # LLM: ~$0.30, hard cap
  (open data/egy_test/review.csv in Excel, fill the 'ok' / 'fix' columns, save)
  python scripts/build_egy_testset.py --stage import                       # applies your review

Writes data/egy_test/:
  egy_test.jsonl      one line per question: Egyptian + MSA question, answer, gold passages, set, flags, review
  heldout_ids.json    the 300 labelled questions kept OUT of router training from now on (Phase 7 retrains)
  review.csv          100 translations + 30 pilot answers for YOU to check (row, ok, fix, notes)
  summary.txt         counts, automatic checks, and (after import) your acceptance rates
Needs OPENAI_API_KEY + ARROUTE_LLM_MODEL (gpt-4.1-mini). Every call is cached: re-runs are free.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.egy.testset import (answer_in_context, apply_review, check_translation,       # noqa: E402
                                 egy_markers, review_sample, select_heldout)
from arroute.router.metrics import wilson                                                 # noqa: E402
from arroute.router.outcome import load_outcomes                                          # noqa: E402

P_EGY = ("Rewrite this Modern Standard Arabic question the way a native speaker from Cairo would naturally type it "
         "in Egyptian Arabic (عامية مصرية).\n"
         "Rules: keep exactly the same meaning; keep every name, number and date; use typical Egyptian words "
         "(for example إيه، مين، فين، إمتى، إزاي، كام، اللي، ده، دي) where natural; do not answer the question; "
         "do not add or remove information; do not explain.\n"
         "Return only the Egyptian question.\n\nQuestion: {q}")
P_EGY_RETRY = P_EGY + "\n\nYour previous version had these problems: {problems}. Fix them."
P_ANS = ("Read the passage and give the SHORTEST exact span from it that answers the question. "
         "Copy the span exactly as written in the passage, without adding anything. Return only the span.\n\n"
         "Passage: {ctx}\n\nQuestion: {q}")


def llm(prompt, gen):
    text, _, _ = gen["fn"](prompt, model=gen["model"], budget=gen["budget"], temperature=0, max_output_tokens=200)
    return text.strip().strip('"').strip()


def translate(rec, gen):
    msa, ans = rec["question"], rec.get("answer")
    egy = llm(P_EGY.format(q=msa), gen)
    problems = check_translation(msa, egy, ans)
    tries = 1
    if problems:
        egy2 = llm(P_EGY_RETRY.format(q=msa, problems="; ".join(problems)), gen)
        p2 = check_translation(msa, egy2, ans)
        tries = 2
        if len(p2) < len(problems):
            egy, problems = egy2, p2
    return {"id": rec["id"], "q_msa": msa, "q_egy": egy, "flags": problems, "tries": tries}


def extract_answer(item, ctx, gen):
    a = llm(P_ANS.format(ctx=ctx, q=item["q_msa"]), gen)
    flags = [] if answer_in_context(a, ctx) else ["answer not found in context"]
    return a, flags


def gold_map(processed, ids, sources=("arabicaqa", "hotpot_ctx_ar")):
    """id -> gold passage ids, from the processed dataset files (labels.jsonl does not keep them)."""
    from arroute.data import read_jsonl
    want, out = set(ids), {}
    for s in sources:
        f = Path(processed) / f"{s}.jsonl"
        if f.exists():
            for e in read_jsonl(f):
                if e.id in want:
                    out[e.id] = list(e.gold_passage_ids)
    return out


def ardqa_pairs(processed):
    from arroute.data import read_jsonl
    f = Path(processed) / "ardqa.jsonl"
    groups = defaultdict(dict)
    for e in read_jsonl(f):
        if e.split == "test":
            groups[e.parallel_group][e.variety] = e
    out, seen = [], set()
    for g in sorted(groups):
        d = groups[g]
        if "msa" in d and "egy" in d and d["egy"].gold_passage_ids and d["egy"].answer:
            key = (d["egy"].question.strip(), d["egy"].gold_passage_ids[0])
            if key in seen:                                    # exact duplicates exist in ArDQA
                continue
            seen.add(key)
            out.append({"id": f"egy:ardqa:{g.split(':')[-1]}", "set": "ardqa", "q_egy": d["egy"].question,
                        "q_msa": d["msa"].question, "answer": d["egy"].answer, "refs": [d["egy"].answer],
                        "gold_pids": list(d["egy"].gold_passage_ids), "source": "ardqa", "domain": d["egy"].domain,
                        "qtype": None, "label_msa": None, "flags": [], "origin_id": g})
    return out


def pilot_pairs(processed):
    from arroute.data import read_jsonl
    groups = defaultdict(dict)
    for e in read_jsonl(Path(processed) / "pilot.jsonl"):
        groups[e.parallel_group][e.variety] = e
    out = []
    for g in sorted(groups):
        d = groups[g]
        if "msa" in d and "egy" in d:
            out.append({"id": f"egy:{g}", "set": "pilot", "q_egy": d["egy"].question, "q_msa": d["msa"].question,
                        "answer": None, "refs": [], "gold_pids": list(d["msa"].gold_passage_ids), "source": "pilot",
                        "domain": "general", "qtype": None, "label_msa": None, "flags": [], "origin_id": g})
    return out


def build(a):
    from arroute.llm import Budget, complete_with_usage
    from arroute.retrieval.index import load_corpus
    from label_outcomes import parse_prices
    model = os.environ.get("ARROUTE_LLM_MODEL")
    if not model:
        raise SystemExit("set ARROUTE_LLM_MODEL (e.g. gpt-4.1-mini)")
    budget = Budget(a.budget_usd, parse_prices(a.price))
    budget.price(model)
    gen = {"fn": complete_with_usage, "model": model, "budget": budget}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    recs = load_outcomes(a.labels)
    hf = out / "heldout_ids.json"
    if hf.exists():                                         # never re-draw: Phase 7 depends on this list
        held = json.loads(hf.read_text(encoding="utf-8"))
    else:
        held = select_heldout(recs, a.n_heldout, a.seed)
        hf.write_text(json.dumps(held, indent=0), encoding="utf-8")
    by_id = {r["id"]: r for r in recs}
    print(f"[1/4] held-out labelled questions: {len(held)} (saved to {hf}; Phase 7 trains WITHOUT them)")

    print("[2/4] rewriting them into Egyptian ...")
    with ThreadPoolExecutor(a.workers) as pool:
        tr = list(pool.map(lambda i: translate(by_id[i], gen), held))
    gold = gold_map(a.processed, held)
    heldout = []
    for t in tr:
        r = by_id[t["id"]]
        heldout.append({"id": f"egy:held:{r['id']}", "set": "heldout", "q_egy": t["q_egy"], "q_msa": t["q_msa"],
                        "answer": r["answer"], "refs": r["refs"],
                        "gold_pids": gold.get(r["id"], []),
                        "source": r["source"], "domain": "wikipedia", "qtype": r.get("qtype"), "label_msa": r["label"],
                        "flags": t["flags"], "origin_id": r["id"], "tries": t["tries"]})

    print("[3/4] native sets: ArDQA test pairs + pilot pairs; extracting pilot reference answers ...")
    ardqa = ardqa_pairs(a.processed)
    pilot = pilot_pairs(a.processed)
    ids, texts, _ = load_corpus(a.corpus)
    ctx = dict(zip(ids, texts))
    with ThreadPoolExecutor(a.workers) as pool:
        ans = list(pool.map(lambda it: extract_answer(it, ctx.get(it["gold_pids"][0], ""), gen), pilot))
    for it, (an, fl) in zip(pilot, ans):
        it["answer"], it["refs"], it["flags"] = an, [an], fl
    missing_ctx = sum(p not in ctx for it in pilot + ardqa + heldout for p in it["gold_pids"])
    no_gold = sum(not it["gold_pids"] for it in heldout)

    items = ardqa + pilot + heldout
    for it in items:
        it["egy_markers"] = egy_markers(it["q_egy"])
        it.setdefault("review", {})
    with (out / "egy_test.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print("[4/4] review file for you ...")
    rev_t = review_sample(heldout, a.n_review, seed=a.seed)
    rev_a = review_sample(pilot, a.n_review_answers, seed=a.seed, by="set")
    row_map = {}
    with (out / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["row", "kind", "msa_question", "egyptian_question", "answer", "context", "auto_flags", "ok", "fix", "notes"])
        for k, it in enumerate(rev_t, 1):
            row = f"T{k:03d}"
            row_map[row] = (it["id"], "q_egy")
            w.writerow([row, "translation", it["q_msa"], it["q_egy"], it["answer"], "", "; ".join(it["flags"]), "", "", ""])
        for k, it in enumerate(rev_a, 1):
            row = f"A{k:03d}"
            row_map[row] = (it["id"], "answer")
            w.writerow([row, "pilot answer", it["q_msa"], it["q_egy"], it["answer"], ctx.get(it["gold_pids"][0], "")[:600],
                        "; ".join(it["flags"]), "", "", ""])
    (out / "review_map.json").write_text(json.dumps(row_map, ensure_ascii=False, indent=0), encoding="utf-8")

    L = [f"EGYPTIAN TEST SET -> {out / 'egy_test.jsonl'}  ({time.time() - t0:.0f}s, LLM spend ${budget.spent:.3f})", ""]
    for s in ("ardqa", "pilot", "heldout"):
        part = [it for it in items if it["set"] == s]
        flagged = sum(bool(it["flags"]) for it in part)
        mk = sum(bool(it["egy_markers"]) for it in part)
        L.append(f"  {s:<8} {len(part):4d} questions | with an Egyptian marker word {100 * mk / max(1, len(part)):5.1f}% "
                 f"| auto-flagged {flagged}")
    hl = Counter(it["label_msa"] for it in heldout)
    L.append("  held-out label mix (MSA outcomes): " + ", ".join(f"{k} {v}" for k, v in sorted(hl.items()))
             + " | sources: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(it['source'] for it in heldout).items())))
    fl = Counter(p for it in items for p in it["flags"])
    L.append(f"  automatic flags: {dict(fl) or 'none'}")
    L.append(f"  gold passages missing from the corpus: {missing_ctx} | held-out questions without gold ids: {no_gold} (both should be 0)")
    L.append("  (ArDQA/pilot questions were written by people, so a missing marker word there is normal: not every "
             "Egyptian question uses one)")
    L += ["", f"YOUR TURN: open {out / 'review.csv'} in Excel. For each row:",
          "  ok  = y if the Egyptian question (T rows) or the answer (A rows) is right, n if not",
          "  fix = (optional) the corrected text; leave empty if ok",
          "Save as CSV, then run:  python scripts/build_egy_testset.py --stage import"]
    text = "\n".join(L)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")


def do_import(a):
    from judge_sample import read_graded_csv
    out = Path(a.out)
    rows = read_graded_csv(Path(a.reviewed or out / "review.csv"))
    row_map = {k: tuple(v) for k, v in json.loads((out / "review_map.json").read_text(encoding="utf-8")).items()}
    items = [json.loads(x) for x in (out / "egy_test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    items, stats = apply_review(items, rows, row_map)
    for it in items:
        it["egy_markers"] = egy_markers(it["q_egy"])
    with (out / "egy_test.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    L = ["REVIEW IMPORTED"]
    for kind, s in stats.items():
        done = s["ok"] + s["fixed"] + s["rejected"]
        lo, hi = wilson(s["ok"], done) if done else (0, 1)
        L.append(f"  {kind:<12} checked {done:3d} | correct as generated {s['ok']} ({100 * s['ok'] / max(1, done):.0f}%, "
                 f"95% CI {100 * lo:.0f}-{100 * hi:.0f}%) | fixed {s['fixed']} | rejected {s['rejected']} | left blank {s['blank']}")
    if not stats:
        L.append("  nothing matched: was the 'row' column kept? (T001..., A001...)")
    rej = sum(1 for it in items if "rejected" in it.get("review", {}).values())
    L.append(f"  rejected items stay in the file but are EXCLUDED from Phase 8 ({rej})")
    text = "\n".join(L)
    print(text)
    with (out / "summary.txt").open("a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "import"], default="build")
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/egy_test")
    ap.add_argument("--n_heldout", type=int, default=300)
    ap.add_argument("--n_review", type=int, default=100)
    ap.add_argument("--n_review_answers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget_usd", type=float, default=1.0)
    ap.add_argument("--price", action="append", default=["gpt-4.1-mini=0.40,1.60"])
    ap.add_argument("--reviewed", default=None, help="import: path of the reviewed CSV (default: the review.csv)")
    a = ap.parse_args()
    build(a) if a.stage == "build" else do_import(a)


if __name__ == "__main__":
    main()
