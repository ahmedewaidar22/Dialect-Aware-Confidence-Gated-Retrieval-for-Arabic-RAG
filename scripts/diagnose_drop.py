"""EXPLORATORY diagnosis of why the frozen final system lost on the confirmatory questions.

  python scripts/diagnose_drop.py                      # after run_confirmatory.py; local GPU, ~0 API cost
  python scripts/diagnose_drop.py --limit_old 300      # faster: only 300 questions of the first test set

Nothing here may be used to tune the system on the 561 confirmatory questions: any fix found this way must be tested
on a NEW sample. The script reruns retrieval only (the rewrite calls come from the LLM cache; a hard cap of
--budget_usd stops any uncached call from costing more), for the first test set (data/egy_test/egy_test.jsonl) and
the confirmatory set, and compares them on:

  DATASET   question length; word overlap between the Egyptian question and its gold passage; reference answer found
            word for word in the gold passage; share of questions the dialect head sends to rewriting
  RETRIEVAL gold passage in standard RAG's hybrid top 5 (question as typed); in the final system's top-20 candidates;
            in the final system's reranked top 2; rank of the gold passage after reranking
  ERRORS    (confirmatory set, and the first set if runs_final/per_question.csv exists) every question the final system
            got wrong, put in ONE bucket:
              no_gold_top20   the gold passage never reached the 20 candidates (rewrite / search miss)
              cut_by_k        gold in the top 20 but not in the 2 passages C1 read (reranker + k=2 cut it)
              read_but_wrong  gold among the 2 passages, C1 answered (not "I don't know") and was wrong
              escalated_wrong C1 said "I don't know", C2 ran and was still wrong
            plus the same buckets for the questions standard RAG got right and the final system got wrong.

Writes runs_confirm/diagnosis_report.txt and runs_confirm/diagnosis_cases.csv (every final-system error on the new
set, with the question, rewrite, answers, gold rank, bucket) for reading by hand.
"""
import argparse
import csv
import json
import os
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

IDK = "لا أعرف"
FINAL_COL = "FINAL: rewrite + C1 on reranked top-k, 'I don't know' -> C2"
STD_COL = "Standard RAG (C1, hybrid top-5)"


def toks(text, tokenize):
    return set(tokenize(text or ""))


def probe(it, system, tokenize, normalize, check_rewrite):
    """Retrieval-side facts for one question (no generation)."""
    q = it["q_egy"]
    gold = set(it.get("gold_pids") or [])
    st = system.retrieve(q)
    hyb5 = system.r.search(q, k=5)
    order = sorted(range(len(st["cands"])), key=lambda i: (-st["rr"][i], i))
    reranked = [st["cands"][i] for i in order]
    g_rank = next((r + 1 for r, p in enumerate(reranked) if p in gold), None)
    gtext = " ".join(system.r.text(p) for p in gold)
    qt, gt = toks(q, tokenize), toks(gtext, tokenize)
    refs = it.get("refs") or [it.get("answer")]
    return {"id": it["id"], "q_egy": q, "query": st["query"], "p_egy": st["p_egy"], "plan": st["dialect"],
            "q_len": len(qt), "overlap": len(qt & gt) / max(1, len(qt)),
            "answer_in_gold": any(r and normalize(str(r)) in normalize(gtext) for r in refs),
            "gold_top5_hybrid": bool(gold & set(hyb5)), "gold_top20": bool(gold & set(st["cands"])),
            "gold_topk": bool(gold & set(st["passages"])), "gold_rerank_rank": g_rank, "k": len(st["passages"]),
            "rewrite_flags": "; ".join(check_rewrite(q, st["query"], refs[0] if refs else None)) if st["query"] != q else ""}


def bucket(p, final_answer, escalated):
    if escalated:
        return "escalated_wrong"
    if not p["gold_top20"]:
        return "no_gold_top20"
    if not p["gold_topk"]:
        return "cut_by_k"
    return "read_but_wrong"


def pct(xs):
    xs = list(xs)
    return f"{100 * sum(map(bool, xs)) / max(1, len(xs)):5.1f}%"


def summary(name, P):
    ranks = [p["gold_rerank_rank"] for p in P if p["gold_rerank_rank"]]
    return [f"{name}: {len(P)} questions",
            f"  question length (words)           {stats.mean(p['q_len'] for p in P):6.1f}",
            f"  question/gold word overlap         {100 * stats.mean(p['overlap'] for p in P):5.1f}%",
            f"  answer word for word in gold       {pct(p['answer_in_gold'] for p in P)}",
            f"  sent to rewriting (P(egy) >= hi)   {pct(p['plan'] == 'egy' for p in P)}",
            f"  rewrite flagged (number lost etc.) {pct(p['rewrite_flags'] for p in P)}",
            f"  gold in hybrid top 5 (as typed)    {pct(p['gold_top5_hybrid'] for p in P)}   <- what standard RAG reads",
            f"  gold in final top 20 candidates    {pct(p['gold_top20'] for p in P)}",
            f"  gold in final reranked top k       {pct(p['gold_topk'] for p in P)}   <- what the final system's C1 reads",
            f"  gold rank after reranking (median) {stats.median(ranks) if ranks else float('nan'):6.1f}"]


def error_table(title, P, outcome):
    """outcome: id -> dict(final, std, escalated, final_answer)."""
    rows = [(p, outcome[p["id"]]) for p in P if p["id"] in outcome]
    wrong = [(p, o) for p, o in rows if not o["final"]]
    lost = [(p, o) for p, o in wrong if o["std"]]
    L = [f"{title}: final wrong on {len(wrong)} of {len(rows)}; of these, standard RAG right on {len(lost)}"]
    for label, group in (("all final errors", wrong), ("standard right, final wrong", lost)):
        c = Counter(bucket(p, o.get("final_answer"), o.get("escalated")) for p, o in group)
        L.append(f"  {label:28s} " + "  ".join(f"{k} {c.get(k, 0)}" for k in
                                                   ("no_gold_top20", "cut_by_k", "read_but_wrong", "escalated_wrong")))
    std5 = [o["std"] for p, o in rows if p["gold_top5_hybrid"] and not p["gold_topk"]]
    if std5:
        L.append(f"  questions with gold in hybrid top 5 but NOT in final top k: {len(std5)}; standard RAG right on {pct(std5)}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--old", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--old_results", default="runs_final/per_question.csv")
    ap.add_argument("--new", default="data/confirm/confirm_test.jsonl")
    ap.add_argument("--new_results", default="runs_confirm/results.jsonl")
    ap.add_argument("--limit_old", type=int, default=None)
    ap.add_argument("--budget_usd", type=float, default=0.3)
    ap.add_argument("--out", default="runs_confirm")
    a = ap.parse_args()

    from arroute.egy.rewrite import check_rewrite, rewrite_llm
    from arroute.llm import Budget
    from arroute.llm import complete_with_usage as complete_fn
    from arroute.pipeline.final import load_final
    from arroute.retrieval.text import normalize_ar, tokenize_ar

    budget = Budget(a.budget_usd)
    gen_rw = {"fn": complete_fn, "model": os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini"), "budget": budget}
    system = load_final(a.models, a.corpus, rewrite=lambda q: rewrite_llm(q, gen_rw))
    load = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]   # noqa

    new_items = load(a.new)
    new_res = {r["id"]: r["systems"] for r in load(a.new_results)}
    new_items = [it for it in new_items if it["id"] in new_res]
    old_items = [it for it in load(a.old) if it.get("gold_pids")]
    old_out = {}
    if Path(a.old_results).exists():
        with open(a.old_results, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if f"{FINAL_COL} | correct" in r:
                    old_out[r["id"]] = {"final": r[f"{FINAL_COL} | correct"] == "1", "std": r[f"{STD_COL} | correct"] == "1"}
        old_items = [it for it in old_items if it["id"] in old_out] or old_items
    if a.limit_old:
        old_items = old_items[:a.limit_old]

    def run(items, label):
        P = []
        for n, it in enumerate(items, 1):
            P.append(probe(it, system, tokenize_ar, normalize_ar, check_rewrite))
            if n % 100 == 0:
                print(f"  {label}: {n}/{len(items)} | spend ${budget.spent:.3f}")
        return P

    print(f"first test set: {len(old_items)} questions with a gold passage; confirmatory set: {len(new_items)}")
    P_old, P_new = run(old_items, "first set"), run(new_items, "confirmatory set")
    new_out = {i: {"final": s["final|egy"]["correct"], "std": s["standard_rag|egy"]["correct"],
                   "escalated": s["final|egy"].get("escalated", False), "final_answer": s["final|egy"]["answer"],
                   "std_answer": s["standard_rag|egy"]["answer"], "c2": s["always_c2|egy"]["correct"]}
               for i, s in new_res.items()}

    R = ["EXPLORATORY diagnosis (do not tune on the confirmatory questions)", ""]
    R += summary("FIRST TEST SET", P_old) + [""] + summary("CONFIRMATORY SET", P_new) + [""]
    R += error_table("CONFIRMATORY SET", P_new, new_out)
    if old_out:
        R += [""] + error_table("FIRST TEST SET (escalation not stored: escalated errors counted by retrieval bucket)",
                                P_old, old_out)
    text = "\n".join(R)
    print("\n" + text)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis_report.txt").write_text(text + "\n", encoding="utf-8")
    with (out / "diagnosis_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "bucket", "standard_right", "q_egy", "rewritten_query", "plan", "p_egy", "gold_rerank_rank",
                    "gold_top5_hybrid", "final_answer", "standard_answer", "reference", "rewrite_flags"])
        refs = {it["id"]: " | ".join(it.get("refs") or []) for it in new_items}
        for p in P_new:
            o = new_out[p["id"]]
            if not o["final"]:
                w.writerow([p["id"], bucket(p, o["final_answer"], o["escalated"]), int(o["std"]), p["q_egy"], p["query"],
                            p["plan"], f"{p['p_egy']:.2f}", p["gold_rerank_rank"], int(p["gold_top5_hybrid"]),
                            o["final_answer"], o["std_answer"], refs.get(p["id"], ""), p["rewrite_flags"]])
    print(f"\nwritten {out / 'diagnosis_report.txt'} and {out / 'diagnosis_cases.csv'} | API spend ${budget.spent:.4f}")


if __name__ == "__main__":
    main()
