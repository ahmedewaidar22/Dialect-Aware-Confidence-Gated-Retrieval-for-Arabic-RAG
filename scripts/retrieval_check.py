"""Phase 3a: does retrieval find the gold passage? Per question pool, per retriever, and the DIALECT GAP
(paired MSA vs Egyptian versions of the same question) measured on the full corpus with the FIXED tokenizer.

  python scripts/retrieval_check.py                  # BM25 + dense + hybrid
  python scripts/retrieval_check.py --no-dense       # BM25 only (no GPU needed)
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.data import read_jsonl                      # noqa: E402
from arroute.retrieval.index import load_retriever       # noqa: E402

KS = (1, 5, 10, 20)


def pools(processed, n_arabicaqa, n_ardqa_groups, seed):
    P = Path(processed)
    rd = lambda n: read_jsonl(P / f"{n}.jsonl") if (P / f"{n}.jsonl").exists() else []
    out = {}
    ara = [e for e in rd("arabicaqa") if e.split == "test"]
    out["arabicaqa_test"] = Random(seed).sample(ara, min(n_arabicaqa, len(ara)))
    for base, exs, cap in (("pilot", rd("pilot"), None), ("ardqa", [e for e in rd("ardqa") if e.split == "test"], n_ardqa_groups)):
        groups = defaultdict(dict)
        for e in exs:
            groups[e.parallel_group][e.variety] = e
        paired = sorted(g for g, d in groups.items() if "msa" in d and "egy" in d
                        and d["msa"].gold_passage_ids and d["egy"].gold_passage_ids)
        if cap is not None:
            paired = Random(seed).sample(paired, min(cap, len(paired)))
        for v in ("msa", "egy"):                      # index i in both lists = the SAME question
            out[f"{base}_{v}"] = [groups[g][v] for g in paired]
    out["mafqa"] = rd("mafqa")
    out["hotpot_ctx"] = rd("hotpot_ctx_ar")
    return {k: [e for e in v if e.gold_passage_ids] for k, v in out.items() if v}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate(retriever, pool, methods):
    """-> {method: ({k: recall@k}, all_gold@10, {k: [hit per question]})}"""
    per_q = {m: {k: [] for k in KS} for m in methods}
    allg = {m: 0 for m in methods}
    for e in pool:
        ranked = retriever.all_methods(e.question, k=max(KS))
        gold = set(e.gold_passage_ids)
        for m in methods:
            r = ranked[m]
            for k in KS:
                per_q[m][k].append(bool(gold & set(r[:k])))
            allg[m] += gold <= set(r[:10])
    n = len(pool)
    return {m: ({k: sum(per_q[m][k]) / n for k in KS}, allg[m] / n, per_q[m]) for m in methods}


def main(argv=None, encode=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--n_arabicaqa", type=int, default=500)
    ap.add_argument("--n_ardqa_groups", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    retriever = load_retriever(a.corpus, dense=not a.no_dense, encode=encode)
    methods = ["bm25"] + ([] if a.no_dense else ["dense", "hybrid"])
    res = {}
    print(f"{'pool':<16}{'n':>6}  {'method':<8}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'all-gold@10':>13}")
    for name, pool in pools(a.processed, a.n_arabicaqa, a.n_ardqa_groups, a.seed).items():
        res[name] = evaluate(retriever, pool, methods)
        for m in methods:
            rk, ag, _ = res[name][m]
            print(f"{name:<16}{len(pool):>6}  {m:<8}" + "".join(f"{rk[k]:>8.3f}" for k in KS) + f"{ag:>13.3f}")
    print("\nDIALECT GAP (same questions, MSA minus Egyptian; positive = Egyptian retrieves worse)")
    print("  p = exact McNemar test on the paired questions (below 0.05: the gap is unlikely to be chance)")
    for base in ("pilot", "ardqa"):
        if f"{base}_msa" in res and f"{base}_egy" in res:
            for m in methods:
                gap = {k: res[f"{base}_msa"][m][0][k] - res[f"{base}_egy"][m][0][k] for k in KS}
                cells = []
                for k in KS:
                    hm, he = res[f"{base}_msa"][m][2][k], res[f"{base}_egy"][m][2][k]
                    b = sum(x and not y for x, y in zip(hm, he))
                    c = sum(y and not x for x, y in zip(hm, he))
                    cells.append(f"  @{k}: {100 * gap[k]:+5.1f} (p={mcnemar_exact(b, c):.3g})")
                print(f"  {base:<6} {m:<8}" + "".join(cells))
    print("\nall-gold@10 = every gold passage in the top 10 (for HotpotQA: BOTH supporting paragraphs).")
    return res


if __name__ == "__main__":
    main()
