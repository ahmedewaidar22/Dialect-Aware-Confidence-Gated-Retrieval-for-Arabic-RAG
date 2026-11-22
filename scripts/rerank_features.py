"""Phase 4d step 1: reranker signals for every labelled question. Local and free (GPU, ~10-15 min).

  python scripts/rerank_features.py
Downloads BAAI/bge-reranker-v2-m3 once (~2.3 GB) from Hugging Face. Writes
data/router_feats/rerank_feats.jsonl and a summary showing how well each signal separates C1 successes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.outcome import IDK, load_outcomes                          # noqa: E402
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE, rerank_features  # noqa: E402
from arroute.router.route_report import auc                                    # noqa: E402
from arroute.retrieval.rerank import Reranker                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--out", default="data/router_feats/rerank_feats.jsonl")
    a = ap.parse_args()

    from arroute.retrieval.index import load_retriever
    recs = load_outcomes(a.labels)
    t0 = time.time()
    retriever = load_retriever(a.corpus, dense=True, device=a.device)
    cands = [retriever.search(r["question"], k=a.depth, method="hybrid") for r in recs]
    same = np.mean([c[:5] == (r["results"]["C1"].get("passages") or [])[:5] for c, r in zip(cands, recs)])
    print(f"top-{a.depth} candidates for {len(recs):,} questions ({time.time() - t0:.0f}s); "
          f"first 5 = C1's passages for {100 * same:.1f}%")

    rr = Reranker(a.model, a.device, batch=a.batch)
    pairs, owner = [], []
    for i, (r, c) in enumerate(zip(recs, cands)):
        for p in c:
            pairs.append((r["question"], retriever.text(p)))
            owner.append(i)
    scores = rr.score(pairs)
    s20 = [[] for _ in recs]
    for i, s in zip(owner, scores):
        s20[i].append(s)
    print(f"reranked {len(pairs):,} question-passage pairs ({time.time() - t0:.0f}s)")

    # AFTER C1: question + C1's answer against C1's own 5 passages
    apairs, aown = [], []
    for i, r in enumerate(recs):
        ans = r["results"]["C1"].get("answer") or ""
        if ans.strip() and IDK not in ans:
            for p in (r["results"]["C1"].get("passages") or [])[:5]:
                apairs.append((f"{r['question']} الجواب: {ans}", retriever.text(p)))
                aown.append(i)
    ascores = rr.score(apairs) if apairs else np.array([])
    a5 = [[] for _ in recs]
    for i, s in zip(aown, ascores):
        a5[i].append(s)
    feats = [rerank_features(s20[i], a5[i] or None, IDK in (r["results"]["C1"].get("answer") or ""))
             for i, r in enumerate(recs)]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, ft in zip(recs, feats):
            f.write(json.dumps({"id": r["id"], "feats": ft}) + "\n")
    c1 = np.array([r["results"]["C1"]["correct"] for r in recs], bool)
    rescue = np.array([(not r["results"]["C1"]["correct"]) and r["results"]["C2"]["correct"] for r in recs])
    lines = [f"reranker features ({a.model}) for {len(recs):,} questions in {time.time() - t0:.0f}s -> {out}",
             f"first 5 candidates = C1's passages: {100 * same:.1f}%  (should be ~99%+)", "",
             f"{'feature':<16} {'mean':>7}  {'AUC: C1 correct':>16} {'AUC: C2 rescues':>16}"]
    for k in RERANK_PRE + RERANK_POST:
        v = np.array([ft[k] for ft in feats])
        lines.append(f"{k:<16} {v.mean():7.2f}  {auc(v, c1):16.3f} {auc(v, rescue):16.3f}")
    lines.append("(AUC far from 0.5 in EITHER direction = informative; < 0.5 means 'lower value -> more likely')")
    text = "\n".join(lines)
    print(text)
    out.with_name("rerank_feats_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
