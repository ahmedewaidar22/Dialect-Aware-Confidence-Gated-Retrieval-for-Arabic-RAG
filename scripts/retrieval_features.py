"""Phase 4b step 1: compute retrieval-confidence features for every labelled question. Local, no API calls.

  python scripts/retrieval_features.py                      # GPU if available (BGE-M3 encodes the questions)

Writes data/router_feats/retrieval_feats.jsonl  ({"id", "feats": {...}} per question) and a summary.
Sanity check: the top-5 passages recomputed here should match the passages C1 actually used when labelling.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.outcome import load_outcomes                  # noqa: E402
from arroute.router.retfeats import FEATURES, compute_all          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/router_feats/retrieval_feats.jsonl")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    a = ap.parse_args()

    from arroute.retrieval.index import bge_m3_encoder, load_corpus
    from rank_bm25 import BM25Okapi
    from arroute.retrieval.text import tokenize_ar

    recs = load_outcomes(a.labels)
    ids, texts, _ = load_corpus(a.corpus)
    emb_f = Path(a.corpus) / "emb_bgem3.npy"
    emb = np.load(emb_f)
    if len(emb) != len(ids):
        raise SystemExit(f"{emb_f} and passages.jsonl differ in size - rebuild the corpus (run_phase3a.ps1)")
    print(f"corpus {len(ids):,} passages | {len(recs):,} questions")
    t0 = time.time()
    bm25 = BM25Okapi([tokenize_ar(t) or ["_"] for t in texts])        # same construction as BM25Index
    encode = bge_m3_encoder(device=a.device)
    feats, hybs = compute_all([r["question"] for r in recs], ids, texts, emb, encode, bm25)

    # sanity check against what C1 really retrieved during labelling
    same = [set(h) == set(r["results"]["C1"].get("passages") or []) for h, r in zip(hybs, recs)]
    top1 = [bool(h) and bool(r["results"]["C1"].get("passages")) and h[0] == r["results"]["C1"]["passages"][0]
            for h, r in zip(hybs, recs)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, ft in zip(recs, feats):
            f.write(json.dumps({"id": r["id"], "feats": ft}, ensure_ascii=False) + "\n")
    X = np.array([[ft[k] for k in FEATURES] for ft in feats])
    lines = [f"retrieval features for {len(recs):,} questions in {time.time() - t0:.0f}s -> {out}",
             f"same top-5 set as C1 used during labelling: {100 * np.mean(same):.1f}%   same top-1: {100 * np.mean(top1):.1f}%",
             "(small differences come from GPU half-precision in the question encoder; below ~90% means the corpus changed)",
             "", f"{'feature':<26} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8}"]
    lines += [f"{k:<26} {X[:, j].mean():8.3f} {X[:, j].std():8.3f} {X[:, j].min():8.3f} {X[:, j].max():8.3f}"
              for j, k in enumerate(FEATURES)]
    text = "\n".join(lines)
    print(text)
    out.with_name("retrieval_feats_summary.txt").write_text(text + "\n", encoding="utf-8")
    if np.mean(same) < 0.9:
        print("\nWARNING: low match with the labelling run. Features still work, but tell Claude.")


if __name__ == "__main__":
    main()
