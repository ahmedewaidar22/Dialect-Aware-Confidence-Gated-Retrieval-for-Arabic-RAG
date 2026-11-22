"""Phase 3a: one shared Arabic retrieval corpus + BGE-M3 embeddings.

  python scripts/build_corpus.py            # passages only (BM25 needs nothing else)
  python scripts/build_corpus.py --dense    # + data/corpus/emb_bgem3.npy (GPU recommended)
Sources (all must already be in data/processed, except hotpot_ctx_ar which is built here from the
translated file): ArabicaQA contexts, MAFQA passages, pilot contexts, ArDQA MSA contexts, HotpotQA paragraphs.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.data import LOADERS, Passage, read_jsonl, write_jsonl  # noqa: E402

SOURCES = ["arabicaqa", "mafqa", "mafqa_sub", "pilot", "ardqa", "hotpot_ctx_ar"]


def ensure_hotpot_ctx(raw, processed):
    """Turn data/raw/translated/hotpot_ctx_ar.jsonl into processed files (no full build_dataset needed)."""
    try:
        exs, ps = LOADERS["hotpot_ctx_ar"](raw)
    except FileNotFoundError:
        print("  hotpot_ctx_ar: translated file not present yet, skipping")
        return
    write_jsonl(exs, Path(processed) / "hotpot_ctx_ar.jsonl")
    write_jsonl(ps.values(), Path(processed) / "hotpot_ctx_ar_passages.jsonl")
    print(f"  hotpot_ctx_ar: {len(exs):,} questions, {len(ps):,} passages written to {processed}")


def main(argv=None, encode=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args(argv)

    ensure_hotpot_ctx(a.raw, a.processed)
    seen, passages = set(), []
    for src in SOURCES:
        f = Path(a.processed) / f"{src}_passages.jsonl"
        if not f.exists():
            print(f"  {src}: {f.name} not found, skipping")
            continue
        for p in read_jsonl(f, cls=Passage):
            if p.id not in seen and p.text.strip():
                seen.add(p.id)
                passages.append(Passage(p.id, p.text, src, p.title))
    out = Path(a.out)
    write_jsonl(passages, out / "passages.jsonl")
    counts = Counter(p.source for p in passages)
    stats = "\n".join([f"corpus: {len(passages):,} passages"] + [f"  {k:<14}{v:>8,}" for k, v in counts.most_common()])
    print(stats)
    (out / "corpus_stats.txt").write_text(stats + "\n", encoding="utf-8")

    if a.dense:
        if encode is None:
            from arroute.retrieval.index import bge_m3_encoder
            encode = bge_m3_encoder()
        t0 = time.time()
        try:
            emb = encode([p.text for p in passages], show=True)
        except TypeError:                                    # fake encoders in tests take one argument
            emb = encode([p.text for p in passages])
        np.save(out / "emb_bgem3.npy", emb.astype(np.float16))
        (out / "emb_ids.json").write_text(json.dumps([p.id for p in passages]), encoding="utf-8")
        print(f"embedded {len(passages):,} passages in {time.time() - t0:.0f}s -> {out / 'emb_bgem3.npy'}")


if __name__ == "__main__":
    main()
