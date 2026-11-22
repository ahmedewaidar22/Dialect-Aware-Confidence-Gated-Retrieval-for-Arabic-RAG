"""Corpus + retrievers: BM25 (lexical), dense (BGE-M3 or any encoder), hybrid (reciprocal rank fusion)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .text import normalize_ar, tokenize_ar


def load_corpus(corpus_dir):
    ids, texts, sources = [], [], []
    with (Path(corpus_dir) / "passages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                ids.append(p["id"])
                texts.append(p["text"])
                sources.append(p.get("source", ""))
    return ids, texts, sources


class BM25Index:
    def __init__(self, ids, texts):
        from rank_bm25 import BM25Okapi
        self.ids = list(ids)
        self.bm25 = BM25Okapi([tokenize_ar(t) or ["_"] for t in texts])

    def search(self, query, k=100):
        toks = tokenize_ar(query)
        if not toks:
            return []
        s = self.bm25.get_scores(toks)
        top = np.argsort(-s)[:k]
        return [self.ids[i] for i in top]


class DenseIndex:
    def __init__(self, ids, emb, encode):
        self.ids, self.encode = list(ids), encode
        self.emb = np.asarray(emb, dtype=np.float32)

    def search(self, query, k=100):
        q = self.encode([normalize_ar(query)])[0]
        top = np.argsort(-(self.emb @ q))[:k]
        return [self.ids[i] for i in top]


def rrf(rank_lists, k=60):
    """Reciprocal rank fusion: robust way to combine rankings from different retrievers."""
    score = defaultdict(float)
    for ranking in rank_lists:
        for r, pid in enumerate(ranking):
            score[pid] += 1.0 / (k + r + 1)
    return sorted(score, key=lambda p: -score[p])


class Retriever:
    """search(query, k, method) with method in {'bm25', 'dense', 'hybrid'}; runs each index once per call."""

    def __init__(self, bm25, dense=None, depth=100, passages=None):
        self.bm25, self.dense, self.depth = bm25, dense, depth
        self.passages = passages or {}                      # id -> text, for building prompts

    def text(self, pid):
        return self.passages.get(pid, "")

    def all_methods(self, query, k=20):
        lex = self.bm25.search(query, self.depth)
        out = {"bm25": lex[:k]}
        if self.dense is not None:
            den = self.dense.search(query, self.depth)
            out["dense"] = den[:k]
            out["hybrid"] = rrf([lex, den])[:k]
        return out

    def search(self, query, k=10, method="hybrid"):
        return self.all_methods(query, k)[method]


def bge_m3_encoder(device=None, max_len=512, batch_size=32):
    """Lazy BGE-M3 loader (needs torch + sentence-transformers). Returns encode(texts) -> unit vectors."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = max_len
    if device == "cuda":
        model.half()

    def encode(texts, show=False):
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=show, convert_to_numpy=True).astype(np.float32)
    return encode


def load_retriever(corpus_dir, dense=True, encode=None, device=None):
    ids, texts, _ = load_corpus(corpus_dir)
    bm25 = BM25Index(ids, texts)
    passages = dict(zip(ids, texts))
    if not dense:
        return Retriever(bm25, passages=passages)
    emb_f = Path(corpus_dir) / "emb_bgem3.npy"
    if not emb_f.exists():
        raise FileNotFoundError(f"{emb_f} missing: run scripts/build_corpus.py --dense")
    emb = np.load(emb_f)
    if len(emb) != len(ids):
        raise ValueError("embeddings and passages.jsonl differ in size: rebuild with build_corpus.py --dense")
    return Retriever(bm25, DenseIndex(ids, emb, encode or bge_m3_encoder(device=device)), passages=passages)
