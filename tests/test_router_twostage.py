"""Phase 4b tests: retrieval features match the real Retriever; retfeat/two-stage backends run end to end."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.router.retfeats import FEATURES, compute_all     # noqa: E402
from arroute.router.route_report import auc                  # noqa: E402
from test_outcome_router import synth                        # noqa: E402

rank_bm25 = pytest.importorskip("rank_bm25")

DOCS = ["القاهرة عاصمة مصر وأكبر مدنها", "نهر النيل أطول نهر في أفريقيا", "الأهرامات في الجيزة قرب القاهرة",
        "باريس عاصمة فرنسا", "برج إيفل في باريس", "كرة القدم رياضة شعبية في مصر", "محمد صلاح لاعب كرة قدم مصري",
        "الإسكندرية مدينة ساحلية في مصر"]


def _fake_encoder(dim=16):
    from arroute.retrieval.text import tokenize_ar

    def enc(texts):
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize_ar(t):
                out[i, hash(tok) % dim] += 1.0
            out[i] /= max(np.linalg.norm(out[i]), 1e-9)
        return out
    return enc


def test_features_match_real_retriever():
    from arroute.retrieval.index import BM25Index, DenseIndex, Retriever
    from arroute.retrieval.text import normalize_ar
    ids = [f"p{i}" for i in range(len(DOCS))]
    enc = _fake_encoder()
    emb = enc([normalize_ar(d) for d in DOCS])
    bm = BM25Index(ids, DOCS)
    ret = Retriever(bm, DenseIndex(ids, emb, enc), depth=100)
    qs = ["ما هي عاصمة مصر؟", "أين يقع برج إيفل؟", "من هو محمد صلاح", "؟؟"]
    feats, hybs = compute_all(qs, ids, DOCS, emb, enc, bm.bm25, depth=100, k=5, log=lambda *_: None)
    for q, h, f in zip(qs, hybs, feats):
        assert h == ret.search(q, k=5, method="hybrid")          # exactly what C1 would retrieve
        assert list(f) == FEATURES and all(np.isfinite(v) for v in f.values())
    assert feats[0]["cover_top5"] >= 0.5 and feats[3]["bm25_top1"] == 0.0


def test_auc():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(auc([0.1, 0.2], [1, 1]))


def test_twostage_cli_end_to_end(tmp_path):
    recs = synth(260)
    lab = tmp_path / "labels.jsonl"
    lab.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    rng = np.random.RandomState(0)
    feats = tmp_path / "feats.jsonl"
    with feats.open("w", encoding="utf-8") as f:              # feature 0 carries the 'needs C2' signal
        for r in recs:
            v = {k: float(rng.randn()) for k in FEATURES}
            v[FEATURES[0]] += 3.0 * (r["label"] == "C2")
            f.write(json.dumps({"id": r["id"], "feats": v}) + "\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    runs = tmp_path / "runs"

    def run(*args):
        r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", env=env)
        assert r.returncode == 0, r.stderr + r.stdout
        return r.stdout
    tr = ROOT / "scripts/train_router.py"
    run(tr, "--backend", "ngram", "--labels", lab, "--out", runs)                       # stands in for the text head
    run(tr, "--backend", "retfeat", "--labels", lab, "--out", runs, "--feats", feats)
    run(tr, "--backend", "twostage", "--labels", lab, "--out", runs, "--feats", feats, "--text_run", runs / "ngram")
    assert (runs / "twostage__ngram" / "oof.npz").exists()
    d = np.load(runs / "twostage__ngram" / "oof.npz")
    p = np.exp(d["logits"])
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    out = run(ROOT / "scripts/eval_router.py", "--labels", lab, "--root", runs, "--reps", "100")
    assert "3b) WHICH DECISION" in out and "ref: dataset name" in out
    js = json.loads((runs / "router_report.json").read_text(encoding="utf-8"))
    assert js["auc"]["twostage__ngram"]["all"][2] > 0.8                    # it learned the planted C1-vs-C2 signal
    # wrong folds must be refused
    r = subprocess.run([sys.executable, str(tr), "--backend", "twostage", "--labels", str(lab), "--out", str(runs),
                        "--feats", str(feats), "--text_run", str(runs / "ngram"), "--fold_seed", "7", "--force"],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode != 0 and "different folds" in (r.stdout + r.stderr)
