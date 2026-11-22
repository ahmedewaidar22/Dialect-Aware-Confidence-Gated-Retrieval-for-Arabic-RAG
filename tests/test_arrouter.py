"""Phase 7 tests: the full router's decisions with fake models on a toy corpus, and the query-plan evaluation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("rank_bm25")
from arroute.pipeline.arrouter import ARRouter, fuse            # noqa: E402
from arroute.router.escalate import ANSWER_FEATURES             # noqa: E402
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE  # noqa: E402
from arroute.router.retfeats import FEATURES                    # noqa: E402
from test_rewrite import DOCS, _encoder                         # noqa: E402

TH = {"quality": {"ta": None, "tb": 0.5, "tc": 0.5}, "budget": {"ta": 0.5, "tb": None, "tc": None}}


def _retriever():
    import arroute.retrieval.index as I
    from arroute.retrieval.text import normalize_ar
    enc = _encoder()
    ids, texts = list(DOCS), list(DOCS.values())
    return I.Retriever(I.BM25Index(ids, texts), I.DenseIndex(ids, enc([normalize_ar(t) for t in texts]), enc), passages=DOCS)


def _router(p_egy=0.9, p_c0=0.1, p_c2=0.1, p_rescue=0.9, log=None):
    log = log if log is not None else {}
    n_pre = 1 + len(FEATURES) + len(RERANK_PRE)

    def stage_b(X):
        log["xb"] = X.shape[1]
        return np.full(len(X), p_c2)

    def stage_c(X):
        log["xc"] = X.shape[1]
        return np.full(len(X), p_rescue)
    rw = {"السد العالي اتبنى فين؟": "أين بني السد العالي؟"}
    r = ARRouter(_retriever(), TH, lambda t: np.full(len(t), p_egy), lambda t: np.tile([p_c0, 1 - p_c0, 0], (len(t), 1)),
                 lambda t: np.full(len(t), 0.6), lambda t: np.full(len(t), 0.4), stage_b, stage_c,
                 lambda pairs: np.linspace(3, -3, len(pairs)), lambda q: rw.get(q, q + " (فصحى)"))
    return r, log, n_pre


def test_dialect_band_and_routes():
    r, log, n_pre = _router(p_egy=0.95)
    st = r.route("السد العالي اتبنى فين؟")
    assert st["dialect"] == "egy" and st["search"] == ["أين بني السد العالي؟"] and st["query"] == "أين بني السد العالي؟"
    assert st["strategy"] == "C1" and log["xb"] == n_pre and st["passages"][0] == "p_dam"
    r, _, _ = _router(p_egy=0.05)
    st = r.route("أين بني السد العالي؟")
    assert st["dialect"] == "msa" and st["search"] == ["أين بني السد العالي؟"]
    r, _, _ = _router(p_egy=0.5)
    st = r.route("السد العالي اتبنى فين؟")
    assert st["dialect"] == "unsure" and len(st["search"]) == 2 and "p_dam" in st["passages"]
    r, _, _ = _router(p_c2=0.9)
    assert r.route("أين بني السد العالي؟")["strategy"] == "C2"
    r, _, _ = _router(p_c0=0.9)
    assert r.route("أين بني السد العالي؟", mode="budget")["strategy"] == "C0"
    assert r.route("أين بني السد العالي؟", mode="quality")["strategy"] != "C0"     # quality: stage A switched off


def test_after_c1_feature_vector_and_escalation():
    r, log, n_pre = _router(p_rescue=0.9)
    st = r.route("أين بني السد العالي؟")
    assert r.after_c1(st, "أسوان") is True
    assert log["xc"] == n_pre + len(ANSWER_FEATURES) + len(RERANK_POST) + 1
    r, _, _ = _router(p_rescue=0.1)
    st = r.route("أين بني السد العالي؟")
    assert r.after_c1(st, "لا أعرف") is False


def test_answer_runs_strategies_with_routed_passages():
    calls = []

    class Gen:
        def ask(self, prompt):
            calls.append(prompt)
            if "step by step" in prompt or "SEVERAL" in prompt:
                return '{"action": "answer", "answer": "أسوان"}', {"input_tokens": 10, "output_tokens": 2}, False
            return "أسوان", {"input_tokens": 10, "output_tokens": 2}, False
    r, _, _ = _router(p_egy=0.95, p_rescue=0.9)
    st = r.answer("السد العالي اتبنى فين؟", Gen())
    assert st["escalate"] and st["result"]["strategy"] == "C2" and st["result"]["calls"] >= 2
    assert any("أسوان" in DOCS["p_dam"] and "السد العالي في أسوان" in c for c in calls)   # C1 read the routed passage


def test_fuse():
    a = {"bm25": ["x", "y"], "dense": ["y", "z"], "hybrid": ["y", "x", "z"]}
    assert fuse([a]) == ["y", "x", "z"]
    assert set(fuse([a, {"bm25": ["w"], "dense": ["w"], "hybrid": ["w"]}])) == {"x", "y", "z", "w"}


def test_eval_query_plan(tmp_path, monkeypatch):
    import arroute.retrieval.index as I
    import eval_query_plan as Q
    items = [{"id": "e1", "set": "pilot", "q_egy": "السد العالي اتبنى فين؟", "q_msa": "أين بني السد العالي؟",
              "gold_pids": ["p_dam"], "review": {}},
             {"id": "e2", "set": "ardqa", "q_egy": "عاصمة مصر ايه؟", "q_msa": "ما عاصمة مصر؟", "gold_pids": ["p_cap"], "review": {}}]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    (tmp_path / "rw.jsonl").write_text("\n".join(json.dumps({"id": i["id"], "egy>llm": i["q_msa"], "msa>llm": i["q_msa"]},
                                                            ensure_ascii=False) for i in items), encoding="utf-8")
    (tmp_path / "ds.jsonl").write_text("\n".join(json.dumps({"id": i["id"], "p_egy_on_egy": 0.9, "p_egy_on_msa": 0.5})
                                                 for i in items), encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({"lo": 0.2, "hi": 0.8}), encoding="utf-8")
    monkeypatch.setattr(I, "load_retriever", lambda *a_, **k: _retriever())
    monkeypatch.setattr(sys, "argv", ["eval_query_plan.py", "--test", str(tmp_path / "t.jsonl"), "--rewrites", str(tmp_path / "rw.jsonl"),
                                      "--dialect_scores", str(tmp_path / "ds.jsonl"), "--dialect_meta", str(tmp_path / "meta.json"),
                                      "--out", str(tmp_path / "out")])
    Q.main()
    js = json.loads((tmp_path / "out" / "query_plan_report.json").read_text(encoding="utf-8"))
    allp = js["plans"]["ALL inputs"]
    assert allp["always both"]["searches"] == 2 and allp["never rewrite"]["searches"] == 1
    assert allp["dialect head + band"]["searches"] == 1.5           # Egyptian -> rewrite (1), MSA unsure -> both (2)
    assert allp["oracle dialect"]["hybrid"] >= allp["never rewrite"]["hybrid"]
