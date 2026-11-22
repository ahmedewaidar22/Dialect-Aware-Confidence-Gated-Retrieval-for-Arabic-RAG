"""Phase 6 tests: rule rewriter, rewrite checks, and the full rewrite_eval script on a toy corpus (fake LLM)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arroute.egy.rewrite import check_rewrite, rewrite_rules     # noqa: E402

pytest.importorskip("rank_bm25")


def test_rules():
    assert rewrite_rules("امتى السد العالي اتبنى؟") == "متى السد العالي بنى؟"
    assert rewrite_rules("مين اللي بيلعب في الاتحاد؟") == "من الذي يلعب في الاتحاد؟"
    assert rewrite_rules("وإيه أغراضها؟") == "وما أغراضها؟"
    assert rewrite_rules("ما عاصمة مصر؟") == "ما عاصمة مصر؟"                 # MSA stays MSA


def test_checks():
    assert check_rewrite("اتولد امتى سنة 1911؟", "متى ولد عام 1911؟") == []
    assert "number lost" in check_rewrite("اتولد سنة 1911؟", "متى ولد؟")
    assert "answer added" in check_rewrite("مين بنى الهرم؟", "من بنى الهرم خوفو؟", answer="خوفو")
    assert "still Egyptian words" in check_rewrite("مين ده؟", "مين هذا؟")


DOCS = {"p_dam": "بني السد العالي في أسوان", "p_cap": "القاهرة هي عاصمة مصر", "p_nile": "نهر النيل أطول نهر",
        "p_salah": "محمد صلاح لاعب كرة قدم", "p_x": "نص آخر عن شيء مختلف"}


def _encoder(dim=32):
    from arroute.retrieval.text import tokenize_ar

    def enc(texts):
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize_ar(t):
                out[i, hash(tok) % dim] += 1
            out[i] /= max(np.linalg.norm(out[i]), 1e-9)
        return out
    return enc


def test_rewrite_eval_end_to_end(tmp_path, monkeypatch):
    import arroute.llm as L
    import arroute.retrieval.index as I
    import rewrite_eval as RE
    from arroute.retrieval.text import normalize_ar
    items = [{"id": "e1", "set": "pilot", "q_egy": "السد العالي اتبنى فين؟", "q_msa": "أين بني السد العالي؟",
              "answer": "أسوان", "gold_pids": ["p_dam"], "review": {}},
             {"id": "e2", "set": "ardqa", "q_egy": "عاصمة مصر ايه؟", "q_msa": "ما عاصمة مصر؟", "answer": "القاهرة",
              "gold_pids": ["p_cap"], "review": {}},
             {"id": "e3", "set": "heldout", "q_egy": "مين محمد صلاح؟", "q_msa": "من هو محمد صلاح؟", "answer": "لاعب",
              "gold_pids": ["p_salah"], "review": {"q_egy": "rejected"}}]
    test = tmp_path / "egy_test.jsonl"
    test.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    msa_of = {i["q_egy"]: i["q_msa"] for i in items}

    def fake(prompt, model=None, budget=None, **kw):
        q = prompt.split("السؤال: ")[-1]
        return msa_of.get(q, q), {}, False                           # 'perfect' rewriter; MSA comes back unchanged
    monkeypatch.setattr(L, "complete_with_usage", fake)
    monkeypatch.setenv("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    enc = _encoder()
    ids, texts = list(DOCS), list(DOCS.values())
    emb = enc([normalize_ar(t) for t in texts])
    retr = I.Retriever(I.BM25Index(ids, texts), I.DenseIndex(ids, emb, enc), passages=DOCS)
    monkeypatch.setattr(I, "load_retriever", lambda *a_, **k: retr)
    monkeypatch.setattr(sys, "argv", ["rewrite_eval.py", "--test", str(test), "--out", str(tmp_path / "out"),
                                      "--reps", "50", "--n_review", "5"])
    RE.main()
    js = json.loads((tmp_path / "out" / "rewrite_report.json").read_text(encoding="utf-8"))
    rec = js["recall"]["ALL"]
    assert rec["egy>llm"]["hybrid"]["20"] == rec["msa"]["hybrid"]["20"]       # perfect rewriter = MSA twin
    assert js["diagnostics"]["msa_unchanged_by_llm"] == 1.0
    rows = (tmp_path / "out" / "ranks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2                                                        # the rejected item is excluded
    assert "THE DIALECT GAP" in (tmp_path / "out" / "rewrite_report.txt").read_text(encoding="utf-8")


def test_fused_lists_and_variants_end_to_end(tmp_path, monkeypatch):
    import arroute.llm as L
    import arroute.retrieval.index as I
    import rewrite_eval as RE
    import rewrite_variants as RV
    from arroute.retrieval.text import normalize_ar
    # fusion: a passage found by either query survives; hybrid fuses all four lists
    f = RV.fused_lists([{"bm25": ["a", "b"], "dense": ["c", "d"], "hybrid": []},
                        {"bm25": ["x", "a"], "dense": ["c", "y"], "hybrid": []}])
    assert f["bm25"][0] == "a" and f["dense"][0] == "c" and set(f["hybrid"]) == {"a", "b", "c", "d", "x", "y"}

    items = [{"id": "e1", "set": "pilot", "q_egy": "السد العالي اتبنى فين؟", "q_msa": "أين بني السد العالي؟",
              "answer": "أسوان", "gold_pids": ["p_dam"], "review": {}},
             {"id": "e2", "set": "ardqa", "q_egy": "عاصمة مصر ايه؟", "q_msa": "ما عاصمة مصر؟", "answer": "القاهرة",
              "gold_pids": ["p_cap"], "review": {}}]
    test = tmp_path / "egy_test.jsonl"
    test.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    msa_of = {i["q_egy"]: i["q_msa"] for i in items}
    seen = []

    def fake(prompt, model=None, budget=None, **kw):
        seen.append("بأقل تغيير" in prompt)
        q = prompt.split("السؤال: ")[-1]
        return msa_of.get(q, q), {}, False
    monkeypatch.setattr(L, "complete_with_usage", fake)
    monkeypatch.setenv("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    enc = _encoder()
    ids, texts = list(DOCS), list(DOCS.values())
    emb = enc([normalize_ar(t) for t in texts])
    retr = I.Retriever(I.BM25Index(ids, texts), I.DenseIndex(ids, emb, enc), passages=DOCS)
    monkeypatch.setattr(I, "load_retriever", lambda *a_, **k: retr)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["rewrite_eval.py", "--test", str(test), "--out", str(out), "--reps", "50"])
    RE.main()
    assert not any(seen)                                   # Phase 6 uses the original prompt
    monkeypatch.setattr(sys, "argv", ["rewrite_variants.py", "--test", str(test), "--rewrites", str(out / "rewrites.jsonl"),
                                      "--out", str(out), "--reps", "50"])
    RV.main()
    assert any(seen)                                       # 6b uses the minimal-edit prompt
    js = json.loads((out / "variants_report.json").read_text(encoding="utf-8"))
    rec = js["recall"]["ALL"]
    assert rec["egy>min"]["hybrid"]["20"] == rec["msa"]["hybrid"]["20"]
    assert rec["egy+min (fused)"]["hybrid"]["20"] >= rec["egy"]["hybrid"]["20"]
    assert js["best_hybrid_r5_all"] in rec
    assert (out / "variants_ranks.jsonl").exists()
