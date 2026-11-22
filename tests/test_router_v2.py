"""Router v2 tests: confidence features, the cascade arithmetic and threshold rules, the cached log-prob call,
and the whole v2 pipeline (prepare -> answer -> features -> report) with fake models, a fake LLM and a toy corpus."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.router import v2 as V   # noqa: E402


def test_conf_features():
    sure = V.conf_features([{"t": "القاهرة", "lp": -0.01, "top": [-0.01, -6.0]}], "القاهرة")
    unsure = V.conf_features([{"t": "الجيزة", "lp": -1.2, "top": [-0.9, -1.2, -1.5]}, {"t": " ", "lp": -0.1}], "الجيزة")
    assert sure["lp_mean"] > unsure["lp_mean"] and sure["ent_mean"] < unsure["ent_mean"]
    assert sure["margin_mean"] > unsure["margin_mean"] and unsure["n_tok"] == pytest.approx(np.log1p(1))
    assert V.conf_features([], "لا أعرف")["idk"] == 1.0
    assert set(sure) == set(V.CONF)


def test_cascade_arithmetic_and_rules():
    assert V.reranked(["a", "b", "c", "d"], [0.1, 3.0, 3.0, -1], 2) == ["b", "c"]
    OC = {"correct": np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], bool),
          "cost": np.tile([0.00004, 0.0003, 0.001], (4, 1)), "calls": np.tile([1, 1, 2], (4, 1)).astype(float),
          "searches": np.tile([0, 1, 2], (4, 1)).astype(float)}
    p0, p1 = np.array([0.9, 0.2, 0.1, 0.1]), np.array([0.9, 0.8, 0.1, 0.2])
    acc0, esc = V.decide(p0, p1, 0.5, 0.5)
    hit, cost, calls, srch, route = V.outcome(OC, acc0, esc)
    assert route.tolist() == ["C0", "C1R", "C1R>C2", "C1R>C2"] and hit.tolist() == [1, 1, 1, 0]
    assert np.allclose(cost, [0.00004, 0.00034, 0.00134, 0.00134]) and calls.tolist() == [1, 2, 4, 4]
    assert srch.tolist() == [0, 1, 3, 3]
    ta, tc = V.search(p0, p1, OC, "quality")
    hit, cost, *_ = V.outcome(OC, *V.decide(p0, p1, ta, tc))
    assert hit.mean() == pytest.approx(0.75) and cost.mean() < OC["cost"][:, :2].sum(1).mean() + OC["cost"][:, 2].mean()
    ta, tc = V.search(p0, p1, OC, "budget")
    assert V.outcome(OC, *V.decide(p0, p1, ta, tc))[1].mean() <= OC["cost"][:, 1].mean() + 1e-12
    assert set(V.F1) > set(V.F0) and "c0_lp_mean" not in V.NO_CONF(V.F0) and "c0_idk" in V.NO_CONF(V.F0)


def test_detectors_handle_tiny_data():
    X = np.random.RandomState(0).rand(12, 3)
    p = V.oof_scores(X, np.ones(12))                       # one class only: constant, no crash
    assert np.allclose(p, 1.0)
    y = (X[:, 0] > 0.5).astype(int)
    assert V.oof_scores(X, y, groups=np.arange(12)).shape == (12,)


def test_answer_checks():
    f = V.answer_checks("أسوان", "أسوان", ["السد العالي في أسوان", "نص آخر"], rr_ans=[2.0, -1.0], rr_top=1.0)
    assert f["c1_in_passages"] == 1 and f["c1_in_top1"] == 1 and f["c0c1_same"] == 1 and f["rr_ans_gain"] == 1.0
    g = V.answer_checks("لا أعرف", "القاهرة", ["نص"], rr_top=1.5)
    assert g["c1_in_passages"] == 0 and g["rr_ans_max"] == 1.5 and set(f) == set(V.CHECKS)


def test_complete_logprobs_cached(tmp_path, monkeypatch):
    import arroute.llm as L
    n = {"calls": 0}

    def create(**kw):
        n["calls"] += 1
        assert kw["logprobs"] is True and kw["top_logprobs"] == 5
        tok = SimpleNamespace(token="أسوان", logprob=-0.05, top_logprobs=[SimpleNamespace(logprob=-0.05),
                                                                          SimpleNamespace(logprob=-3.2)])
        ch = SimpleNamespace(message=SimpleNamespace(content="أسوان"), finish_reason="stop",
                             logprobs=SimpleNamespace(content=[tok]))
        return SimpleNamespace(choices=[ch], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=3))
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(L, "_get_client", lambda: client)
    b = L.Budget(1.0)
    t, u, cached, toks = L.complete_logprobs("سؤال", model="gpt-4.1-mini", budget=b)
    assert t == "أسوان" and not cached and toks[0]["top"] == [-0.05, -3.2] and u["input_tokens"] == 50 and b.spent > 0
    t2, _, cached2, toks2 = L.complete_logprobs("سؤال", model="gpt-4.1-mini", budget=b)
    assert cached2 and toks2 == toks and n["calls"] == 1


def test_router_v2_end_to_end(tmp_path):
    pytest.importorskip("rank_bm25")
    import eval_phase8 as E
    import router_v2 as R2
    from test_arrouter import _router
    from test_phase8 import TH4, _fake_llm
    r, _, _ = _router()
    r.th = TH4
    r.dialect = lambda t: np.array([0.95 if ("اتبنى" in x or "ايه" in x) else 0.05 for x in t])
    egy = [("السد العالي اتبنى فين؟", "أين بني السد العالي؟", "أسوان"), ("عاصمة مصر ايه؟", "ما عاصمة مصر؟", "القاهرة")]
    items = []
    for i, s in enumerate(["heldout", "pilot", "ardqa", "ardqa", "ardqa", "ardqa"]):
        qe, qm, ans = egy[i % 2]
        items.append({"id": f"egy:{s}:{i}", "set": s, "q_egy": qe, "q_msa": qm, "refs": [ans], "gold_pids": ["p_dam"],
                      "review": {}})
    test = tmp_path / "t.jsonl"
    test.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    log = []
    p8 = tmp_path / "p8"
    E.main(["--test", str(test), "--out", str(p8), "--labels", str(tmp_path / "none"), "--n_ardqa", "1", "--workers", "1"],
           router=r, complete_fn=_fake_llm(log))
    recs = []
    for i in range(8):
        qm, ans = egy[i % 2][1], egy[i % 2][2]
        res = {s: {"answer": ans if (i + j) % 3 else "لا أعرف", "correct": bool((i + j) % 3), "input_tokens": 100 * (j + 1),
                   "output_tokens": 5, "calls": j + 1, "searches": j} for j, s in enumerate(["C0", "C1", "C2"])}
        recs.append({"id": f"tr{i}", "source": "arabicaqa", "question": qm + " " * i, "refs": [ans], "results": res})
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs), encoding="utf-8")
    (tmp_path / "held.json").write_text(json.dumps(["tr7"]), encoding="utf-8")
    (tmp_path / "models" / "complexity_router").mkdir(parents=True)
    (tmp_path / "models" / "complexity_router" / "policy.json").write_text(json.dumps({"thresholds": TH4}), encoding="utf-8")
    fake = _fake_llm(log)

    def lp(prompt, model=None, budget=None):
        t, u, c = fake(prompt)
        return t, u, c, [{"t": t, "lp": -0.2, "top": [-0.2, -2.0]}]
    out = tmp_path / "v2"
    argv = ["--stage", "all", "--test", str(test), "--labels", str(tmp_path / "labels.jsonl"), "--heldout_ids",
            str(tmp_path / "held.json"), "--processed", str(tmp_path / "none"), "--phase8", str(p8),
            "--text_run", str(tmp_path / "none"), "--models", str(tmp_path / "models"), "--n_ardqa", "1",
            "--workers", "2", "--out", str(out)]
    R2.main(argv, router=r, lp_fn=lp, complete_fn=fake)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = [x["split"] for x in rows]
    assert splits.count("train") == 7 and splits.count("calib") == 3 and splits.count("test") == 6
    cfg = json.loads((out / "v2_config.json").read_text(encoding="utf-8"))
    assert 1 <= cfg["k"] <= 5
    js = json.loads((out / "v2_report.json").read_text(encoding="utf-8"))
    names = {x["system"] for x in js["systems"]}
    assert {"ROUTER V2 - quality", "ROUTER V2 - budget", "Router v1 - quality", "Standard RAG (C1, hybrid top-5)"} <= names
    assert js["n"] == 3 and "egy" in js["auc"]
    calib_ans = [json.loads(x) for x in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(("C2" in a["results"]) == a["key"].startswith("calib|") for a in calib_ans)
    n_before = len(log)
    R2.main(argv[:1] + ["report"] + argv[2:], router=r, lp_fn=lp, complete_fn=fake)     # report only: no calls
    assert len(log) == n_before
