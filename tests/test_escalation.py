"""Phase 4c tests: answer checks, escalation costs, honest thresholds, reader CV, report + repeat CLIs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.router import escalate as E                    # noqa: E402
from arroute.router.outcome import IDK, fixed_frontier, outcome_matrix  # noqa: E402
from test_outcome_router import synth                       # noqa: E402

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _rec(a0, a1, pids):
    return {"results": {"C0": {"answer": a0}, "C1": {"answer": a1, "passages": pids}}}


def test_answer_features():
    pt = {"p1": "ولد نجيب محفوظ في القاهرة عام 1911", "p2": "النيل نهر طويل"}
    f = E.answer_features(_rec("القاهرة", "القاهرة", ["p1", "p2"]), pt)
    assert f["c1_in_passages"] == 1 and f["c1_in_top1"] == 1 and f["c0c1_same"] == 1 and f["c1_idk"] == 0
    f = E.answer_features(_rec(IDK, "الإسكندرية", ["p1", "p2"]), pt)
    assert f["c1_in_passages"] == 0 and f["c0_idk"] == 1 and f["c0c1_overlap"] == 0
    f = E.answer_features(_rec("لا", IDK, ["p2"]), pt)
    assert f["c1_idk"] == 1 and f["c1_tok_cover"] == 0
    assert list(f) == E.ANSWER_FEATURES + E.C0_FEATURES
    assert E.reader_input("س", ["a b c d", "e"], words=2) == "س [SEP] a b [SEP] e"


def test_escalate_costs_and_top_share():
    correct = np.array([[0, 0, 1], [0, 1, 1]], bool)
    cost = np.array([[1, 10, 100], [1, 10, 100]], float)
    hit, c = E.escalate_outcome(correct, cost, [True, False])
    assert hit.tolist() == [1, 1] and c.tolist() == [110, 10]
    hit, c = E.escalate_outcome(correct, cost, [True, False], pre=True, with_c0=True)
    assert c.tolist() == [101, 11]
    esc = E.top_share(np.array([0.1, 0.9, 0.5, 0.7]), 0.5)
    assert esc.tolist() == [False, True, False, True]


def test_threshold_and_bootstrap():
    recs = synth(400)
    correct, cost, calls = outcome_matrix(recs)
    rescue = (~correct[:, 1] & correct[:, 2]).astype(float)
    t = E.threshold_for(rescue, correct, cost, calls, "match_c2")
    esc = rescue >= t
    hit, c = E.escalate_outcome(correct, cost, esc)
    assert hit.mean() >= correct[:, 2].mean() - 0.01 and c.mean() < cost[:, 2].mean()
    assert E.threshold_for(np.zeros(len(recs)), correct, cost, calls, "best_gain") in (np.inf, 0.0)
    b = E.bootstrap_items(hit, c, correct, cost, calls, {"always C2": (correct[:, 2].astype(float), cost[:, 2])}, reps=100)
    assert b["vs_mix"] > 0 and b["vs_mix_ci"][0] <= b["vs_mix"] <= b["vs_mix_ci"][1]
    assert b["always C2"]["cost_ratio"] < 1
    assert fixed_frontier(correct, cost, calls)


def _write_fixture(tmp_path, n=260):
    recs = synth(n)
    for i, r in enumerate(recs):                       # give C1 passages + answers the features can read
        r["results"]["C1"]["passages"] = [f"p{i % 7}", f"p{(i + 1) % 7}"]
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "passages.jsonl").write_text("\n".join(json.dumps({"id": f"p{k}", "text": f"نص رقم {k} ج"}, ensure_ascii=False)
                                                     for k in range(7)), encoding="utf-8")
    return recs


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def test_cli_local_pipeline(tmp_path):
    _write_fixture(tmp_path)
    lab, cor = tmp_path / "labels.jsonl", tmp_path / "corpus"
    feats = tmp_path / "feats" / "escalation_feats.jsonl"
    out = _run(ROOT / "scripts/escalation_features.py", "--labels", lab, "--corpus", cor, "--out", feats)
    assert "not found in the corpus: 0" in out
    _run(ROOT / "scripts/train_sufficiency.py", "--backend", "tfidf", "--labels", lab, "--corpus", cor,
         "--out", tmp_path / "suff")
    d = np.load(tmp_path / "suff" / "tfidf_c1_ok" / "oof.npz")
    assert ((d["p"] >= 0) & (d["p"] <= 1)).all() and len(d["p"]) == 260
    out = _run(ROOT / "scripts/eval_escalation.py", "--labels", lab, "--esc_feats", feats, "--ret_feats", tmp_path / "none.jsonl",
               "--rerank_feats", tmp_path / "none.jsonl", "--readers", tmp_path / "suff", "--reps", "50")
    assert "CEILING: perfect 'C2 rescues'" in out and "reader tfidf_c1_ok" in out
    js = json.loads((tmp_path / "suff" / "escalation_report.json").read_text(encoding="utf-8"))
    ceil = js["detectors"]["CEILING: perfect 'C2 rescues'"]["after C1 | best_gain"]
    assert ceil["vs_mix"] > 0 and js["detectors"]["CEILING: perfect 'C2 rescues'"]["auc"]["rescue"] == 1.0


def test_repeat_report_and_sampling():
    import repeat_check as RC
    recs = synth(300)
    s = RC.sample(recs, 100, 0)
    assert 80 <= len(s) <= 120 and len({r["id"] for r in s}) == len(s)
    pairs = [(r, {"C1": {"correct": r["results"]["C1"]["correct"], "answer": "x"},
                  "C2": {"correct": r["results"]["C2"]["correct"], "answer": "x"}}) for r in s]
    text, js = RC.report(pairs, "t")
    assert js["c1_same"] == 1.0 and js["c2_same"] == 1.0 and js["rescue_both"] == js["rescue_1st"]
    assert RC.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_underscore_params_only_change_cache_key(tmp_path, monkeypatch):
    import arroute.llm as L
    monkeypatch.setattr(L, "CACHE_DIR", tmp_path)
    sent = []

    class FakeResp:
        output_text, status, usage = "جواب", "completed", None

    class FakeClient:
        class responses:
            @staticmethod
            def create(model, input, **kw):
                sent.append(kw)
                return FakeResp()
    monkeypatch.setattr(L, "_get_client", lambda: FakeClient)
    L.complete_with_usage("سؤال", model="m", temperature=0)
    _, _, cached = L.complete_with_usage("سؤال", model="m", temperature=0)
    assert cached and len(sent) == 1
    _, _, cached = L.complete_with_usage("سؤال", model="m", temperature=0, _repeat="r1")
    assert not cached and len(sent) == 2 and "_repeat" not in sent[-1]


def test_rerank_features():
    from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE, rerank_features
    s = [2.0, -1.0, 0.5, -3.0, -2.0] + [-5.0] * 14 + [4.0]
    f = rerank_features(s, [3.0, 0.0, -1, -1, -1])
    assert list(f) == RERANK_PRE + RERANK_POST
    assert f["rr_top5_max"] == 2.0 and f["rr_top5_2nd"] == 0.5 and f["rr_top5_npos"] == 2 and f["rr_rank1"] == 2.0
    assert f["rr_deep_max"] == 4.0 and f["rr_deep_gain"] == 2.0 and f["rr_deep_npos"] == 1
    assert f["rr_ans_max"] == 3.0 and f["rr_ans_gain"] == 1.0
    g = rerank_features(s, None, c1_idk=True)
    assert g["rr_ans_gain"] == 0.0 and g["rr_ans_max"] == 2.0


def test_cli_eval_with_reranker_and_answer_reader(tmp_path):
    from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE
    recs = _write_fixture(tmp_path)
    lab, cor = tmp_path / "labels.jsonl", tmp_path / "corpus"
    feats = tmp_path / "feats" / "escalation_feats.jsonl"
    _run(ROOT / "scripts/escalation_features.py", "--labels", lab, "--corpus", cor, "--out", feats)
    rng = np.random.RandomState(1)
    rr = tmp_path / "feats" / "rerank_feats.jsonl"
    with rr.open("w", encoding="utf-8") as f:                  # plant a signal: rescues have low top-5 relevance
        for r in recs:
            resc = (not r["results"]["C1"]["correct"]) and r["results"]["C2"]["correct"]
            v = {k: float(rng.randn()) for k in RERANK_PRE + RERANK_POST}
            v["rr_top5_max"] -= 3.0 * resc
            f.write(json.dumps({"id": r["id"], "feats": v}) + "\n")
    suff = tmp_path / "suff"
    _run(ROOT / "scripts/train_sufficiency.py", "--backend", "tfidf", "--labels", lab, "--corpus", cor, "--out", suff)
    _run(ROOT / "scripts/train_sufficiency.py", "--backend", "tfidf", "--with_answer", "--labels", lab, "--corpus", cor,
         "--out", suff)
    assert (suff / "tfidf_c1_ok_ans" / "oof.npz").exists()
    out = _run(ROOT / "scripts/eval_escalation.py", "--labels", lab, "--esc_feats", feats, "--ret_feats", tmp_path / "none",
               "--rerank_feats", rr, "--readers", suff, "--reps", "50")
    assert "reranker (LR)" in out and "BEFORE C1: all signals (trees)" in out and "after C1 = tfidf_c1_ok_ans" in out
    js = json.loads((suff / "escalation_report.json").read_text(encoding="utf-8"))
    assert js["detectors"]["reranker (LR)"]["auc"]["rescue"] > 0.8
    assert "before C1 | match_c2" not in js["detectors"]["reader tfidf_c1_ok_ans"]   # answer-aware: after C1 only
