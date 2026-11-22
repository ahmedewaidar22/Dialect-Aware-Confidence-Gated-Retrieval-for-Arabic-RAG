"""Phase 4e tests: 3-stage policy, threshold search, final report and policy saving (no torch)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.router import policy as P                      # noqa: E402
from arroute.router.escalate import ANSWER_FEATURES, C0_FEATURES  # noqa: E402
from arroute.router.outcome import outcome_matrix           # noqa: E402
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE  # noqa: E402
from arroute.router.retfeats import FEATURES                # noqa: E402
from test_outcome_router import synth                       # noqa: E402

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def test_decide_and_outcome():
    correct = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1]], bool)
    cost = np.tile([1.0, 10.0, 100.0], (4, 1))
    calls = np.tile([1.0, 1.0, 2.0], (4, 1))
    sa, sb, sc = np.array([.9, .1, .1, .1]), np.array([0, .9, 0, 0]), np.array([0, 0, 0, .9])
    skip, c2, esc = P.decide(sa, sb, sc, (0.5, 0.5, 0.5))
    assert skip.tolist() == [1, 0, 0, 0] and c2.tolist() == [0, 1, 0, 0] and esc.tolist() == [0, 0, 0, 1]
    hit, c, k = P.outcome(correct, cost, calls, skip, c2, esc)
    assert hit.tolist() == [1, 1, 1, 1] and c.tolist() == [1, 100, 10, 110] and k.tolist() == [1, 2, 1, 3]
    skip, c2, esc = P.decide(sa, sb, sc, (np.inf, np.inf, np.inf))       # all stages off = always C1
    assert not skip.any() and not c2.any() and not esc.any()


def test_search_rules_on_perfect_scores():
    recs = synth(400)
    correct, cost, calls = outcome_matrix(recs)
    sa = correct[:, 0].astype(float)
    sb = (~correct[:, 1] & correct[:, 2]).astype(float)
    sc = sb.copy()
    res = {}
    for rule in P.RULES:
        t = P.search(sa, sb, sc, correct, cost, calls, rule)
        hit, c, _ = P.outcome(correct, cost, calls, *P.decide(sa, sb, sc, t))
        res[rule] = (hit.mean(), c.mean())
    assert res["budget"][1] <= cost[:, 1].mean() + 1e-12 and res["budget"][0] >= correct[:, 1].mean()
    assert res["quality"][0] >= correct[:, 2].mean() - 0.01 and res["quality"][1] < cost[:, 2].mean()
    assert res["match_c1"][1] < cost[:, 1].mean()
    t = P.search(sa, sb, sc, correct, cost, calls, "quality", use=(False, True, False))
    assert t[0] == np.inf and t[2] == np.inf
    fold = np.arange(400) % 5
    skip, c2, esc, ts = P.honest(sa, sb, sc, fold, correct, cost, calls, "balanced")
    assert len(ts) == 5 and skip.dtype == bool


def _fixture(tmp_path):
    recs = synth(300)
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    rng = np.random.RandomState(0)
    feats = tmp_path / "feats"
    feats.mkdir()
    for name, keys in (("retrieval_feats", FEATURES), ("rerank_feats", RERANK_PRE + RERANK_POST),
                       ("escalation_feats", ANSWER_FEATURES + C0_FEATURES)):
        with (feats / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                v = {k: float(rng.randn()) for k in keys}
                v[keys[0]] += 3.0 * ((not r["results"]["C1"]["correct"]) and r["results"]["C2"]["correct"])
                f.write(json.dumps({"id": r["id"], "feats": v}) + "\n")
    ids = np.array([r["id"] for r in recs])
    lab = np.array(["C0", "C1", "C2", "none"])
    text = tmp_path / "runs_router" / "text"
    text.mkdir(parents=True)
    lg = np.stack([[3.0 * (r["label"] == s) for s in ("C0", "C1", "C2")] for r in recs]) + rng.randn(len(recs), 3)
    np.savez_compressed(text / "oof.npz", ids=ids, fold=np.arange(len(recs)) % 5, logits=lg, T=np.ones(len(recs)),
                        lam=np.full(len(recs), 0.8), lam_esc=np.full(len(recs), 0.8))
    (text / "config.json").write_text(json.dumps({"backend": "fake"}), encoding="utf-8")
    for name, ans in (("pre", False), ("ans", True)):
        d = tmp_path / "runs_suff" / name
        d.mkdir(parents=True)
        np.savez_compressed(d / "oof.npz", ids=ids, fold=np.arange(len(recs)) % 5, p=rng.rand(len(recs)))
        (d / "config.json").write_text(json.dumps({"target": "c1_ok", "with_answer": ans}), encoding="utf-8")
    assert lab is not None
    common = ["--labels", tmp_path / "labels.jsonl", "--text_run", text, "--readers", tmp_path / "runs_suff",
              "--pre_reader", "pre", "--ans_reader", "ans", "--esc_feats", feats / "escalation_feats.jsonl",
              "--ret_feats", feats / "retrieval_feats.jsonl", "--rerank_feats", feats / "rerank_feats.jsonl"]
    return recs, common


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def test_cli_eval_and_save(tmp_path):
    recs, common = _fixture(tmp_path)
    rep = tmp_path / "repeat.jsonl"
    rep.write_text("\n".join(json.dumps({"id": r["id"], "C1": {**r["results"]["C1"]}, "C2": {**r["results"]["C2"]}})
                             for r in recs[:50]), encoding="utf-8")
    out = _run(ROOT / "scripts/eval_final_router.py", *common, "--repeat", rep, "--reps", "50", "--out", tmp_path / "final")
    assert "A+B+C (full router) | quality" in out and "FRESH-RUN CHECK on the 50" in out
    js = json.loads((tmp_path / "final" / "final_report.json").read_text(encoding="utf-8"))
    full = js["policies"]["A+B+C (full router) | quality"]
    assert full["accuracy"] >= js["policies"]["always C2"]["accuracy"] - 0.03
    assert js["fresh"]["quality"]["original"] == js["fresh"]["quality"]["fresh"]   # identical outcomes -> identical
    _run(ROOT / "scripts/train_final_router.py", *common, "--skip_torch", "--out", tmp_path / "model")
    pol = json.loads((tmp_path / "model" / "policy.json").read_text(encoding="utf-8"))
    assert set(pol["thresholds"]) == set(P.RULES) and (tmp_path / "model" / "stage_c.joblib").exists()
    assert len(pol["features"]["stage_c"]) == 1 + len(FEATURES) + len(RERANK_PRE) + len(ANSWER_FEATURES) + len(RERANK_POST) + 1
