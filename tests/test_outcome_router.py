"""Phase 4a tests: outcome-label router core, CV with a fake backend, report, CLI end-to-end (ngram)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arroute.router import outcome as O                       # noqa: E402
from arroute.router.route_report import build_report, load_run  # noqa: E402
import train_router as TR                                     # noqa: E402

IDK = O.IDK


def _res(correct, tin=100, tout=10, calls=1, answer="x"):
    return {"correct": correct, "input_tokens": tin, "output_tokens": tout, "calls": calls, "answer": answer}


def synth(n=300, seed=0):
    """Questions whose text reveals the label ('easy'/'mid'/'hard'), plus some 'none' and comparison ones."""
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        lab = ["C0", "C1", "C2", "none"][rng.choice(4, p=[0.3, 0.4, 0.2, 0.1])]
        word = {"C0": "سهل", "C1": "متوسط", "C2": "صعب", "none": "غريب"}[lab]
        c = {"C0": (1, 1, 1), "C1": (0, 1, 1), "C2": (0, 0, 1), "none": (0, 0, 0)}[lab]
        recs.append({"id": f"q{i}", "source": "hotpot_ctx_ar" if i % 2 else "arabicaqa", "label": lab,
                     "qtype": "comparison" if i % 2 and rng.rand() < 0.1 else ("bridge" if i % 2 else None),
                     "question": f"سؤال {word} رقم {i} " + "كلمة " * rng.randint(1, 6),
                     "results": {"C0": _res(bool(c[0]), 50, 5, answer="ج" if c[0] else (IDK if rng.rand() < .3 else "خطأ")),
                                 "C1": _res(bool(c[1]), 1000, 8, answer="ج" if c[1] else IDK),
                                 "C2": _res(bool(c[2]), 2000, 40, 2, answer="ج" if c[2] else IDK)}})
    return recs


def test_load_outcomes_requires_full_mode(tmp_path):
    r = synth(3)
    del r[1]["results"]["C2"]
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in r), encoding="utf-8")
    with pytest.raises(ValueError, match="full"):
        O.load_outcomes(p)


def test_trainable_and_label_id():
    none = {"label": "none", "qtype": None}
    comp = {"label": "C0", "qtype": "comparison"}
    assert not O.trainable(none) and O.trainable(none, "C0") and O.label_id(none, "C0") == 0
    assert not O.trainable(comp, "C0")
    with pytest.raises(ValueError):
        O.label_id(none)


def test_folds_stratified_and_duplicates_together():
    recs = synth(400)
    recs[5]["question"] = recs[7]["question"] + "  "        # same text after whitespace normalisation
    fold = O.make_folds(recs, k=5, seed=1)
    assert fold[5] == fold[7]
    assert set(fold.tolist()) == set(range(5))
    sizes = np.bincount(fold)
    assert sizes.max() - sizes.min() <= 0.1 * len(recs)
    for lab in ("C0", "C1", "C2"):                          # every fold sees every label
        assert all(((fold == f) & np.array([r["label"] == lab for r in recs])).any() for f in range(5))
    fit, dev = O.inner_dev_split(np.where(fold != 0)[0], recs, 0.15, 0)
    assert not set(fit) & set(dev) and len(dev) > 0
    assert not {recs[i]["question"].strip() for i in fit} & {recs[i]["question"].strip() for i in dev}


def test_policies():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.4, 0.4, 0.2]])
    assert O.choose_argmax(p).tolist() == [0, 2, 0]
    assert O.choose_cautious(p, 0.5).tolist() == [0, 2, 1]
    assert O.choose_cautious(p, 1.0).tolist() == [2, 2, 2]     # needs certainty -> always C2
    assert O.choose_cautious(p, 0.0).tolist() == [0, 0, 0]
    correct = np.array([[0, 1, 1], [0, 0, 0], [1, 1, 1]], bool)
    assert O.oracle_choice(correct).tolist() == [1, 0, 0]


def test_evaluate_and_escalation_costs():
    correct = np.array([[0, 1, 1], [0, 0, 1]], bool)
    cost = np.array([[1, 10, 100], [1, 10, 100]], float)
    calls = np.array([[1, 1, 2], [1, 1, 2]], float)
    idk = np.array([[1, 0, 0], [0, 1, 0]], bool)
    ev = O.evaluate_routing(correct, cost, calls, np.array([0, 1]))
    assert ev["accuracy"] == 0 and abs(ev["cost_per_1k"] - 5.5 * 1000) < 1e-9
    ev = O.evaluate_routing(correct, cost, calls, np.array([0, 1]), idk)
    # q0: C0 says IDK -> C1 (correct), pays 1+10; q1: C1 says IDK -> C2 (correct), pays 10+100
    assert ev["accuracy"] == 1.0 and abs(ev["cost_per_1k"] - (11 + 110) / 2 * 1000) < 1e-9
    assert ev["calls"] == (2 + 3) / 2 and ev["escalated"] == 1.0


def test_frontier_is_upper_hull():
    # C0 (cost 0, acc .2), C1 (cost 1, acc .5), C2 (cost 3, acc .6): C1 is on the hull
    n = 10
    correct = np.zeros((n, 3), bool)
    correct[:2, 0] = True
    correct[:5, 1] = True
    correct[:6, 2] = True
    cost = np.tile([0.0, 0.001, 0.003], (n, 1))
    hull = O.fixed_frontier(correct, cost, np.ones((n, 3)))
    assert [round(h[0], 6) for h in hull] == [0.0, 1.0, 3.0]
    assert abs(O.frontier_acc(hull, 0.5) - 0.35) < 1e-9 and abs(O.frontier_acc(hull, 2.0) - 0.55) < 1e-9
    assert O.frontier_acc(hull, 9.0) == 0.6


def test_bootstrap_and_mcnemar():
    recs = synth(200)
    c, cost, calls = O.outcome_matrix(recs)
    b = O.bootstrap_vs(c, cost, O.fixed_choice(200, 2), O.fixed_choice(200, 1), reps=200, calls=calls)
    assert b["acc_diff"] > 0 and b["acc_diff_ci"][0] <= b["acc_diff"] <= b["acc_diff_ci"][1]
    assert b["cost_ratio"] > 1 and b["c_only_b"] == 0
    assert O.mcnemar_p(0, 0) == 1.0 and O.mcnemar_p(20, 0) < 1e-5
    vm = O.bootstrap_vs_mix(c, cost, calls, O.oracle_choice(c), reps=100)
    assert vm["vs_mix"] > 0 and vm["ci"][0] > 0


def test_label_metrics():
    m = O.label_metrics([0, 1, 2, 2], [0, 1, 1, 2])
    assert m["accuracy"] == 0.75 and m["recall"]["C2"] == 0.5 and m["confusion"][2] == [0, 1, 1]


class FakeBackend:
    """Reads the label word out of the question text: a near-perfect router."""
    def fit(self, texts, y, dt, dy, seed):
        self.calls = getattr(self, "calls", 0) + 1
        self.seen = set(texts)
        return {"best_epoch": 0, "history": []}

    def logits(self, texts):
        out = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            out[i, 0 if "سهل" in t else 2 if "صعب" in t else 1] = 4.0
        return out


def test_cv_and_report(tmp_path):
    recs = synth(300)
    be = FakeBackend()
    res = TR.run_cv(recs, be, k=5, fold_seed=42, seed=0, none_as=None, tol=0.01, log=lambda *_: None)
    assert be.calls == 5
    assert ((res["lam"] >= 0) & (res["lam"] <= 1)).all() and (res["T"] > 0).all()
    run_dir = tmp_path / "fake"
    run_dir.mkdir()
    np.savez_compressed(run_dir / "oof.npz", ids=np.array([r["id"] for r in recs]), fold=res["fold"],
                        logits=res["logits"], T=res["T"], lam=res["lam"], lam_esc=res["lam_esc"])
    (run_dir / "config.json").write_text(json.dumps({"backend": "fake"}), encoding="utf-8")
    run = load_run(run_dir, recs[::-1])                      # order-independent loading
    text, js = build_report(recs[::-1], [run], reps=100)
    assert "BETTER than random mixing" in text
    lab = js["runs"]["fake"]["label"]
    assert lab["accuracy"] > 0.95
    oracle = js["routing_all"]["oracle (upper bound)"]["accuracy"]
    assert js["routing_all"]["fake | argmax"]["accuracy"] >= oracle - 0.05


def test_cv_folds_exclude_test_questions():
    recs = synth(200)

    class Spy(FakeBackend):
        def fit(self, texts, y, dt, dy, seed):
            self.fits = getattr(self, "fits", []) + [set(texts) | set(dt)]
            return super().fit(texts, y, dt, dy, seed)
    be = Spy()
    res = TR.run_cv(recs, be, k=4, fold_seed=0, seed=0, none_as=None, tol=0.01, log=lambda *_: None)
    for f, seen in enumerate(be.fits):
        test_q = {recs[i]["question"] for i in np.where(res["fold"] == f)[0]}
        assert not seen & test_q


def test_cli_ngram_end_to_end(tmp_path):
    lab = tmp_path / "labels.jsonl"
    lab.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in synth(250)), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    for extra in ([], ["--none_as", "C0"]):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/train_router.py"), "--backend", "ngram",
                            "--labels", str(lab), "--out", str(tmp_path / "runs")] + extra,
                           capture_output=True, text=True, encoding="utf-8", env=env)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_router.py"), "--labels", str(lab),
                        "--root", str(tmp_path / "runs"), "--reps", "100"],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "runs" / "router_report.txt").exists() and (tmp_path / "runs" / "ngram_noneC0").is_dir()
    js = json.loads((tmp_path / "runs" / "router_report.json").read_text(encoding="utf-8"))
    assert "ngram | λ dev + IDK→up" in js["routing_all"]
