"""Router package tests. A FakeBackend stands in for the transformer so the whole pipeline around it runs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.data import write_jsonl                                   # noqa: E402
from arroute.router import metrics as M                              # noqa: E402
from arroute.router.runner import run_seed                            # noqa: E402
from arroute.router.tasks import load_task                            # noqa: E402
import make_splits as ms                                              # noqa: E402
from test_splits import make_P, _translated                           # noqa: E402


def test_temperature_recovers_overconfidence():
    rng = np.random.RandomState(0)
    z = rng.randn(4000, 2)
    y = np.array([rng.choice(2, p=p) for p in M.softmax(z, 1.0)])      # labels drawn from the TRUE probs
    T = M.fit_temperature(z * 3.0, y)                                   # model is 3x overconfident
    assert 2.5 < T < 3.6
    assert M.ece(M.softmax(z * 3.0, T), y) < M.ece(M.softmax(z * 3.0, 1.0), y)


def test_ece_zero_when_perfectly_calibrated_and_metrics():
    p = np.array([[0.5, 0.5]] * 10)
    assert M.ece(p, np.array([0, 1] * 5)) < 1e-9
    m = M.binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert m["accuracy"] == 0.5 and m["recall_pos"] == 0.5 and m["fpr"] == 0.5 and abs(m["macro_f1"] - 0.5) < 1e-9


def _write_splits(tmp):
    P = make_P()
    P["nq_ar"] = _translated("nq", "C1", {8: 120, 10: 100, 12: 80})
    P["hotpot_ar"] = _translated("hp", "C2", {8: 100, 10: 100, 12: 100})
    out = {**ms.build_dialect(P, 1), **ms.build_complexity(P, 1, 150, 30), **ms.build_complexity_clean(P, 1)}
    for name, items in out.items():
        write_jsonl(items, Path(tmp) / f"{name}.jsonl")


class FakeBackend:
    """Logits correlated with the label so metrics are non-trivial; records how it was called."""
    def __init__(self):
        self.rng, self.fit_calls, self.lookup = np.random.RandomState(0), [], {}

    def fit(self, tx, ty, sx, sy, seed):
        self.fit_calls.append((len(tx), len(sx), seed))
        self.lookup = {t: y for t, y in zip(tx, ty)}
        return {"best_epoch": 1, "history": [{"epoch": 0}, {"epoch": 1}]}

    def logits(self, texts):
        n = len(texts)
        base = self.rng.randn(n, 2)
        return base + np.stack([np.zeros(n), np.full(n, 0.5)], axis=1)


def test_tasks_and_runner_end_to_end(tmp_path):
    _write_splits(tmp_path)
    for name in ("dialect", "complexity"):
        task = load_task(name, tmp_path)
        assert task.train_texts and set(task.train_y) == {0, 1} and task.select in task.evals
        fb = FakeBackend()
        res = run_seed(task, 3, fb, tmp_path / "runs" / name, {"model": "fake/model", "add_ardqa_dev": False})
        assert fb.fit_calls[0][2] == 3
        assert (tmp_path / "runs" / name / "metrics.json").exists() and (tmp_path / "runs" / name / "logits.npz").exists()
        for ev in res["eval"].values():
            assert set(ev) == {"raw", "calibrated"} and 0 <= ev["raw"]["ece"] <= 1
    leg = res["eval"]["complexity_test"]["raw"]["pos_rate_by_source"]
    assert {"arabicaqa", "mafqa", "mafqa_sub"} <= set(leg)                # per-source stress breakdown present


def test_dialect_flag_moves_ardqa_dev_into_training(tmp_path):
    _write_splits(tmp_path)
    plain, flagged = load_task("dialect", tmp_path), load_task("dialect", tmp_path, add_ardqa_dev=True)
    assert len(flagged.train_texts) > len(plain.train_texts)
    assert flagged.select == "dialect_dev" and "dialect_dev_ardqa [IN TRAINING]" in flagged.evals
    assert "dialect_dev_ardqa" not in flagged.evals


def test_summarize_prints_tables(tmp_path):
    _write_splits(tmp_path)
    (tmp_path / "baselines.json").write_text(json.dumps({"dialect": {"dialect_test_pilot": {
        "ngram_acc": 0.95, "heuristic_acc": 0.84}}, "complexity_clean": {"ngram_bal_acc": 0.88, "words_bal_acc": 0.87}}))
    for name in ("dialect", "complexity"):
        task = load_task(name, tmp_path)
        for seed in (0, 1):
            run_seed(task, seed, FakeBackend(), tmp_path / "runs" / f"{name}__fake_m__seed{seed}",
                     {"model": "org/fake_m", "add_ardqa_dev": False})
    for name in ("dialect", "complexity"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_runs.py"), "--task", name,
                            "--runs", str(tmp_path / "runs"), "--splits", str(tmp_path)], capture_output=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, r.stderr
        assert "2 seed(s)" in r.stdout and "fake_m" in r.stdout and "+-" in r.stdout
    assert "predicted C2" in r.stdout


def test_shortcut_check_writes_baselines_json(tmp_path):
    _write_splits(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/shortcut_check.py"), "--splits", str(tmp_path)],
                       capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    b = json.loads((tmp_path / "baselines.json").read_text())
    assert {"complexity", "complexity_clean", "dialect"} <= set(b)
    assert "ngram_bal_acc" in b["complexity_clean"] and "ngram_acc" in b["dialect"]["dialect_test_pilot"]


def test_torch_version_gate():
    from arroute.router.envcheck import torch_at_least, version_tuple
    assert version_tuple("2.5.1+cu121") == (2, 5, 1) and version_tuple("2.13.0") == (2, 13, 0)
    assert not torch_at_least("2.5.1+cu121") and torch_at_least("2.6.0") and torch_at_least("2.13.0+cu126")
    assert torch_at_least("3.0.0") and not torch_at_least("1.13.1")


def test_complexity_task_has_single_hop_by_design_sets(tmp_path):
    _write_splits(tmp_path)
    task = load_task("complexity", tmp_path)
    names = [n for n in task.evals if n.startswith("c1_by_design_")]
    assert {"c1_by_design_pilot_msa", "c1_by_design_pilot_egy", "c1_by_design_ardqa_msa", "c1_by_design_ardqa_egy"} <= set(names)
    for n in names:
        texts, y, _ = task.evals[n]
        assert texts and set(y) == {0}                     # all single-hop by construction
    res = run_seed(task, 0, FakeBackend(), tmp_path / "r", {"model": "m", "add_ardqa_dev": False})
    m = res["eval"]["c1_by_design_pilot_egy"]["raw"]
    assert np.isnan(m["recall_pos"]) and 0 <= m["fpr"] <= 1       # only false positives are possible


def test_inspect_errors_lists_flagged_questions(tmp_path):
    _write_splits(tmp_path)
    task = load_task("complexity", tmp_path)
    run_dir = tmp_path / "runs" / "complexity__fake__seed0"
    run_seed(task, 0, FakeBackend(), run_dir, {"model": "org/fake", "add_ardqa_dev": False})
    import inspect_errors
    n_fp = inspect_errors.main(["--run", str(run_dir), "--set", "complexity_test", "--source", "arabicaqa",
                                "--kind", "fp", "--n", "5", "--splits", str(tmp_path)])
    texts, y, src = task.evals["complexity_test"]
    logits = np.load(run_dir / "logits.npz")["complexity_test"]
    expect = sum(1 for t, yy, s, lg in zip(texts, y, src, logits) if s == "arabicaqa" and yy == 0 and lg[1] > lg[0])
    assert n_fp == expect


def test_transfer_check_reports_both_directions(tmp_path, capsys):
    P = make_P()
    P["nq_ar"] = _translated("nq", "C1", {8: 120, 10: 100, 12: 80})
    P["hotpot_ar"] = _translated("hp", "C2", {8: 100, 10: 100, 12: 100})
    out = {**ms.build_dialect(P, 1), **ms.build_complexity(P, 1, 150, 30), **ms.build_complexity_clean(P, 1)}
    for name, items in out.items():
        write_jsonl(items, Path(tmp_path) / f"{name}.jsonl")
    import shortcut_check as sc
    res = sc.transfer_check(tmp_path)
    text = capsys.readouterr().out
    assert "clean -> legacy" in text and "legacy -> clean" in text and "arabicaqa" in text
    assert "clean -> legacy | char n-gram LR" in res and 0 <= res["clean -> legacy | char n-gram LR"]["fpr"] <= 1
