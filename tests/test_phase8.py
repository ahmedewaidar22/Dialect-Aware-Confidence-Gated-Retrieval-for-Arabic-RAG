"""Phase 8 tests: offline scoring of every system, the rewriter's cost accounting, and the full eval pipeline
end to end with fake models, a fake LLM and a toy corpus."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.router.final_eval import MODES, arrays, paired, report, run_policy, tok_cost   # noqa: E402

TH4 = {"budget": {"ta": 0.5, "tb": 0.5, "tc": 0.5}, "match_c1": {"ta": 0.5, "tb": 0.5, "tc": 0.5},
       "balanced": {"ta": 0.5, "tb": None, "tc": None}, "quality": {"ta": None, "tb": 0.5, "tc": 0.5}}


def _unit(correct, p_c0=0.1, p_c2=0.1, p_res=0.1, rewrote=0, n_search=1):
    return {"correct": correct, "cost": [0.0001, 0.0005, 0.001], "calls": [1, 1, 2], "searches": [0, 1, 2],
            "probs3": [p_c0, 1 - p_c0, 0.0], "p_need_c2": p_c2, "p_rescue": p_res,
            "rewrite_cost": 0.00005 * rewrote, "rewrote": rewrote, "n_search": n_search}


def test_policies_and_costs():
    U = [_unit([1, 1, 1], p_c0=0.9), _unit([0, 0, 1], p_c2=0.9), _unit([0, 0, 1], p_res=0.9), _unit([0, 1, 1])]
    a = arrays(U)
    r = run_policy(a, "router", thresholds=TH4["budget"])
    assert list(r["route"]) == ["C0", "C2", "C1>C2", "C1"]
    assert r["hit"].tolist() == [1, 1, 1, 1]
    assert np.allclose(r["cost"], [0.0001, 0.001, 0.0015, 0.0005])
    assert r["calls"].tolist() == [1, 2, 3, 1] and r["searches"].tolist() == [0, 2, 3, 1]
    q = run_policy(a, "router", thresholds=TH4["quality"])          # stage A off: nothing skips retrieval
    assert "C0" not in set(q["route"])
    assert run_policy(a, "fixed", s=1)["hit"].tolist() == [1, 0, 0, 1]
    assert run_policy(a, "oracle")["route"].tolist() == ["C0", "C2", "C2", "C1"]
    # the rewrite is paid whatever the strategy, and counts as one more LLM call; 'both' searches twice
    b = arrays([_unit([1, 1, 1], p_c0=0.9, rewrote=1, n_search=2), _unit([0, 1, 1], rewrote=1, n_search=2)])
    r = run_policy(b, "router", thresholds=TH4["budget"])
    assert np.allclose(r["cost"], [0.00015, 0.00055]) and r["calls"].tolist() == [2, 2]
    assert r["searches"].tolist() == [0, 2]
    assert tok_cost({"input_tokens": 1e6, "output_tokens": 1e6}) == pytest.approx(2.0)


def test_paired_and_report():
    x, y = np.array([1, 1, 1, 0] * 10, bool), np.array([1, 0, 0, 0] * 10, bool)
    d, lo, hi, p = paired(x, y)
    assert d == pytest.approx(0.5) and lo <= d <= hi and p < 0.001
    rng = np.random.RandomState(0)
    A = {}
    for v in ("egy_t", "egy_r", "msa_t", "msa_r"):
        boost = 0.2 if v != "egy_t" else 0.0
        A[v] = arrays([_unit([rng.rand() < 0.2, rng.rand() < 0.5 + boost, rng.rand() < 0.7 + boost], p_c0=rng.rand(),
                             p_c2=rng.rand(), p_res=rng.rand(), rewrote=int(v == "egy_r")) for _ in range(60)])
    text, js = report(A, TH4, 0.6, ["ardqa"] * 30 + ["pilot"] * 20 + ["heldout"] * 10, {"x": 1})
    names = [r["system"] for r in js["rows"]]
    for m in MODES:
        assert f"YOUR ROUTER - {m}" in names
    assert "Adaptive-RAG (text classifier, tuned)" in names and "Standard RAG (always C1)" in names
    assert "DIALECT GAP" in text and "BY TEST SET" in text and "HEADLINE DIFFERENCES" in text
    assert set(js["gap"]) == {"msa", "egy", "egy_plan", "closed"}


def test_rewrite_usage_keeps_cache_params():
    from arroute.egy.rewrite import rewrite_llm, rewrite_llm_usage
    seen = []

    def fn(prompt, model, budget, **params):
        seen.append(params)
        return "ما عاصمة مصر؟", {"input_tokens": 50, "output_tokens": 8}, False
    g = {"fn": fn, "model": "m", "budget": None}
    assert rewrite_llm("عاصمة مصر ايه؟", g) == "ما عاصمة مصر؟"
    t, u, _ = rewrite_llm_usage("عاصمة مصر ايه؟", {**g, "params": {"reasoning": {"effort": "low"}, "max_output_tokens": 999}})
    assert seen[0] == {"temperature": 0, "max_output_tokens": 200}     # unchanged: Phase 6 cache still hits
    assert seen[1] == {"reasoning": {"effort": "low"}, "max_output_tokens": 999} and u["output_tokens"] == 8


def test_router_force_typed_and_full_scores():
    pytest.importorskip("rank_bm25")
    from test_arrouter import _router
    r, _, _ = _router(p_egy=0.95, p_c0=0.9)
    st = r.route("السد العالي اتبنى فين؟", mode="budget", force="typed", full=True)
    assert st["search"] == ["السد العالي اتبنى فين؟"] and st["strategy"] == "C0"
    assert "p_need_c2" in st and "rr20" in st and len(st["probs3"]) == 3        # full: every score computed
    assert "p_need_c2" not in r.route("السد العالي اتبنى فين؟", mode="budget")    # live: early exit stays


def _fake_llm(log):
    def fn(prompt, model=None, budget=None, **params):
        log.append(prompt[:30])
        u = {"input_tokens": 100, "output_tokens": 10}
        if "Rewrite" in prompt or "Egyptian" in prompt or "عامية" in prompt:
            return "أين بني السد العالي؟", u, False
        if "grading" in prompt:
            return '{"correct": true}', u, False
        if "SEVERAL" in prompt:
            return '{"action": "answer", "answer": "أسوان"}', u, False
        if "own knowledge" in prompt:
            return "لا أعرف", u, False
        return "أسوان", u, False
    return fn


def test_eval_phase8_end_to_end(tmp_path):
    pytest.importorskip("rank_bm25")
    import eval_phase8 as E
    from test_arrouter import _router
    r, _, _ = _router()
    r.th = TH4
    r.dialect = lambda t: np.array([0.95 if ("اتبنى" in x or "ايه" in x) else 0.05 for x in t])
    items = []
    for i, s in enumerate(["heldout", "pilot", "ardqa", "ardqa"]):
        items.append({"id": f"q{i}", "set": s, "q_egy": "السد العالي اتبنى فين؟" if i % 2 == 0 else "عاصمة مصر ايه؟",
                      "q_msa": "أين بني السد العالي؟" if i % 2 == 0 else "ما عاصمة مصر؟",
                      "refs": ["أسوان"] if i % 2 == 0 else ["القاهرة"], "gold_pids": ["p_dam"], "review": {}})
    test = tmp_path / "t.jsonl"
    test.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    log = []
    argv = ["--test", str(test), "--out", str(tmp_path / "out"), "--labels", str(tmp_path / "none.jsonl"),
            "--n_ardqa", "1", "--workers", "2"]
    E.main(argv, router=r, complete_fn=_fake_llm(log))
    js = json.loads((tmp_path / "out" / "final_report.json").read_text(encoding="utf-8"))
    assert js["n"] == 3                                               # heldout + pilot + 1 of the 2 ArDQA
    rows = {(x["section"][:14], x["system"]): x for x in js["rows"]}
    router_q = rows[("EGYPTIAN INPUT", "YOUR ROUTER - quality")]
    std = rows[("EGYPTIAN INPUT", "Standard RAG (always C1)")]
    assert router_q["calls"] > std["calls"]                           # the rewrite is counted as a call ...
    assert router_q["usd_per_1k"] > 0                                 # ... and as cost
    states = [json.loads(x) for x in (tmp_path / "out" / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(states) == 12 and {s["dialect"] for s in states if s["variant"] == "msa_r"} == {"msa"}
    n_calls = len(log)
    E.main(argv, router=r, complete_fn=_fake_llm(log))               # re-run: everything is reused
    assert len(log) - n_calls <= 12                                   # only the (cached in real life) rewrites
    E.main(argv + ["--report_only"], router=r)
    assert (tmp_path / "out" / "final_report.txt").exists()


def test_sol_rewrite_check(tmp_path):
    pytest.importorskip("rank_bm25")
    import sol_rewrite_check as S
    from test_arrouter import _retriever
    items = [{"id": f"a{i}", "set": "ardqa", "q_egy": "السد العالي اتبنى فين؟", "q_msa": "أين بني السد العالي؟",
              "gold_pids": ["p_dam"], "refs": ["أسوان"]} for i in range(3)]
    test = tmp_path / "t.jsonl"
    test.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items), encoding="utf-8")
    log = []
    S.main(["--test", str(test), "--out", str(tmp_path), "--n", "3", "--workers", "1"], complete_fn=_fake_llm(log),
           retriever=_retriever())
    rep = (tmp_path / "sol_rewrite_report.txt").read_text(encoding="utf-8")
    assert "rewritten by gpt-6.1-sol" in rep and "gap closed by sol" in rep
    assert len((tmp_path / "sol_rewrites.jsonl").read_text(encoding="utf-8").splitlines()) == 3
