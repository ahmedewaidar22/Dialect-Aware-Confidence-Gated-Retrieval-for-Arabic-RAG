"""Phase 10 tests: the final system's logic (dialect -> rerank top-k -> C1 -> 'I don't know' -> C2), the demo, and
the consolidated thesis report built from Phase 8 + Phase 9 outputs."""
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


def _system(k=2):
    from arroute.pipeline.final import FinalSystem
    from test_arrouter import _retriever
    r = _retriever()
    dialect = lambda t: np.array([0.95 if ("اتبنى" in x or "ايه" in x) else (0.5 if "؟؟" in x else 0.05) for x in t])  # noqa: E731
    rer = lambda pairs: np.array([5.0 if "أسوان" in t else -1.0 for _, t in pairs])                                    # noqa: E731
    return FinalSystem(r, dialect, rer, lambda q: "أين بني السد العالي؟", k=k)


def _llm(c1_answer):
    calls = []

    def lp(prompt, model=None, budget=None):
        calls.append("C1")
        return c1_answer, {"input_tokens": 300, "output_tokens": 4}, False, [{"t": c1_answer, "lp": -0.1, "top": [-0.1]}]

    class Gen:
        def ask(self, prompt):
            calls.append("C2")
            return '{"action": "answer", "answer": "أسوان"}', {"input_tokens": 900, "output_tokens": 8}, False
    return lp, Gen(), calls


def test_final_system_routes_and_escalates():
    s = _system()
    st = s.retrieve("السد العالي اتبنى فين؟")
    assert st["dialect"] == "egy" and st["query"] == "أين بني السد العالي؟" and len(st["passages"]) == 2
    assert st["passages"][0] == "p_dam"                                   # reranker put the dam passage first
    assert s.retrieve("أين بني السد العالي؟")["dialect"] == "msa"
    assert len(s.retrieve("السد العالي فين؟؟")["search"]) == 2            # unsure: search both
    lp, gen, calls = _llm("أسوان")
    st = s.answer("السد العالي اتبنى فين؟", lp, gen)
    assert not st["escalated"] and st["answer"] == "أسوان" and calls == ["C1"] and st["calls"] == 1
    lp, gen, calls = _llm("لا أعرف")
    st = s.answer("السد العالي اتبنى فين؟", lp, gen)
    assert st["escalated"] and st["answer"] == "أسوان" and calls[0] == "C1" and "C2" in calls
    assert st["input_tokens"] == 1200 and st["calls"] == 2


def test_final_demo(tmp_path):
    import final_demo as D
    from test_phase8 import _fake_llm
    lp, _, _ = _llm("أسوان")
    D.main(["--question", "السد العالي اتبنى فين؟", "--out", str(tmp_path / "demo.jsonl")], system=_system(),
           lp_fn=lp, complete_fn=_fake_llm([]))
    rec = json.loads((tmp_path / "demo.jsonl").read_text(encoding="utf-8"))
    assert rec["answer"] == "أسوان" and rec["dialect"] == "egy" and rec["escalated"] is False


def test_final_report_end_to_end(tmp_path):
    import final_report as F
    from test_router_v2 import test_router_v2_end_to_end
    test_router_v2_end_to_end(tmp_path)                    # builds p8/ (Phase 8), v2/ (Phase 9) and models/ in tmp_path
    out = tmp_path / "final"
    F.main(["--phase8", str(tmp_path / "p8"), "--v2", str(tmp_path / "v2"), "--router7", str(tmp_path / "none"),
            "--models", str(tmp_path / "models"), "--test", str(tmp_path / "t.jsonl"), "--out", str(out)])
    md = (out / "thesis_tables.md").read_text(encoding="utf-8")
    for t in ("Table 3.", "Table 4.", "Table 5.", "Table 6.", "Table 7.", "Table 8.", "FINAL"):
        assert t in md
    assert md.index("Table 4.") < md.index("Table 5.")
    js = json.loads((out / "thesis_results.json").read_text(encoding="utf-8"))
    names = [r["system"] for r in js["egy"]["systems"]]
    assert F.FINAL in names and "Standard RAG (C1, hybrid top-5)" in names and js["n"] == 3
    assert len((out / "per_question.csv").read_text(encoding="utf-8-sig").splitlines()) == 4
    try:
        import matplotlib  # noqa: F401
        assert (out / "cost_accuracy_egy.png").exists()
    except ImportError:
        pass
