"""Phase 5 tests: Egyptian checks, held-out selection, build + review import with a fake LLM (no API)."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
from arroute.data import QAExample, write_jsonl                           # noqa: E402
from arroute.egy import testset as T                                    # noqa: E402
from test_outcome_router import synth                                   # noqa: E402


def test_markers_and_checks():
    assert T.egy_markers("امتى السد العالي اتبنى؟") and not T.egy_markers("متى تم بناء السد العالي؟")
    assert T.check_translation("متى بني السد؟", "السد اتبنى امتى؟") == []
    assert "unchanged" in T.check_translation("متى بني السد؟", "متى بني السد؟")
    assert any("number lost" in p for p in T.check_translation("ماذا حدث عام 1952؟", "حصل ايه في السنة دي؟"))
    assert "answer leaked" in T.check_translation("من بنى الهرم؟", "مين خوفو اللي بنى الهرم؟", answer="خوفو")
    assert T.answer_in_context("أسوان", "بني السد في مدينة أسوان جنوب مصر")
    assert not T.answer_in_context("القاهرة", "بني السد في مدينة أسوان")


def test_select_heldout_exact_and_grouped():
    recs = synth(400)
    for r in recs:
        r["question"] = r["question"].split(" كلمة")[0]
    recs[3]["question"] = recs[5]["question"]                   # a duplicate text: must stay together
    held = T.select_heldout(recs, n=100, seed=1)
    assert 100 <= len(held) <= 101 and len(set(held)) == len(held)
    assert (recs[3]["id"] in held) == (recs[5]["id"] in held)
    assert T.select_heldout(recs, n=100, seed=1) == held        # deterministic


def test_apply_review():
    items = [{"id": "a", "q_egy": "x", "answer": "y", "refs": ["y"]}, {"id": "b", "q_egy": "x2"},
             {"id": "c", "q_egy": "x3"}]
    rows = [{"row": "T001", "ok": "y", "fix": ""}, {"row": "T002", "ok": "n", "fix": "مصحح"},
            {"row": "T003", "ok": "لا", "fix": ""}, {"row": "A001", "ok": "", "fix": "جواب"}]
    m = {"T001": ("a", "q_egy"), "T002": ("b", "q_egy"), "T003": ("c", "q_egy"), "A001": ("a", "answer")}
    items, st = T.apply_review(items, rows, m)
    assert items[1]["q_egy"] == "مصحح" and items[2]["review"]["q_egy"] == "rejected"
    assert items[0]["answer"] == "جواب" and items[0]["refs"] == ["جواب"]
    assert st["translation"] == {"ok": 1, "fixed": 1, "rejected": 1, "blank": 0}


def _fixture(tmp):
    proc, cor = tmp / "processed", tmp / "corpus"
    recs = synth(120)
    for r in recs:
        r["answer"], r["refs"] = "ج", ["ج"]
    (tmp / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    write_jsonl([QAExample(id=r["id"], question=r["question"], variety="msa", complexity=None, answer="ج",
                           gold_passage_ids=[f"g{i}"], source=r["source"]) for i, r in enumerate(recs)
                 if r["source"] == "arabicaqa"], proc / "arabicaqa.jsonl")
    write_jsonl([QAExample(id=r["id"], question=r["question"], variety="msa", complexity=None, answer="ج",
                           gold_passage_ids=[f"g{i}", f"h{i}"], source=r["source"]) for i, r in enumerate(recs)
                 if r["source"] == "hotpot_ctx_ar"], proc / "hotpot_ctx_ar.jsonl")
    ex = []
    for k in range(6):
        g = f"ardqa:squad:test:{k:03d}"
        ex += [QAExample(id=f"{g}:msa", question=f"متى حدث الأمر رقم {k}؟", variety="msa", complexity=None, answer="أمس",
                         gold_passage_ids=["pa"], source="ardqa", parallel_group=g, split="test"),
               QAExample(id=f"{g}:egy", question=f"الحاجة رقم {k} حصلت امتى؟", variety="egy", complexity=None, answer="أمس",
                         gold_passage_ids=["pa"], source="ardqa", parallel_group=g, split="test")]
    write_jsonl(ex, proc / "ardqa.jsonl")
    write_jsonl([QAExample(id=f"pilot:q{k}:{v}", question=q, variety=v, complexity="C1", gold_passage_ids=["pp"],
                           source="pilot", parallel_group=f"pilot:q{k}", split="heldout")
                 for k in range(5) for v, q in (("msa", "متى بني السد العالي؟"), ("egy", "السد العالي اتبنى امتى؟"))],
                proc / "pilot.jsonl")
    cor.mkdir()
    ps = [{"id": "pa", "text": "حدث الأمر أمس"}, {"id": "pp", "text": "بني السد العالي في أسوان بين 1960 و1970"}]
    ps += [{"id": f"{c}{i}", "text": "نص"} for i in range(120) for c in "gh"]
    (cor / "passages.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in ps), encoding="utf-8")
    return recs


def test_build_and_import(tmp_path, monkeypatch):
    import arroute.llm as L
    import build_egy_testset as B
    _fixture(tmp_path)
    calls = []

    def fake(prompt, model=None, budget=None, **kw):
        calls.append(prompt[:20])
        if prompt.startswith("Rewrite"):
            q = prompt.split("Question: ")[1].split("\n")[0]
            return "هو ايه " + q.replace("؟", "") + " ده؟", {}, False
        return "بين 1960 و1970", {}, False
    monkeypatch.setattr(L, "complete_with_usage", fake)
    monkeypatch.setenv("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
    a = argparse.Namespace(stage="build", labels=str(tmp_path / "labels.jsonl"), processed=str(tmp_path / "processed"),
                           corpus=str(tmp_path / "corpus"), out=str(tmp_path / "egy"), n_heldout=40, n_review=20,
                           n_review_answers=5, seed=7, workers=2, budget_usd=1.0, price=["gpt-4.1-mini=0.40,1.60"],
                           reviewed=None)
    B.build(a)
    items = [json.loads(x) for x in (tmp_path / "egy" / "egy_test.jsonl").read_text(encoding="utf-8").splitlines()]
    sets = {s: [i for i in items if i["set"] == s] for s in ("ardqa", "pilot", "heldout")}
    assert len(sets["ardqa"]) == 6 and len(sets["pilot"]) == 5 and 40 <= len(sets["heldout"]) <= 41
    assert all(i["gold_pids"] for i in items) and all(i["answer"] for i in items)
    assert all(i["egy_markers"] for i in sets["heldout"]) and not any(i["flags"] for i in sets["pilot"])
    hot = [i for i in sets["heldout"] if i["source"] == "hotpot_ctx_ar"]
    assert hot and all(len(i["gold_pids"]) == 2 for i in hot)
    summ = (tmp_path / "egy" / "summary.txt").read_text(encoding="utf-8")
    assert "gold passages missing from the corpus: 0 | held-out questions without gold ids: 0" in summ
    held1 = json.loads((tmp_path / "egy" / "heldout_ids.json").read_text(encoding="utf-8"))
    B.build(a)                                             # re-run: the held-out list must not change
    assert json.loads((tmp_path / "egy" / "heldout_ids.json").read_text(encoding="utf-8")) == held1

    rev = tmp_path / "egy" / "review.csv"
    rows = list(csv.DictReader(rev.open(encoding="utf-8-sig")))
    assert sum(r["row"].startswith("T") for r in rows) == 20 and sum(r["row"].startswith("A") for r in rows) == 5
    for k, r in enumerate(rows):
        r["ok"] = "n" if k == 0 else "y"
        r["fix"] = "سؤال مصحح امتى؟" if k == 1 else ""
    with rev.open("w", newline="", encoding="utf-8-sig") as f:   # Excel-style: semicolons
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    a.stage = "import"
    B.do_import(a)
    items = [json.loads(x) for x in (tmp_path / "egy" / "egy_test.jsonl").read_text(encoding="utf-8").splitlines()]
    st = [i.get("review", {}).get("q_egy") for i in items if i["set"] == "heldout"]
    assert st.count("rejected") == 1 and st.count("fixed") == 1 and st.count("ok") == 18
    assert "REVIEW IMPORTED" in (tmp_path / "egy" / "summary.txt").read_text(encoding="utf-8")
