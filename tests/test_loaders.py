"""Fixtures copy the REAL structures seen in inspect_report.txt."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arroute.data import LOADERS, map_dialect  # noqa: E402

AR = "ar"


def _w(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def make_raw(tmp):
    raw = Path(tmp)
    # ArabicaQA MRC: trailing \n in question, one impossible question, answers w/ document_id
    para = {"context": "تم بناء السد العالي في أسوان.", "qas": [
        {"question": "متى بني السد؟\n", "id": 1, "is_impossible": False,
         "answers": [{"text": "أسوان", "document_id": 9, "answer_start": 0}]},
        {"question": "سؤال لا جواب له؟", "id": 2, "is_impossible": True, "answers": []},
        {"question": " \n", "id": 3, "is_impossible": False, "answers": [{"text": "x"}]}]}
    for f in ("train.json", "val.json", "test.json"):
        _w(raw / "ArabicaQA/MRC" / f, {"data": [{"paragraphs": [para]}]})
    # ArDQA: 25 files per split; Narratives-style: the SAME qa id repeats in 2 paragraphs
    for split in ("dev", "test"):
        for cv in ("egy", "glf", "lev", "mgr", "msa"):
            for qv in ("egy", "glf", "lev", "mgr", "msa"):
                paras = [{"context": f"سياق {k}", "qas": [
                    {"question": f"سؤال {k} {qv}؟", "id": "story_1",
                     "answers": [{"text": "x", "answer_start": 0}]}]} for k in (1, 2)]
                if split == "dev" and cv == "msa" and qv == "glf":
                    paras = paras[:1]            # this file lacks item 2: sizes differ, as in real data
                _w(raw / f"ArDQA/ArDQA-Narratives/{split}/N-{split}-context-{cv}-questions-{qv}.json",
                   {"data": [{"title": "story", "paragraphs": paras}]})
    # ArabicRAGB (HF on disk)
    from datasets import Dataset, DatasetDict
    mk = lambda: Dataset.from_dict({
        "id": ["a", "b"], "query": ["سؤال أ", "سؤال ب"], "query_dialect": ["Egyptian", "MSA"],
        "query_complexity": ["simple", "multi_hop"], "passage_id": ["1", "2"],
        "passage_title": ["ت1", "ت2"], "passage_text": ["نص أ", "نص ب"],
        "source_url": ["u", "u"], "source_category": ["c", "c"], "split": ["train", "train"]})
    DatasetDict({"train": mk(), "test": mk()}).save_to_disk(str(raw / "ArabicRAGB"))
    # MAFQA
    _w(raw / "MAFQA/Dataset-MAFQA.json", [
        {"original_index": 7, "context": "نص الفتوى", "question": "ما حكم كذا؟",
         "decomposition": {"p_a": "مقطع أ", "p_b": "مقطع ب", "sq1": "س1؟", "sq2": "س2؟", "sq3": "ما الدليل؟", "sa1": "ج1", "sa2": "ج2", "sa3": "د"}, "a_gen": "الجواب"},
        {"original_index": 7, "context": "نص الفتوى", "question": "وما حكم كذا أيضا؟",
         "decomposition": {"p_a": "مقطع ج", "p_b": "مقطع د", "sq1": "{س3؟}", "sq2": "", "sa1": "ج3", "sa2": "", "sa3": ""}, "a_gen": "الجواب 2"}])
    # pilot
    _w(raw / "pilot/pilot_dataset_full200.json", [{"id": "q001", "question_msa": "متى؟",
        "question_egy": "امتى؟", "context": "نص"}])
    return raw


def test_all_loaders(tmp_path):
    raw = make_raw(tmp_path)
    OPTIONAL = {"hotpot_ar", "nq_ar", "hotpot_ctx_ar"}          # need translated files; covered in test_translate.py
    res = {n: fn(raw) for n, fn in LOADERS.items() if n not in OPTIONAL}

    ex, ps = res["arabicaqa"]
    assert len(ex) == 3                                   # impossible + blank questions dropped
    assert all(not e.question.endswith("\n") for e in ex)
    assert all(g in ps for e in ex for g in e.gold_passage_ids)
    assert {e.split for e in ex} == {"train", "dev", "test"}

    ex, ps = res["ardqa"]
    assert len(ex) == 9 + 10             # dev: item 2 lacks 'glf'; test: both items complete
    groups = {}
    for e in ex:
        groups.setdefault(e.parallel_group, set()).add(e.variety)
    assert len(groups) == 4              # repeated ids did NOT collapse; the missing item did not break alignment
    assert sorted(len(v) for v in groups.values()) == [4, 5, 5, 5]
    assert all(g in ps for e in ex for g in e.gold_passage_ids)
    assert {e.extra["ardqa_id"] for e in ex} == {"story_1"}

    ex, ps_rag = res["arabicragb"]
    assert [e.variety for e in ex if e.split == "train"] == ["egy", "msa"]
    assert ps_rag and all(g in ps_rag for e in ex for g in e.gold_passage_ids)
    assert ex[1].extra["ragb_complexity"] == "multi_hop" and ex[1].complexity is None

    ex, _ = res["mafqa"]
    assert len(ex) == 2 and len({e.id for e in ex}) == 2           # shared original_index -> unique ids
    assert {e.parallel_group for e in ex} == {"mafqa:7"}           # but grouped for splitting
    assert ex[0].complexity == "C2" and ex[0].domain == "fatwa" and ex[0].extra["decomposition"]

    ex, ps = res["mafqa_sub"]
    assert [e.question for e in ex] == ["س1؟", "س2؟", "س3؟"]            # sq3 (both passages) excluded,
                                                                        # empty sq2 skipped, {braces} stripped
    assert all(e.complexity == "C1" and e.domain == "fatwa" for e in ex)
    assert {e.parallel_group for e in ex} == {"mafqa:7"}
    assert all(g in ps for e in ex for g in e.gold_passage_ids) and ex[0].gold_passage_ids != ex[1].gold_passage_ids

    ex, _ = res["pilot"]
    assert {e.variety for e in ex} == {"msa", "egy"} and ex[0].parallel_group == "pilot:q001"


def test_dialect_map():
    assert map_dialect("Egyptian") == "egy" and map_dialect("Gulf Arabic") == "glf"
    assert map_dialect("Levantine") == "lev" and map_dialect("Maghrebi") == "mgr"
    assert map_dialect("MSA") == "msa" and map_dialect("klingon") is None


def test_ardqa_misalignment_is_detected(tmp_path):
    raw = make_raw(tmp_path)
    for qv in ("egy", "lev", "mgr"):          # three varieties describe totally different items
        bad = raw / f"ArDQA/ArDQA-Narratives/dev/N-dev-context-msa-questions-{qv}.json"
        d = json.loads(bad.read_text(encoding="utf-8"))
        for para in d["data"][0]["paragraphs"]:
            para["qas"][0]["id"] = f"other_{qv}"
        bad.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="coverage"):
        LOADERS["ardqa"](raw)
