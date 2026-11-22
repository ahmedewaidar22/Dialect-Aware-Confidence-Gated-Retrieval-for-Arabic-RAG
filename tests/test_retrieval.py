"""Retrieval + HotpotQA-context tests. A hashing 'encoder' stands in for BGE-M3 (no torch needed)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arroute.data import LOADERS, QAExample, Passage, write_jsonl   # noqa: E402
from arroute.retrieval.index import BM25Index, DenseIndex, Retriever, rrf   # noqa: E402
from arroute.retrieval.text import tokenize_ar                     # noqa: E402
import translate_hotpot_context as th                               # noqa: E402
import build_corpus as bc                                           # noqa: E402
import retrieval_check as rc                                        # noqa: E402


def fake_encode(texts, show=False):
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize_ar(t):
            for j in range(len(w) - 1):
                out[i, hash(w[j:j + 2]) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_tokenizer_drops_arabic_punctuation():                    # the pilot bug
    assert tokenize_ar("متى تم بناء السد العالي؟") == ["متي", "تم", "بناء", "السد", "العالي"]
    assert tokenize_ar("عاصمة مصر، وأكبر مدنها؛") == ["عاصمه", "مصر", "واكبر", "مدنها"]


def test_bm25_dense_hybrid():
    ids = ["a", "b", "c"]
    texts = ["تم بناء السد العالي في أسوان", "القاهرة عاصمة مصر", "نهر النيل أطول نهر"]
    r = Retriever(BM25Index(ids, texts), DenseIndex(ids, fake_encode(texts), fake_encode))
    out = r.all_methods("متى تم بناء السد العالي؟", k=3)
    assert out["bm25"][0] == "a" and out["dense"][0] == "a" and out["hybrid"][0] == "a"
    assert rrf([["x", "y"], ["y", "x"]])[:2] in (["x", "y"], ["y", "x"])


ROW_DICT = {"id": "h1", "question": "Which magazine started first?", "answer": "Arthur's Magazine",
            "type": "comparison", "level": "medium",
            "supporting_facts": {"title": ["Arthur's Magazine", "First for Women"], "sent_id": [0, 0]},
            "context": {"title": ["Arthur's Magazine", "First for Women", "Other"],
                        "sentences": [["Arthur's Magazine was founded in 1844."], ["First for Women began in 1989."],
                                      ["Unrelated text."]]}}
ROW_LIST = {"id": "h2", "question": "Are both X and Y cities?", "answer": "yes", "type": "comparison",
            "level": "hard", "supporting_facts": [["X", 0], ["Y", 0]],
            "context": [["X", ["X is a city."]], ["Y", ["Y is a city."]], ["Arthur's Magazine", ["Arthur's Magazine was founded in 1844."]]]}


def fake_translate(prompt):
    item = json.loads(prompt.split("Item:\n", 1)[1].split("\n\n(Second")[0])
    return json.dumps({"question": "سؤال مترجم؟", "answer": "إجابة",
                       "paragraphs": [{"title": "عنوان", "text": "نص عربي مترجم بالكامل لهذه الفقرة رقم " + str(i)}
                                      for i, _ in enumerate(item["paragraphs"])]}, ensure_ascii=False)


def test_item_parsing_both_formats():
    a, b = th.item_from_row(ROW_DICT), th.item_from_row(ROW_LIST)
    assert [p["gold"] for p in a["paragraphs"]] == [True, True, False]
    assert [p["gold"] for p in b["paragraphs"]] == [True, True, False]


def test_bad_output_retries_then_reports_failure():
    calls = {"n": 0}

    def bad_then_good(prompt):
        calls["n"] += 1
        return '{"question": "x"}' if calls["n"] == 1 else fake_translate(prompt)
    assert th.translate_item(th.item_from_row(ROW_DICT), bad_then_good) and calls["n"] == 2
    assert th.translate_item(th.item_from_row(ROW_DICT), lambda p: "not json") is None


def test_translate_loader_and_shared_paragraph_dedupe(tmp_path):
    out = tmp_path / "raw" / "translated"
    recs = th.main(["--n", "5", "--out", str(out), "--workers", "2"], complete_fn=fake_translate,
                   rows=[ROW_DICT, ROW_LIST])
    assert len(recs) == 2 and recs[1]["answer_ar"] == "نعم"            # yes/no mapped deterministically
    assert (out / "hotpot_ctx_trial.txt").exists() and (out / "hotpot_ctx_review.csv").exists()
    exs, ps = LOADERS["hotpot_ctx_ar"](tmp_path / "raw")
    assert len(exs) == 2 and all(len(e.gold_passage_ids) == 2 for e in exs)
    assert len(ps) == 5                                 # 6 paragraphs, one English paragraph shared -> 5 passages


def _processed(tmp):
    P = Path(tmp) / "processed"
    ctx = {"p1": "تم بناء السد العالي في أسوان بين عامي 1960 و1970", "p2": "القاهرة عاصمة مصر وأكبر مدنها",
           "p3": "يبلغ طول نهر النيل حوالي 6650 كيلومترا", "p4": "الفتوى في حكم الزكاة على المال المدخر"}
    write_jsonl([Passage(k, v, "x") for k, v in ctx.items() if k in ("p2", "p3")], P / "arabicaqa_passages.jsonl")
    write_jsonl([Passage("p1", ctx["p1"], "x")], P / "pilot_passages.jsonl")
    write_jsonl([Passage("p1", ctx["p1"], "x")], P / "ardqa_passages.jsonl")      # duplicate id across sources
    write_jsonl([Passage("p4", ctx["p4"], "x")], P / "mafqa_passages.jsonl")
    write_jsonl([QAExample("a1", "ما هي عاصمة مصر؟", "msa", "C1", gold_passage_ids=["p2"], split="test", source="arabicaqa"),
                 QAExample("a2", "كم يبلغ طول نهر النيل؟", "msa", "C1", gold_passage_ids=["p3"], split="test", source="arabicaqa")],
                P / "arabicaqa.jsonl")
    write_jsonl([QAExample("pm", "متى تم بناء السد العالي؟", "msa", "C1", gold_passage_ids=["p1"], source="pilot", parallel_group="g"),
                 QAExample("pe", "امتى السد العالي اتبنى؟", "egy", "C1", gold_passage_ids=["p1"], source="pilot", parallel_group="g")],
                P / "pilot.jsonl")
    write_jsonl([QAExample("dm", "متى تم بناء السد؟", "msa", None, gold_passage_ids=["p1"], split="test", source="ardqa", parallel_group="d"),
                 QAExample("de", "امتى اتبنى السد؟", "egy", None, gold_passage_ids=["p1"], split="test", source="ardqa", parallel_group="d")],
                P / "ardqa.jsonl")
    write_jsonl([QAExample("m1", "ما حكم الزكاة على المال المدخر؟", "msa", "C2", gold_passage_ids=["p4"], source="mafqa")],
                P / "mafqa.jsonl")
    return P


def test_build_corpus_and_retrieval_check_end_to_end(tmp_path, capsys):
    P = _processed(tmp_path)
    th.main(["--n", "2", "--out", str(tmp_path / "raw" / "translated")], complete_fn=fake_translate,
            rows=[ROW_DICT, ROW_LIST])
    bc.main(["--raw", str(tmp_path / "raw"), "--processed", str(P), "--out", str(tmp_path / "corpus"), "--dense"],
            encode=fake_encode)
    stats = (tmp_path / "corpus" / "corpus_stats.txt").read_text(encoding="utf-8")
    assert "corpus: 9 passages" in stats                 # 4 unique processed + 5 hotpot; duplicate p1 kept once
    assert np.load(tmp_path / "corpus" / "emb_bgem3.npy").shape[0] == 9
    res = rc.main(["--corpus", str(tmp_path / "corpus"), "--processed", str(P)], encode=fake_encode)
    text = capsys.readouterr().out
    assert {"arabicaqa_test", "pilot_msa", "pilot_egy", "ardqa_msa", "ardqa_egy", "mafqa", "hotpot_ctx"} <= set(res)
    assert res["arabicaqa_test"]["bm25"][0][1] == 1.0     # both easy questions found at rank 1
    assert "DIALECT GAP" in text and "hybrid" in text and "(p=" in text


def test_leak_is_detected_and_retried():
    item = {"question": "Which actress from Morris from America also appeared in Wetlands?", "answer": "Carla Juri",
            "paragraphs": [{"title": "Morris", "text": "Morris from America is a film.", "gold": True}]}
    leaked = '{"question": "من هي الممثلة كارلا جوري من فيلم موريس من أمريكا؟", "answer": "كارلا جوري", "paragraphs": [{"title": "موريس", "text": "موريس من أمريكا فيلم كوميدي"}]}'
    clean = '{"question": "من هي الممثلة من فيلم موريس من أمريكا التي ظهرت في ويتلاندز؟", "answer": "كارلا جوري", "paragraphs": [{"title": "موريس", "text": "موريس من أمريكا فيلم كوميدي"}]}'
    assert th.leaks_answer(item["question"], item["answer"], "من هي الممثلة كارلا جوري؟", "كارلا جوري")
    assert not th.leaks_answer("Is X older than Carla Juri?", "Carla Juri", "هل كارلا جوري أكبر؟", "كارلا جوري")
    assert not th.leaks_answer("Are both cities?", "yes", "هل كلتاهما مدينتان؟ نعم", "نعم")
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return leaked if len(calls) == 1 else clean
    out = th.translate_item(item, fake)
    assert out and "كارلا" not in out["question"] and "must NOT contain" in calls[1]
    assert th.translate_item(item, lambda p: leaked) is None          # still leaking after the retry -> dropped


def test_mcnemar_exact_values():
    assert rc.mcnemar_exact(0, 0) == 1.0 and rc.mcnemar_exact(7, 7) == 1.0
    assert abs(rc.mcnemar_exact(10, 0) - 2 * 0.5 ** 10) < 1e-12
    assert rc.mcnemar_exact(30, 10) < 0.01 < rc.mcnemar_exact(12, 8)


def test_check_report_counts_leaks(tmp_path, capsys):
    import check_hotpot_ctx as ch
    rec = {"id": "x", "model": "m", "question_en": "Which actress appeared in Wetlands?", "answer_en": "Carla Juri",
           "question_ar": "من هي الممثلة كارلا جوري؟", "answer_ar": "كارلا جوري",
           "paragraphs": [{"gold": True, "title_en": "a", "text_en": "a", "title_ar": "أ", "text_ar": "أ"}]}
    f = tmp_path / "h.jsonl"
    f.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert ch.main(str(f)) == 1 and "LEAK" in capsys.readouterr().out
