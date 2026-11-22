import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import translate_questions as tq   # noqa: E402
from arroute.data import LOADERS    # noqa: E402

ROWS = [(f"Which team won the match number {i} in the league?", {"type": "bridge", "level": "hard"})
        for i in range(30)] + [("Too short", {}), ("Which team won the match number 3 in the league?", {})]


def good_fake(prompt):
    qs = json.loads(prompt.split("Questions:\n")[1])
    return json.dumps([f"من فاز بالمباراة {q.split()[-5]} في الدوري؟" for q in qs], ensure_ascii=False)


def test_parse_and_arabic_check():
    assert tq.parse_json_list('```json\n["أ", "ب"]\n```') == ["أ", "ب"]
    assert tq.looks_arabic("من فاز؟") and not tq.looks_arabic("Who won?")


def test_sampling_dedupes_and_filters():
    s = tq.sample_english(ROWS, 100, seed=1)
    assert len(s) == 30                                  # duplicate + too-short removed


def test_bad_batch_falls_back_to_single_calls():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        qs = json.loads(prompt.split("Questions:\n")[1])
        if len(qs) > 1:
            return '["only one item"]'                   # wrong count -> whole batch rejected
        return good_fake(prompt)
    qs = [q for q, _ in ROWS[:5]]
    done, failed = tq.translate_all(qs, flaky, batch=5)
    assert len(done) == 5 and not failed and calls["n"] == 1 + 5


def test_unusable_output_is_reported_not_hidden():
    done, failed = tq.translate_all(["Who won?"], lambda p: '["Who won?"]', batch=1)   # not Arabic
    assert failed == ["Who won?"] and not done


def test_end_to_end_and_loader(tmp_path, capsys):
    out = tmp_path / "raw" / "translated"
    rows = tq.main(["--kind", "hotpot", "--n", "12", "--out", str(out), "--batch", "5"],
                   complete_fn=good_fake, rows=ROWS)
    assert len(rows) == 12 and (out / "hotpot_review.csv").exists()
    ex, _ = LOADERS["hotpot_ar"](tmp_path / "raw")
    assert len(ex) == 12 and {e.complexity for e in ex} == {"C2"} and ex[0].extra["question_en"]
    tq.main(["--kind", "nq", "--n", "6", "--out", str(out)], complete_fn=good_fake, rows=ROWS)
    ex, _ = LOADERS["nq_ar"](tmp_path / "raw")
    assert {e.complexity for e in ex} == {"C1"} and ex[0].source == "nq_ar"


def test_missing_translated_file_is_a_skip_not_a_crash(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        LOADERS["hotpot_ar"](tmp_path)
