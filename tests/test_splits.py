"""Synthetic data that reproduces the leaks seen in the real build report."""
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import make_splits as ms          # noqa: E402
import shortcut_check as sc       # noqa: E402
from arroute.data import QAExample, write_jsonl   # noqa: E402


def ex(id_, q, variety="msa", cx=None, src="s", split="", gold=None, group=None, dom=""):
    return QAExample(id=id_, question=q, variety=variety, complexity=cx, source=src, split=split,
                     gold_passage_ids=[gold] if gold else [], parallel_group=group, domain=dom)


def make_P():
    rnd = Random(0)
    rag = []                                            # official split SHARES passages (the real leak)
    for i in range(60):
        rag.append(ex(f"r{i}m", f"ما هو الشيء رقم {i} في النص؟" if i % 10 == 0 else f"ما هو الشيء رقم {i} المهم؟",
                      "msa", src="arabicragb", split=rnd.choice(["train", "test"]), gold=f"p{i}"))
        rag.append(ex(f"r{i}e", f"ايه الحاجه رقم {i} بقى؟", "egy", src="arabicragb",
                      split=rnd.choice(["train", "test"]), gold=f"p{i}"))
    ara = []
    for s, n in (("train", 200), ("dev", 40), ("test", 40)):
        for i in range(n):
            ara.append(ex(f"a{s}{i}", f"سؤال قصير عن الموضوع {s}{i}؟", cx="C1", src="arabicaqa",
                          split=s, gold=f"{s}p{i}"))
    ara.append(ex("dupd", "سؤال قصير عن الموضوع train3؟", cx="C1", src="arabicaqa", split="dev", gold="x1"))
    ara.append(ex("dupt", "سؤال قصير عن الموضوع train4؟", cx="C1", src="arabicaqa", split="test", gold="x2"))
    long_q = " ".join(["كلمة"] * 25)
    maf, sub = [], []
    for g in range(40):
        for k in range(2 if g % 5 == 0 else 1):         # some groups have two items (shared original_index)
            maf.append(ex(f"m{g}_{k}", f"{long_q} {g} {k}", cx="C2", src="mafqa", group=f"mafqa:{g}"))
            sub.append(ex(f"ms{g}_{k}", f"سؤال فرعي {g} {k}؟", cx="C1", src="mafqa_sub", group=f"mafqa:{g}"))
    ard = [ex(f"d{s}{i}{v}", f"ما هو الشيء {s}{i}؟" if v == "msa" else f"ايه ده يا {s}{i}؟", v, src="ardqa",
              split=s) for s in ("dev", "test") for i in range(20) for v in ("msa", "egy")]
    pil = [ex(f"pl{i}{v}", f"سؤال بايلوت {i}؟" if v == "msa" else f"ايه ده بايلوت {i}؟", v, src="pilot",
              split="heldout") for i in range(10) for v in ("msa", "egy")]
    return {"arabicragb": rag, "arabicaqa": ara, "mafqa": maf, "mafqa_sub": sub, "ardqa": ard, "pilot": pil}


def test_splits_are_leak_free():
    P = make_P()
    d = ms.build_dialect(P, seed=1)
    c = ms.build_complexity(P, seed=1, cap_c1_train=150, cap_c1_eval=30)
    # ArabicRAGB re-split by passage: no passage on two sides (official split had many)
    sides = [{e.gold_passage_ids[0] for e in d[k]} for k in ("dialect_train", "dialect_dev", "dialect_test_ragb")]
    assert not (sides[0] & sides[1]) and not (sides[0] & sides[2]) and not (sides[1] & sides[2])
    # passage-referencing questions removed
    assert all("النص" not in e.question for e in d["dialect_train"])
    # duplicates across ArabicaQA splits removed
    train_q = {e.question for e in c["complexity_train"]}
    assert not any(e.question in train_q for e in c["complexity_dev"] + c["complexity_test"])
    # MAFQA originals and their sub-questions share one split
    where = {}
    for name, items in c.items():
        for e in items:
            if e.parallel_group:
                where.setdefault(e.parallel_group, set()).add(name)
    assert all(len(v) == 1 for v in where.values())
    assert len(c["complexity_train"]) <= 150 + 80 * 2 + 20     # cap respected (plus MAFQA share)


def test_shortcut_check_runs(tmp_path, capsys):
    P = make_P()
    out = {**ms.build_dialect(P, 1), **ms.build_complexity(P, 1, 150, 30)}
    for name, items in out.items():
        write_jsonl(items, tmp_path / f"{name}.jsonl")
    sc.complexity_check(tmp_path)
    sc.dialect_check(tmp_path)
    text = capsys.readouterr().out
    assert "length-only stump learned" in text and "dialect_test_ardqa" in text
    thr = float(text.split("more than")[1].split("words")[0])
    assert 8 < thr < 25                      # stump found the planted length gap


def test_ardqa_dev_test_duplicate_text_drops_whole_group():
    P = make_P()
    dup = "كم كمية الطماطم المهروسة؟"
    P["ardqa"] = [e for e in P["ardqa"] if not e.id.startswith(("ddev0", "dtest0"))]
    for sp in ("dev", "test"):
        for v, q in (("msa", dup), ("egy", f"ايه ده يا طماطم {sp}؟")):
            P["ardqa"].append(ex(f"dup_{sp}_{v}", q, v, src="ardqa", split=sp, group=f"g_{sp}"))
    d = ms.build_dialect(P, seed=1)                    # used to raise: LEAK (question text)
    test_ids = {e.id for e in d["dialect_test_ardqa"]}
    assert not any(i.startswith("dup_test") for i in test_ids)       # whole test group dropped
    assert {"dup_dev_msa", "dup_dev_egy"} <= {e.id for e in d["dialect_dev_ardqa"]}
    groups = {}
    for e in d["dialect_test_ardqa"]:
        groups.setdefault(e.parallel_group, set()).add(e.variety)
    assert all(len(v) == len(set(v)) for v in groups.values())


def test_controlled_views_compute_rates(capsys):
    import numpy as np
    te = ([ex(f"c1{i}", " ".join(["ك"] * 10), cx="C1", dom="fatwa") for i in range(6)] +
          [ex(f"c2{i}", " ".join(["ك"] * 12), cx="C2", dom="fatwa") for i in range(4)])
    y = np.array([0] * 6 + [1] * 4)
    p_len = np.array([0] * 6 + [1] * 4)                  # perfect
    p_ng = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0])      # 1 FP of 6 C1, 2 of 4 C2 found
    sc.controlled_views(te, y, p_len, p_ng)
    out = capsys.readouterr().out
    assert "fatwa only" in out and "9-24 words" in out
    row = [l for l in out.splitlines() if l.strip().startswith("fatwa only")][0].split()
    nums = [float(x) for x in row[-6:]]
    assert nums[:2] == [6, 4] and abs(nums[4] - 0.5) < 1e-9 and abs(nums[5] - 1 / 6) < 1e-3


def _translated(kind, label, lengths):
    return [ex(f"{kind}{i}_{n}", " ".join([f"{kind}ك{n}"] + [f"ب{i}"] * (n - 1)), cx=label, src=kind)
            for n, count in lengths.items() for i in range(count)]


def test_clean_set_is_length_matched():
    P = make_P()
    # C1 short-skewed, C2 long-skewed, overlapping in the middle (like nq_ar vs hotpot_ar)
    P["nq_ar"] = _translated("nq", "C1", {6: 80, 8: 120, 10: 60, 12: 10})
    P["hotpot_ar"] = _translated("hp", "C2", {8: 20, 10: 60, 12: 100, 15: 90})
    out = ms.build_complexity_clean(P, seed=3)
    assert set(out) == {"complexity_clean_train", "complexity_clean_dev", "complexity_clean_test"}
    allc1 = [e for v in out.values() for e in v if e.complexity == "C1"]
    allc2 = [e for v in out.values() for e in v if e.complexity == "C2"]
    h = lambda xs: sorted(len(e.question.split()) for e in xs)
    assert h(allc1) == h(allc2) and len(allc1) == 20 + 60 + 10       # only the overlapping bins survive
    assert not (set(h(allc1)) - {8, 10, 12})
    # length is now uninformative: the stump learned on this data is at chance
    assert abs(sum(e.complexity == "C2" for e in out["complexity_clean_train"])
               - sum(e.complexity == "C1" for e in out["complexity_clean_train"])) <= 1


def test_clean_set_skipped_when_not_translated():
    assert ms.build_complexity_clean(make_P(), seed=1) == {}


def test_wilson_interval_widens_for_small_n():
    lo16, hi16 = sc.wilson(14, 16)
    lo160, hi160 = sc.wilson(140, 160)
    assert (hi16 - lo16) > 2 * (hi160 - lo160) and lo16 < 0.875 < hi16


def test_shortcut_check_on_clean_set(tmp_path, capsys):
    P = make_P()
    P["nq_ar"] = _translated("nq", "C1", {8: 120, 10: 100, 12: 80})
    P["hotpot_ar"] = _translated("hp", "C2", {8: 100, 10: 100, 12: 100})
    out = {**ms.build_dialect(P, 1), **ms.build_complexity(P, 1, 150, 30),
           **ms.build_complexity_clean(P, 1)}
    for name, items in out.items():
        write_jsonl(items, tmp_path / f"{name}.jsonl")
    sc.complexity_check(tmp_path, "complexity_clean")
    text = capsys.readouterr().out
    assert "[complexity_clean]" in text and "95% interval" in text
    assert "Arabic words only" in text and "surface cues in the TRAIN split" in text


def test_clean_set_drops_questions_with_conflicting_labels():
    P = make_P()
    P["nq_ar"] = _translated("nq", "C1", {8: 40})
    P["hotpot_ar"] = _translated("hp", "C2", {8: 40})
    clash = " ".join(["مشترك"] * 8)
    P["nq_ar"].append(ex("c1x", clash, cx="C1", src="nq_ar"))
    P["hotpot_ar"].append(ex("c2x", clash, cx="C2", src="hotpot_ar"))
    out = ms.build_complexity_clean(P, seed=2)
    assert all(e.question != clash for v in out.values() for e in v)


def test_neutralize_and_cues():
    from arroute.data import neutralize, surface_cues
    assert neutralize('متى تأسست "NBA" عام 2012، في (القاهرة)؟') == "متى تأسست عام في القاهرة"
    c = surface_cues("من فاز، في مباراة ABC عام 2012")
    assert c["latin word"] and c["digits"] and c["arabic comma"] and c["no final ؟"] and not c["quote marks"]
