"""Phase 1, step 3: how much of each task can be solved with SHORTCUTS?

Cheap classical models set the floor your AraBERT router must clearly beat:
  complexity: length-only stump (one word-count threshold), char n-gram LR
  dialect   : marker-word heuristic (own list, in the style of CollectivIA), char n-gram LR
If the length-only stump already scores high, the task as labelled is partly a length detector.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles/redirects default to a legacy code page
from arroute.data import neutralize, read_jsonl, surface_cues  # noqa: E402
from arroute.router.metrics import wilson  # noqa: E402,F401

EGY_MARKERS = ["ايه", "إيه", "دلوقتي", "دلوقت", "ازاي", "إزاي", "ليه", "كده", "ده", "دي", "دول",
               "مش", "عشان", "علشان", "بتاع", "بتاعت", "فين", "امتى", "إمتى", "كام", "مين", "عايز",
               "عاوز", "اللي", "بقى", "لسه", "برضه", "دا", "هل ممكن", "حاجة"]


def nwords(e):
    return len(e.question.split())


def ngram_model():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                         max_features=200_000, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def marker_predict(exs):
    return np.array([int(any(m in e.question.split() for m in EGY_MARKERS)) for e in exs])


def complexity_check(sd, prefix="complexity"):
    tr = [e for e in read_jsonl(sd / f"{prefix}_train.jsonl") if e.complexity in ("C1", "C2")]
    te = [e for e in read_jsonl(sd / f"{prefix}_test.jsonl") if e.complexity in ("C1", "C2")]
    y = lambda ex: np.array([int(e.complexity == "C2") for e in ex])
    ytr, yte = y(tr), y(te)
    print(f"\nCOMPLEXITY [{prefix}]  (C1 vs C2)   train {len(tr):,} (C2 {ytr.sum()}) | test {len(te):,} (C2 {yte.sum()})")
    stump = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)
    stump.fit(np.array([[nwords(e)] for e in tr]), ytr)
    thr = stump.tree_.threshold[0]
    p_len = stump.predict(np.array([[nwords(e)] for e in te]))
    ng = ngram_model().fit([e.question for e in tr], ytr)
    p_ng = ng.predict([e.question for e in te])
    ng_neu = ngram_model().fit([neutralize(e.question) for e in tr], ytr)
    p_neu = ng_neu.predict([neutralize(e.question) for e in te])
    p_maj = np.zeros_like(yte)
    print("  surface cues in the TRAIN split (% of questions)      C1      C2")
    for cue in surface_cues("").keys():
        r = [100 * np.mean([surface_cues(e.question)[cue] for e in tr if e.complexity == c]) for c in ("C1", "C2")]
        print(f"    {cue:<30}{r[0]:>25.1f}{r[1]:>8.1f}")
    print(f"  length-only stump learned: 'C2 if more than {thr:.1f} words'")
    print(f"  {'model':<30}{'accuracy':>9}{'bal.acc':>9}{'macroF1':>9}{'C2 recall':>10}")
    for name, p in (("majority class", p_maj), ("length-only stump", p_len), ("char n-gram LR", p_ng),
                   ("n-gram LR, Arabic words only", p_neu)):
        print(f"  {name:<30}{accuracy_score(yte, p):>9.3f}{balanced_accuracy_score(yte, p):>9.3f}"
              f"{f1_score(yte, p, average='macro'):>9.3f}{recall_score(yte, p, zero_division=0):>10.3f}")
    print("  by question length (is the model right where lengths overlap?):")
    print(f"  {'words':<8}{'#C1':>6}{'#C2':>6}{'len-acc':>9}{'ngram-acc':>10}")
    w = np.array([nwords(e) for e in te])
    for lo, hi in ((0, 8), (9, 14), (15, 24), (25, 10_000)):
        m = (w >= lo) & (w <= hi)
        if m.sum():
            print(f"  {str(lo) + '-' + (str(hi) if hi < 10_000 else '+'):<8}{int((yte[m] == 0).sum()):>6}"
                  f"{int((yte[m] == 1).sum()):>6}{accuracy_score(yte[m], p_len[m]):>9.3f}"
                  f"{accuracy_score(yte[m], p_ng[m]):>10.3f}")
    lo, hi = wilson(int(((p_ng == 1) & (yte == 1)).sum()), int((yte == 1).sum()))
    print(f"  n-gram C2 recall 95% interval on this test set: [{lo:.2f}, {hi:.2f}]  (C2 n = {int(yte.sum())})")
    controlled_views(te, yte, p_len, p_ng)
    return {"n_test": int(len(yte)), "n_c2": int(yte.sum()),
            "length_bal_acc": float(balanced_accuracy_score(yte, p_len)),
            "ngram_bal_acc": float(balanced_accuracy_score(yte, p_ng)),
            "words_bal_acc": float(balanced_accuracy_score(yte, p_neu)),
            "ngram_c2_recall": float(recall_score(yte, p_ng, zero_division=0))}


def controlled_views(te, yte, p_len, p_ng):
    """Two views where the cheap shortcuts are switched off, so the numbers mean more."""
    w = np.array([nwords(e) for e in te])
    fat = np.array([e.domain == "fatwa" for e in te])
    zone = (w >= 9) & (w <= 24)
    print("  controlled views (small n => wide error bars):")
    print(f"  {'view':<30}{'#C1':>6}{'#C2':>6}{'len C2rec':>11}{'len FPR':>9}{'ngram C2rec':>13}{'ngram FPR':>11}")
    for name, m in (("fatwa only (domain fixed)", fat), ("9-24 words (length overlap)", zone)):
        y = yte[m]
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            print(f"  {name:<30}  not enough of both classes")
            continue
        rec = lambda p: (p[m][y == 1] == 1).mean()
        fpr = lambda p: (p[m][y == 0] == 1).mean()
        print(f"  {name:<30}{int((y == 0).sum()):>6}{int((y == 1).sum()):>6}{rec(p_len):>11.3f}{fpr(p_len):>9.3f}"
              f"{rec(p_ng):>13.3f}{fpr(p_ng):>11.3f}")


def _fit_predict(tr, te):
    """The three cheap complexity models trained on `tr`, predictions on `te` (C1=0, C2=1)."""
    ytr = np.array([int(e.complexity == "C2") for e in tr])
    stump = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)
    stump.fit(np.array([[nwords(e)] for e in tr]), ytr)
    out = {"length-only stump": stump.predict(np.array([[nwords(e)] for e in te])),
           "char n-gram LR": ngram_model().fit([e.question for e in tr], ytr).predict([e.question for e in te]),
           "n-gram LR, Arabic words only": ngram_model().fit([neutralize(e.question) for e in tr], ytr)
           .predict([neutralize(e.question) for e in te])}
    return out


def transfer_check(sd):
    """Train on one complexity set, test on ANOTHER. If the cheap models also collapse, the problem is the
    data (sources differ in more than complexity), not the transformer."""
    need = ["complexity_clean_train", "complexity_clean_test", "complexity_train", "complexity_test"]
    if not all((sd / f"{n}.jsonl").exists() for n in need):
        return {}
    keep = lambda ex: [e for e in ex if e.complexity in ("C1", "C2")]
    rd = lambda n: keep(read_jsonl(sd / f"{n}.jsonl"))
    clean_tr, clean_te, leg_tr, leg_te = rd("complexity_clean_train"), rd("complexity_clean_test"), rd("complexity_train"), rd("complexity_test")
    print("\nTRANSFER  (train on one complexity set, test on another)")
    print(f"  {'train -> test':<20}{'model':<30}{'bal.acc':>8}{'C2 rec':>8}{'FPR':>7}   share predicted C2 by source (legacy test only)")
    res = {}
    for label, tr, te in (("clean -> legacy", clean_tr, leg_te), ("legacy -> clean", leg_tr, clean_te)):
        y = np.array([int(e.complexity == "C2") for e in te])
        for name, p in _fit_predict(tr, te).items():
            rec, fpr = float((p[y == 1] == 1).mean()), float((p[y == 0] == 1).mean())
            bal = float(balanced_accuracy_score(y, p))
            srcs = ""
            if label.endswith("legacy"):
                by = {s_: float(np.mean(p[[e.source == s_ for e in te]])) for s_ in sorted({e.source for e in te})}
                srcs = "  " + "  ".join(f"{k} {v:.2f}" for k, v in by.items())
            print(f"  {label:<20}{name:<30}{bal:>8.3f}{rec:>8.3f}{fpr:>7.3f}{srcs}")
            res[f"{label} | {name}"] = {"bal_acc": bal, "c2_recall": rec, "fpr": fpr}
    print("  (compare with the transformer rows in summary_complexity.txt: same legacy test, trained on clean)")
    return res


def dialect_check(sd):
    keep = lambda ex: [e for e in ex if e.variety in ("msa", "egy")]
    tr = keep(read_jsonl(sd / "dialect_train.jsonl"))
    y = lambda ex: np.array([int(e.variety == "egy") for e in ex])
    ng = ngram_model().fit([e.question for e in tr], y(tr))
    print(f"\nDIALECT  (MSA vs Egyptian)   trained on ArabicRAGB: {len(tr):,} (Egyptian {y(tr).sum():,})")
    print(f"  {'evaluation set':<24}{'n':>6}   {'--- marker heuristic ---':<26}{'--- char n-gram LR ---'}")
    print(f"  {'':<24}{'':>6}   {'acc':>7}{'F1(egy)':>9}{'':>10}{'acc':>7}{'F1(egy)':>9}")
    ref = {}
    for name in ("dialect_dev", "dialect_test_ragb", "dialect_dev_ardqa", "dialect_test_ardqa", "dialect_test_pilot"):
        ex = keep(read_jsonl(sd / f"{name}.jsonl"))
        if not ex:
            continue
        yt = y(ex)
        ph, pn = marker_predict(ex), ng.predict([e.question for e in ex])
        ref[name] = {"n": len(ex), "heuristic_acc": float(accuracy_score(yt, ph)),
                     "heuristic_f1": float(f1_score(yt, ph, zero_division=0)),
                     "ngram_acc": float(accuracy_score(yt, pn)), "ngram_f1": float(f1_score(yt, pn))}
        tag = ("  <- in-distribution" if "ragb" in name or name == "dialect_dev" else
               "  <- EXTERNAL (LLM-written, native-validated)" if "pilot" in name else
               "  <- EXTERNAL (human-translated)")
        print(f"  {name:<24}{len(ex):>6}   {accuracy_score(yt, ph):>7.3f}{f1_score(yt, ph, zero_division=0):>9.3f}"
              f"{'':>10}{accuracy_score(yt, pn):>7.3f}{f1_score(yt, pn):>9.3f}{tag}")
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    sd = Path(ap.parse_args().splits)
    base = {"complexity": complexity_check(sd)}
    if (sd / "complexity_clean_train.jsonl").exists():
        print("\n--- CLEAN set: translated, same domain, length-matched (stump should be near chance) ---")
        base["complexity_clean"] = complexity_check(sd, "complexity_clean")
    base["transfer"] = transfer_check(sd)
    base["dialect"] = dialect_check(sd)
    (sd / "baselines.json").write_text(json.dumps(base, indent=1), encoding="utf-8")   # read by summarize_runs.py
    print("\nHow to read this: a big gap between in-distribution and EXTERNAL dialect scores means the "
          "classifier learned ArabicRAGB's generation style, not dialect. A high length-only score on "
          "complexity means the labels are partly a length detector.")


if __name__ == "__main__":
    main()
