"""Phase 4e: train the FINAL router on all labelled questions and save it (for the joint router and the demo).

  python scripts/train_final_router.py                 # ~15 min on an RTX 3070
  python scripts/train_final_router.py --skip_torch    # only the stage-B/C models + thresholds (seconds)

Saves models/final_router/:
  text_head/      AraBERT 3-class head (stage A uses its P(C0))
  reader_pre/     AraBERT passage reader: question + C1's passages -> will C1 succeed?   (stage B input)
  reader_ans/     AraBERT answer reader: question + C1's answer + passages -> supported?  (stage C input)
  stage_b.joblib, stage_c.joblib   boosted-tree detectors ('C2 needed' / 'C2 would rescue')
  policy.json     feature lists, thresholds for budget / balanced / quality, temperatures, model names, versions
Thresholds are chosen on OUT-OF-FOLD scores (the same ones the report evaluates); the saved models are
refit on all data, so their scores are close to, not identical with, those - noted in policy.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.escalate import ANSWER_FEATURES, reader_input                       # noqa: E402
from arroute.router.metrics import fit_temperature                                      # noqa: E402
from arroute.router.outcome import (label_id, load_outcomes, make_folds,  # noqa: E402
                                    outcome_matrix, save_json, trainable)
from arroute.router.policy import RULES, search                                         # noqa: E402
from arroute.router.rerankfeats import RERANK_POST, RERANK_PRE                          # noqa: E402
from arroute.router.retfeats import FEATURES as RET_FEATURES                            # noqa: E402
from eval_final_router import build_scores                                              # noqa: E402


def fit_save(texts, y, num_labels, out_dir, a, max_len, epochs, batch, class_weights=None, select="f1", seed=0):
    from train_head import TorchBackend
    idx = np.arange(len(texts))
    rng = np.random.RandomState(seed)
    dev = np.zeros(len(texts), bool)
    dev[rng.choice(idx, max(1, int(0.1 * len(idx))), replace=False)] = True
    be = TorchBackend(a.model, max_len, epochs, a.lr, batch, num_labels=num_labels,
                      class_weights=class_weights, select=select)
    info = be.fit([texts[i] for i in idx[~dev]], y[~dev], [texts[i] for i in idx[dev]], y[dev], seed)
    T = fit_temperature(be.logits([texts[i] for i in idx[dev]]), y[dev])
    be.save(out_dir)
    return {"best_epoch": info["best_epoch"], "temperature": T, "n_train": int((~dev).sum()), "n_dev": int(dev.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--text_run", default="runs_router/bert-base-arabertv02_seed0")
    ap.add_argument("--readers", default="runs_suff")
    ap.add_argument("--pre_reader", default="bert-base-arabertv02_c1_ok_v2_seed0")
    ap.add_argument("--ans_reader", default="bert-base-arabertv02_c1_ok_ans_v2_seed0")
    ap.add_argument("--esc_feats", default="data/router_feats/escalation_feats.jsonl")
    ap.add_argument("--ret_feats", default="data/router_feats/retrieval_feats.jsonl")
    ap.add_argument("--rerank_feats", default="data/router_feats/rerank_feats.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip_torch", action="store_true")
    ap.add_argument("--exclude", default=None, help="JSON list of question ids to keep OUT (Phase 7: the Egyptian hold-out)")
    ap.add_argument("--out", default="models/final_router")
    a = ap.parse_args()

    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_outcomes(a.labels)
    excluded = set(json.loads(Path(a.exclude).read_text(encoding="utf-8"))) if a.exclude else set()
    n_all = len(recs)
    recs = [r for r in recs if r["id"] not in excluded]
    print(f"training on {len(recs):,} of {n_all:,} labelled questions ({n_all - len(recs)} held out)")
    correct, cost, calls = outcome_matrix(recs)
    fold = make_folds(recs, k=a.folds, seed=a.fold_seed)
    sa, sb, sc, text, XA, mats = build_scores(recs, fold, a)

    # thresholds on out-of-fold scores, all questions
    policy = {"stages": {"A": "P(C0) from text_head >= ta -> C0",
                         "B": "P(C2 needed) from stage_b >= tb -> C2",
                         "C": "after C1: P(C2 rescues) from stage_c >= tc -> also run C2"},
              "thresholds": {}, "created": time.strftime("%Y-%m-%d %H:%M"),
              "n_train_questions": len(recs), "excluded_ids_file": a.exclude, "n_excluded": n_all - len(recs),
              "note": "thresholds chosen on out-of-fold scores; saved models are refit on all data"}
    for rule in RULES:
        t = search(sa, sb, sc, correct, cost, calls, rule, a.tol)
        policy["thresholds"][rule] = {k: (float(v) if np.isfinite(v) else None)      # None = stage switched off
                                      for k, v in zip(("ta", "tb", "tc"), t)}
        print(f"  {rule:<9} ta={t[0]:.3f} tb={t[1]:.3f} tc={t[2]:.3f}")

    # stage B / C detectors refit on all data
    hgb = dict(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
    joblib.dump(HistGradientBoostingClassifier(**hgb).fit(mats["PRE"], mats["rescue"]), out / "stage_b.joblib")
    joblib.dump(HistGradientBoostingClassifier(**hgb).fit(mats["ALL"], mats["rescue"]), out / "stage_c.joblib")
    policy["features"] = {"stage_b": ["reader_pre_p_ok"] + RET_FEATURES + RERANK_PRE,
                          "stage_c": ["reader_pre_p_ok"] + RET_FEATURES + RERANK_PRE + ANSWER_FEATURES + RERANK_POST
                          + ["reader_ans_p_ok"]}
    policy["models"] = {"base": a.model, "reranker": "BAAI/bge-reranker-v2-m3", "dense": "BAAI/bge-m3",
                        "retrieval": "hybrid BM25 + BGE-M3, reciprocal rank fusion k=60, top-5 for C1, top-20 for the reranker",
                        "generator": "gpt-4.1-mini (labels)", "reader_words_per_passage": 60}
    policy["versions"] = {"sklearn": sklearn.__version__}

    if not a.skip_torch:
        import torch
        import transformers
        policy["versions"].update(torch=torch.__version__, transformers=transformers.__version__)
        from arroute.retrieval.index import load_corpus
        ids, ptexts, _ = load_corpus(a.corpus)
        pt = dict(zip(ids, ptexts))
        tr = np.array([trainable(r) for r in recs])
        y3 = np.array([label_id(r) for r, t in zip(recs, tr) if t])
        print("\n[1/3] text head (3 classes, trainable questions)")
        policy["text_head"] = fit_save([r["question"] for r, t in zip(recs, tr) if t], y3, 3, out / "text_head", a,
                                       max_len=128, epochs=4, batch=16)
        y_ok = correct[:, 1].astype(int)
        cw = [len(y_ok) / (2 * max(1, (y_ok == 0).sum())), len(y_ok) / (2 * max(1, (y_ok == 1).sum()))]

        def rin(r, with_answer):
            q = r["question"]
            if with_answer:
                q = f"{q} [SEP] الجواب: {(r['results']['C1'].get('answer') or '').strip()}"
            return reader_input(q, [pt.get(p, "") for p in r["results"]["C1"].get("passages") or []], 60)
        print("\n[2/3] passage reader (before C1)")
        policy["reader_pre"] = fit_save([rin(r, False) for r in recs], y_ok, 2, out / "reader_pre", a, 512, 6, a.batch,
                                        cw, "auc")
        print("\n[3/3] answer reader (after C1)")
        policy["reader_ans"] = fit_save([rin(r, True) for r in recs], y_ok, 2, out / "reader_ans", a, 512, 6, a.batch,
                                        cw, "auc")
    policy["seconds"] = round(time.time() - t0, 1)
    save_json(policy, out / "policy.json")
    print(f"\nsaved {out} ({policy['seconds']}s)")
    print(json.dumps(policy["thresholds"], indent=1))


if __name__ == "__main__":
    main()
