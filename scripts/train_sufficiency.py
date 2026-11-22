"""Phase 4c step 2: the PASSAGE READER - will C1 succeed with the passages it retrieved?

AraBERT reads "question [SEP] passage 1 [SEP] ... passage 5" (the exact passages C1 used) and predicts
whether C1's answer will be correct. Same 5 folds as the router, so its out-of-fold predictions can be
combined with the other signals. Local and free.

  python scripts/train_sufficiency.py --backend tfidf                         # baseline, CPU, seconds
  python scripts/train_sufficiency.py --backend torch --seed 0                # AraBERT, ~15 min on an RTX 3070
  python scripts/train_sufficiency.py --backend torch --seed 0 --target rescue   # predict 'C2 would fix it'

Writes runs_suff/<name>/{oof.npz, config.json, folds.json}.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.escalate import reader_input                               # noqa: E402
from arroute.router.metrics import fit_temperature, softmax                     # noqa: E402
from arroute.router.outcome import inner_dev_split, load_outcomes, make_folds, save_json  # noqa: E402
from arroute.router.route_report import auc                                     # noqa: E402


def targets(recs, target):
    c1 = np.array([r["results"]["C1"]["correct"] for r in recs], bool)
    c2 = np.array([r["results"]["C2"]["correct"] for r in recs], bool)
    return (c1 if target == "c1_ok" else (~c1 & c2)).astype(int)


class TfidfBackend:
    """Word n-gram TF-IDF over question + passages, logistic regression. The 'does content help at all' baseline."""

    def fit(self, texts, y, dev_texts, dev_y, seed):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        self.clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)
        self.clf.fit(self.vec.fit_transform(texts), y)
        return {"best_epoch": 0, "history": []}

    def logits(self, texts):
        lp = self.clf.predict_log_proba(self.vec.transform(texts))
        out = np.full((len(texts), 2), -30.0)
        out[:, self.clf.classes_] = lp
        return out


def run_cv(recs, texts, y, backend, k, fold_seed, seed, log=print):
    n = len(recs)
    fold = make_folds(recs, k=k, seed=fold_seed)
    logits, T, info = np.zeros((n, 2), np.float32), np.zeros(n), []
    for f in range(k):
        test = np.where(fold == f)[0]
        fit_i, dev_i = inner_dev_split(np.where(fold != f)[0], recs, seed=seed + 100 * f)
        t0 = time.time()
        fi = backend.fit([texts[i] for i in fit_i], y[fit_i], [texts[i] for i in dev_i], y[dev_i], seed + f)
        Tf = fit_temperature(backend.logits([texts[i] for i in dev_i]), y[dev_i])
        logits[test] = backend.logits([texts[i] for i in test])
        T[test] = Tf
        p = softmax(logits[test], Tf)[:, 1]
        a = auc(p, y[test])
        info.append({"fold": f, "n_fit": len(fit_i), "n_dev": len(dev_i), "n_test": len(test), "T": Tf, "auc": a,
                     "best_epoch": fi["best_epoch"], "history": fi["history"], "seconds": round(time.time() - t0, 1)})
        log(f"  fold {f}: fit {len(fit_i)} | test {len(test)} | T={Tf:.2f} | test AUC {a:.3f} | {info[-1]['seconds']}s")
    p = np.array([softmax(logits[i:i + 1], T[i])[0, 1] for i in range(n)])
    return {"fold": fold, "logits": logits, "T": T, "p": p, "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--backend", choices=["tfidf", "torch"], required=True)
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--target", choices=["c1_ok", "rescue"], default="c1_ok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42, help="must match the router runs")
    ap.add_argument("--words", type=int, default=60, help="words kept from each passage")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--with_answer", action="store_true", help="reader also sees C1's answer (decides AFTER C1)")
    ap.add_argument("--no_class_weight", action="store_true", help="v1 behaviour: unweighted loss")
    ap.add_argument("--select", choices=["auc", "f1"], default="auc", help="how the best epoch is picked")
    ap.add_argument("--out", default="runs_suff")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from arroute.retrieval.index import load_corpus
    recs = load_outcomes(a.labels)
    ids, ptexts, _ = load_corpus(a.corpus)
    pt = dict(zip(ids, ptexts))
    def q_text(r):                       # with_answer: the reader verifies C1's answer against the passages
        ans = (r["results"]["C1"].get("answer") or "").strip()
        return f"{r['question']} [SEP] الجواب: {ans}" if a.with_answer else r["question"]
    texts = [reader_input(q_text(r), [pt.get(p, "") for p in r["results"]["C1"].get("passages") or []], a.words)
             for r in recs]
    y = targets(recs, a.target)
    tag = "tfidf" if a.backend == "tfidf" else a.model.split("/")[-1]
    name = (f"{tag}_{a.target}" + ("_ans" if a.with_answer else "")
            + ("_v2" if a.backend == "torch" and a.select == "auc" and not a.no_class_weight else "")
            + (f"_seed{a.seed}" if a.backend == "torch" else ""))
    out = Path(a.out) / name
    if (out / "oof.npz").exists() and not a.force:
        print(f"{name}: already done ({out}), skipping. Use --force to redo.")
        return
    print(f"[reader] {name} | {len(recs):,} questions | positive rate {100 * y.mean():.1f}% | {a.folds}-fold CV")
    cfg = {k: v for k, v in vars(a).items()}
    if a.backend == "torch":
        import torch
        import transformers
        from train_head import TorchBackend
        cfg.update(torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        cw = None if a.no_class_weight else [len(y) / (2 * max(1, (y == 0).sum())), len(y) / (2 * max(1, (y == 1).sum()))]
        cfg["class_weights"] = cw
        backend = TorchBackend(a.model, a.max_len, a.epochs, a.lr, a.batch, num_labels=2, class_weights=cw,
                               select=a.select)
    else:
        backend = TfidfBackend()
    t0 = time.time()
    res = run_cv(recs, texts, y, backend, a.folds, a.fold_seed, a.seed)
    cfg["seconds"] = round(time.time() - t0, 1)
    cfg["oof_auc"] = auc(res["p"], y)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "oof.npz", ids=np.array([r["id"] for r in recs]), fold=res["fold"],
                        logits=res["logits"], T=res["T"], p=res["p"])
    save_json(cfg, out / "config.json")
    save_json(res["info"], out / "folds.json")
    print(f"  out-of-fold AUC {cfg['oof_auc']:.3f} | saved {out} ({cfg['seconds']}s)")


if __name__ == "__main__":
    main()
