"""Phase 4a: train the complexity head on OUTCOME labels with 5-fold cross-validation.

Every question gets a prediction from a model that never saw it (out-of-fold). Inside each fold, a
small inner-dev split (15% of the training folds) picks the epoch, the temperature and the caution
knob λ - so nothing about the test fold leaks into any choice.

  python scripts/train_router.py --backend ngram                                   # baseline, CPU, seconds
  python scripts/train_router.py --backend length                                  # 'is it just length?' baseline
  python scripts/train_router.py --backend torch --model aubmindlab/bert-base-arabertv02 --seed 0
  python scripts/train_router.py --backend torch ... --none_as C0                  # variant: 'none' trained as C0

Writes runs_router/<name>/{oof.npz, config.json, folds.json}. Then: python scripts/eval_router.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.metrics import fit_temperature, softmax           # noqa: E402
from arroute.router.outcome import (STRATS, choose_cautious, evaluate_routing, fixed_choice,  # noqa: E402
                                    idk_matrix, inner_dev_split, label_id, label_metrics, load_outcomes,
                                    make_folds, outcome_matrix, save_json, trainable)
from arroute.router.route_report import LAM_GRID                       # noqa: E402

DEFAULT_LABELS = "data/labels/main_full/labels.jsonl"


# ------------------------------------------------------------------ simple backends (no torch)
class NgramBackend:
    """Character n-gram TF-IDF + logistic regression: the strong 'surface' baseline from Phase 2."""

    def __init__(self, balanced=False):
        self.balanced = balanced

    def fit(self, texts, y, dev_texts, dev_y, seed):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.clf = LogisticRegression(C=2.0, max_iter=3000, random_state=seed,
                                      class_weight="balanced" if self.balanced else None).fit(X, y)
        return {"best_epoch": 0, "history": []}

    def logits(self, texts):
        lp = self.clf.predict_log_proba(self.vec.transform(texts))
        out = np.full((len(texts), 3), -30.0)
        out[:, self.clf.classes_] = lp
        return out


class LengthBackend(NgramBackend):
    """Only question length (chars, words). If this were as good as the real head, the head learned length."""

    def fit(self, texts, y, dev_texts, dev_y, seed):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced" if self.balanced else None
                                      ).fit(self._f(texts), y)
        return {"best_epoch": 0, "history": []}

    @staticmethod
    def _f(texts):
        return np.array([[np.log1p(len(t)), np.log1p(len(t.split()))] for t in texts])

    def logits(self, texts):
        lp = self.clf.predict_log_proba(self._f(texts))
        out = np.full((len(texts), 3), -30.0)
        out[:, self.clf.classes_] = lp
        return out


class FeatureBackend:
    """Phase 4b: 3-class classifier on retrieval-confidence features only (no question words)."""

    def __init__(self, feat_by_text, balanced=False, clf="lr"):
        self.f, self.balanced, self.kind = feat_by_text, balanced, clf

    def _X(self, texts):
        return np.array([self.f[t] for t in texts])

    def _model(self, seed):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        cw = "balanced" if self.balanced else None
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                  class_weight=cw, random_state=seed)
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight=cw, random_state=seed))

    def fit(self, texts, y, dev_texts, dev_y, seed):
        self.clf = self._model(seed).fit(self._X(texts), y)
        return {"best_epoch": 0, "history": []}

    def logits(self, texts):
        p = self.clf.predict_proba(self._X(texts))
        out = np.full((len(texts), 3), -30.0)
        out[:, self.clf.classes_] = np.log(np.clip(p, 1e-12, 1))
        return out


class TwoStageBackend(FeatureBackend):
    """Phase 4b two-stage router.
    Stage A (skip retrieval?): P(C0) from an already-trained text head (its out-of-fold predictions).
    Stage B (one search enough?): binary model on retrieval features, trained only on questions labelled C1/C2.
    P = [pC0, (1 - pC0) * (1 - pNeedC2), (1 - pC0) * pNeedC2]."""

    def __init__(self, feat_by_text, p0_by_text, balanced=False, clf="lr"):
        super().__init__(feat_by_text, balanced, clf)
        self.p0 = p0_by_text

    def fit(self, texts, y, dev_texts, dev_y, seed):
        y = np.asarray(y)
        keep = y >= 1
        X = self._X([t for t, k in zip(texts, keep) if k])
        self.clf = self._model(seed).fit(X, (y[keep] == 2).astype(int))
        return {"best_epoch": 0, "history": []}

    def logits(self, texts):
        p2 = self.clf.predict_proba(self._X(texts))[:, list(self.clf.classes_).index(1)]
        p0 = np.array([self.p0[t] for t in texts])
        P = np.stack([p0, (1 - p0) * (1 - p2), (1 - p0) * p2], axis=1)
        return np.log(np.clip(P, 1e-12, 1))


def load_feats(path, recs):
    """question text -> feature vector (identical texts get identical retrieval, so text is a safe key)."""
    from arroute.router.retfeats import FEATURES
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = [r["feats"][k] for k in FEATURES]
    missing = [r["id"] for r in recs if r["id"] not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no retrieval features - re-run scripts/retrieval_features.py")
    return {r["question"]: by_id[r["id"]] for r in recs}


def load_p0(text_run, recs, folds, fold_seed):
    """Calibrated out-of-fold P(C0) from a finished text-head run that used the SAME folds."""
    cfg = json.loads((Path(text_run) / "config.json").read_text(encoding="utf-8"))
    if (cfg.get("folds", 5), cfg.get("fold_seed", 42)) != (folds, fold_seed):
        raise SystemExit(f"{text_run} used different folds - stage A and stage B must share folds")
    d = np.load(Path(text_run) / "oof.npz", allow_pickle=False)
    by_id = {str(i): softmax(lg[None, :] / t, 1.0)[0, 0] for i, lg, t in zip(d["ids"], d["logits"], d["T"])}
    if any(r["id"] not in by_id for r in recs):
        raise SystemExit(f"{text_run} does not cover all questions in the labels file")
    return {r["question"]: float(by_id[r["id"]]) for r in recs}


def make_backend(a):
    if a.backend in ("retfeat", "twostage"):
        feats = load_feats(a.feats, a._recs)
        if a.backend == "retfeat":
            return FeatureBackend(feats, a.balanced, a.clf)
        return TwoStageBackend(feats, load_p0(a.text_run, a._recs, a.folds, a.fold_seed), a.balanced, a.clf)
    if a.backend == "ngram":
        return NgramBackend(a.balanced)
    if a.backend == "length":
        return LengthBackend(a.balanced)
    from train_head import TorchBackend               # scripts/ is on sys.path when run as a script
    return TorchBackend(a.model, a.max_len, a.epochs, a.lr, a.batch, num_labels=3,
                        class_weights=a._class_weights)


# ------------------------------------------------------------------ λ choice on inner dev
def choose_lambda(probs, correct, cost, calls, tol, idk=None):
    """Cheapest λ whose dev accuracy is within `tol` of always-C2 on the same dev questions.
    With `idk`, the policy escalates on 'I don't know' and λ is chosen for that policy."""
    target = evaluate_routing(correct, cost, calls, fixed_choice(len(probs), "C2"))["accuracy"] - tol
    best = None
    for lam in [0.0] + LAM_GRID:
        ev = evaluate_routing(correct, cost, calls, choose_cautious(probs, lam), idk)
        if ev["accuracy"] >= target - 1e-12 and (best is None or ev["cost_per_1k"] < best[1] - 1e-12):
            best = (lam, ev["cost_per_1k"])
    return best[0] if best else 1.0


def run_cv(recs, backend, k, fold_seed, seed, none_as, tol, log=print):
    n = len(recs)
    fold = make_folds(recs, k=k, seed=fold_seed)
    correct, cost, calls = outcome_matrix(recs)
    idk = idk_matrix(recs)
    texts = [r["question"] for r in recs]
    tr = np.array([trainable(r, none_as) for r in recs])
    y = np.array([label_id(r, none_as) if t else -1 for r, t in zip(recs, tr)])
    logits, T, lam, lam_esc = np.zeros((n, 3), np.float32), np.zeros(n), np.zeros(n), np.zeros(n)
    info = []
    for f in range(k):
        test = np.where(fold == f)[0]
        fit_i, dev_i = inner_dev_split(np.where(fold != f)[0], recs, seed=seed + 100 * f)
        fit_t, dev_t = fit_i[tr[fit_i]], dev_i[tr[dev_i]]
        t0 = time.time()
        fi = backend.fit([texts[i] for i in fit_t], y[fit_t], [texts[i] for i in dev_t], y[dev_t], seed + f)
        dev_lg = backend.logits([texts[i] for i in dev_i])
        Tf = fit_temperature(dev_lg[tr[dev_i]], y[dev_t])
        dev_p = softmax(dev_lg, Tf)
        lf = choose_lambda(dev_p, correct[dev_i], cost[dev_i], calls[dev_i], tol)
        le = choose_lambda(dev_p, correct[dev_i], cost[dev_i], calls[dev_i], tol, idk[dev_i])
        logits[test] = backend.logits([texts[i] for i in test])
        T[test], lam[test], lam_esc[test] = Tf, lf, le
        tm = label_metrics(y[test[tr[test]]], logits[test[tr[test]]].argmax(1))
        info.append({"fold": f, "n_test": len(test), "n_fit": len(fit_t), "n_dev": len(dev_i), "T": Tf, "lam": lf, "lam_esc": le,
                     "best_epoch": fi["best_epoch"], "history": fi["history"], "test_label_macro_f1": tm["macro_f1"],
                     "seconds": round(time.time() - t0, 1)})
        log(f"  fold {f}: fit {len(fit_t)} | dev {len(dev_i)} | test {len(test)} | T={Tf:.2f} λ={lf:.2f} λ_esc={le:.2f} | "
            f"test label acc {100 * tm['accuracy']:.1f}% macro-F1 {100 * tm['macro_f1']:.1f}% | {info[-1]['seconds']}s")
    return {"fold": fold, "logits": logits, "T": T, "lam": lam, "lam_esc": lam_esc, "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--backend", choices=["ngram", "length", "torch", "retfeat", "twostage"], required=True)
    ap.add_argument("--feats", default="data/router_feats/retrieval_feats.jsonl", help="retfeat/twostage")
    ap.add_argument("--text_run", default=None, help="twostage: finished text-head run, e.g. runs_router/bert-base-arabertv02_seed0")
    ap.add_argument("--clf", choices=["lr", "hgb"], default="lr", help="retfeat/twostage: logistic regression or boosted trees")
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42, help="keep fixed so all runs share the same folds")
    ap.add_argument("--none_as", choices=["C0", "C1", "C2"], default=None)
    ap.add_argument("--balanced", action="store_true", help="class-weighted loss (C2 is rare)")
    ap.add_argument("--tol", type=float, default=0.01, help="λ target: within this of always-C2 dev accuracy")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out", default="runs_router")
    ap.add_argument("--name", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    recs = load_outcomes(a.labels)
    a._recs = recs
    if a.backend == "twostage" and not a.text_run:
        raise SystemExit("--backend twostage needs --text_run runs_router/<text head run>")
    tr = [trainable(r, a.none_as) for r in recs]
    ytr = [label_id(r, a.none_as) for r, t in zip(recs, tr) if t]
    counts = np.bincount(ytr, minlength=3)
    a._class_weights = (len(ytr) / (3 * np.maximum(counts, 1))).tolist() if a.balanced else None

    tag = {"torch": a.model.split("/")[-1], "twostage": f"twostage_{Path(a.text_run or '').name}"}.get(a.backend, a.backend)
    tag += "_hgb" if a.backend in ("retfeat", "twostage") and a.clf == "hgb" else ""
    name = a.name or (f"{tag}" + ("_bal" if a.balanced else "") + (f"_none{a.none_as}" if a.none_as else "")
                      + (f"_seed{a.seed}" if a.backend == "torch" else ""))
    if a.backend == "twostage" and not a.name:          # keep the text run's '_seedK' at the end (seed grouping)
        name = ("twostage" + ("_hgb" if a.clf == "hgb" else "") + ("_bal" if a.balanced else "")
                + (f"_none{a.none_as}" if a.none_as else "") + f"__{Path(a.text_run).name}")
    out = Path(a.out) / name
    if (out / "oof.npz").exists() and not a.force:
        print(f"{name}: already done ({out}), skipping. Use --force to redo.")
        return
    print(f"[router] {name} | {len(recs):,} questions, {sum(tr):,} trainable | labels "
          + " ".join(f"{s} {c}" for s, c in zip(STRATS, counts)) + f" | {a.folds}-fold CV")

    cfg = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    cfg.update(backend=a.backend, model=a.model if a.backend == "torch" else None, class_weights=a._class_weights)
    if a.backend == "torch":
        import torch
        import transformers
        cfg.update(torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    res = run_cv(recs, make_backend(a), a.folds, a.fold_seed, a.seed, a.none_as, a.tol)
    cfg["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "oof.npz", ids=np.array([r["id"] for r in recs]), fold=res["fold"],
                        logits=res["logits"], T=res["T"], lam=res["lam"], lam_esc=res["lam_esc"])
    save_json(cfg, out / "config.json")
    save_json(res["info"], out / "folds.json")
    print(f"  saved {out}  ({cfg['seconds']}s)")


if __name__ == "__main__":
    main()
