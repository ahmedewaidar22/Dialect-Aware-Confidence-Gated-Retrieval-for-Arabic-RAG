"""Phase 7: train and SAVE the dialect head (Egyptian vs MSA), then check it on the Phase 5 Egyptian test set.

  python scripts/train_dialect_final.py          # ~3 min on an RTX 3070

Same recipe as Phase 2a (AraBERT, trained on the ArabicRAGB dialect split, epoch + temperature chosen on
ArDQA dev). Saves models/dialect_head/ (+ meta.json with the temperature and the confidence band).
The confidence band decides what the router does:
  P(Egyptian) >= hi  -> rewrite to MSA          P(Egyptian) <= lo  -> search as typed
  in between         -> search BOTH (original + rewrite) and merge: safe either way (Phase 6b)
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

from arroute.router.metrics import binary_metrics, ece, fit_temperature, softmax, wilson   # noqa: E402
from arroute.router.tasks import load_task                                               # noqa: E402


def band_report(p_egy_on_egy, p_egy_on_msa, lo, hi):
    """What the router would do with each input, given the band."""
    out = {}
    for name, p in (("Egyptian input", p_egy_on_egy), ("MSA input", p_egy_on_msa)):
        p = np.asarray(p)
        out[name] = {"rewrite": float((p >= hi).mean()), "both (unsure)": float(((p > lo) & (p < hi)).mean()),
                     "as typed": float((p <= lo).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lo", type=float, default=0.2, help="P(Egyptian) at or below: treat as MSA")
    ap.add_argument("--hi", type=float, default=0.8, help="P(Egyptian) at or above: treat as Egyptian")
    ap.add_argument("--out", default="models/dialect_head")
    a = ap.parse_args()

    import torch
    import transformers
    from train_head import TorchBackend
    t0 = time.time()
    task = load_task("dialect", a.splits)
    s_texts, s_y, _ = task.evals[task.select]
    print(f"[dialect] train {len(task.train_texts):,} | select on {task.select} ({len(s_texts)})")
    be = TorchBackend(a.model, task.max_len, a.epochs, a.lr, a.batch, num_labels=2)
    info = be.fit(task.train_texts, task.train_y, s_texts, s_y, a.seed)
    T = fit_temperature(be.logits(s_texts), np.asarray(s_y))
    be.save(a.out)

    L = [f"DIALECT HEAD (saved to {a.out})  best epoch {info['best_epoch']}, temperature {T:.2f}", ""]
    js = {"temperature": T, "lo": a.lo, "hi": a.hi, "positive_class": "egy (label 1)", "model": a.model,
          "best_epoch": info["best_epoch"], "torch": torch.__version__, "transformers": transformers.__version__,
          "max_len": task.max_len, "eval": {}}
    for name, (texts, y, _) in task.evals.items():
        p = softmax(be.logits(texts), T)
        m = binary_metrics(y, p.argmax(1))
        js["eval"][name] = {**m, "ece": ece(p, np.asarray(y))}
        L.append(f"  {name:<22} n={m['n']:5d}  accuracy {100 * m['accuracy']:5.1f}%  macro-F1 {100 * m['macro_f1']:5.1f}%  "
                 f"ECE {100 * js['eval'][name]['ece']:.1f}%")

    if Path(a.test).exists():
        items = [json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()]
        items = [it for it in items if "rejected" not in (it.get("review") or {}).values()]
        pe = softmax(be.logits([it["q_egy"] for it in items]), T)[:, 1]
        pm = softmax(be.logits([it["q_msa"] for it in items]), T)[:, 1]
        sets = np.array([it["set"] for it in items])
        L += ["", "ON THE PHASE 5 EGYPTIAN TEST SET (never used for training or selection)"]
        L.append(f"  {'set':<9} {'Egyptian found':>16} {'95% CI':>13} {'MSA kept as MSA':>17} {'95% CI':>13}")
        for s in ("ardqa", "pilot", "heldout", "ALL"):
            m_ = (sets == s) if s != "ALL" else np.ones(len(items), bool)
            ke, km = int((pe[m_] >= 0.5).sum()), int((pm[m_] < 0.5).sum())
            n = int(m_.sum())
            (le, he), (lm, hm) = wilson(ke, n), wilson(km, n)
            js["eval"][f"egy_test_{s}"] = {"egy_recall": ke / n, "msa_specificity": km / n, "n": n}
            L.append(f"  {s:<9} {100 * ke / n:15.1f}% [{100 * le:4.1f}-{100 * he:4.1f}] {100 * km / n:16.1f}% "
                     f"[{100 * lm:4.1f}-{100 * hm:4.1f}]")
        br = band_report(pe, pm, a.lo, a.hi)
        js["band"] = br
        L += ["", f"CONFIDENCE BAND lo={a.lo} hi={a.hi}: what the router will do"]
        for name, d in br.items():
            L.append(f"  {name:<15} rewrite {100 * d['rewrite']:5.1f}% | both (unsure) {100 * d['both (unsure)']:5.1f}% "
                     f"| as typed {100 * d['as typed']:5.1f}%")
        with (Path(a.out) / "egy_test_scores.jsonl").open("w", encoding="utf-8") as f:
            for it, x, y in zip(items, pe, pm):
                f.write(json.dumps({"id": it["id"], "p_egy_on_egy": float(x), "p_egy_on_msa": float(y)}) + "\n")
    js["seconds"] = round(time.time() - t0, 1)
    (Path(a.out) / "meta.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    text = "\n".join(L)
    print("\n" + text)
    (Path(a.out) / "dialect_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
