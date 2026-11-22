"""Print the questions a finished run got wrong (or right), joined from the saved logits and the splits.

  python scripts/inspect_errors.py --run runs/complexity__aubmindlab_bert-base-arabertv02__seed0 \
      --set complexity_test --source arabicaqa --kind fp --n 25
kind: fp = true negative class predicted positive (e.g. C1 flagged C2), fn = positive missed, tp, tn, all
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.metrics import softmax  # noqa: E402
from arroute.router.tasks import load_task   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a runs/<task>__<model>__seed<k> folder")
    ap.add_argument("--set", required=True, help="eval set name, e.g. complexity_test")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--source", default=None)
    ap.add_argument("--kind", choices=["fp", "fn", "tp", "tn", "all"], default="fp")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args(argv)

    run = Path(a.run)
    meta = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    task = load_task(meta["task"], a.splits, meta["config"].get("add_ardqa_dev", False))
    if a.set not in task.evals:
        raise SystemExit(f"unknown set '{a.set}'. Available: {list(task.evals)}")
    texts, y, sources = task.evals[a.set]
    logits = np.load(run / "logits.npz")[a.set.replace(" ", "_")]
    if len(logits) != len(texts):
        raise SystemExit(f"logits ({len(logits)}) and split ({len(texts)}) differ: the splits changed since this run")
    p_pos = softmax(logits, 1.0)[:, 1]
    pred = (p_pos >= 0.5).astype(int)
    y = np.asarray(y)
    sel = np.ones(len(y), dtype=bool) if a.source is None else np.array([s == a.source for s in sources])
    kinds = {"fp": (y == 0) & (pred == 1), "fn": (y == 1) & (pred == 0),
             "tp": (y == 1) & (pred == 1), "tn": (y == 0) & (pred == 0), "all": np.ones(len(y), dtype=bool)}
    idx = np.where(sel & kinds[a.kind])[0]
    wl = lambda ix: statistics.median([len(texts[i].split()) for i in ix]) if len(ix) else float("nan")
    pos_name = meta.get("positive_class", "positive")
    print(f"{run.name} | set {a.set} | source {a.source or 'all'} | selected {len(idx):,} of {int(sel.sum()):,} "
          f"({a.kind}); positive class = {pos_name}")
    flagged, unflagged = np.where(sel & (pred == 1))[0], np.where(sel & (pred == 0))[0]
    print(f"median words: predicted-positive {wl(flagged)} | predicted-negative {wl(unflagged)}\n")
    order = idx[np.argsort(-p_pos[idx])] if a.kind in ("fp", "tp") else idx[np.argsort(p_pos[idx])]
    for i in order[:a.n]:
        print(f"  P({pos_name})={p_pos[i]:.2f} [{sources[i]}] {texts[i]}")
    return len(idx)


if __name__ == "__main__":
    main()
