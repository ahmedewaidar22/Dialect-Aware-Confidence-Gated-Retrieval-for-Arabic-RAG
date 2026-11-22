"""Phase 4c step 1: answer-check features for every labelled question (local, free, seconds).

  python scripts/escalation_features.py
Writes data/router_feats/escalation_feats.jsonl  ({"id", "feats": {...}}) + a small summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.retrieval.index import load_corpus                                     # noqa: E402
from arroute.router.escalate import ANSWER_FEATURES, C0_FEATURES, answer_features   # noqa: E402
from arroute.router.outcome import load_outcomes                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out", default="data/router_feats/escalation_feats.jsonl")
    a = ap.parse_args()
    recs = load_outcomes(a.labels)
    ids, texts, _ = load_corpus(a.corpus)
    ptext = dict(zip(ids, texts))
    missing = sum(p not in ptext for r in recs for p in r["results"]["C1"].get("passages") or [])
    feats = [answer_features(r, ptext) for r in recs]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, ft in zip(recs, feats):
            f.write(json.dumps({"id": r["id"], "feats": ft}, ensure_ascii=False) + "\n")
    c1 = np.array([r["results"]["C1"]["correct"] for r in recs], bool)
    lines = [f"answer-check features for {len(recs):,} questions -> {out}",
             f"C1 passages not found in the corpus: {missing} (should be 0)", "",
             f"{'feature':<16} {'mean':>6}   C1 correct when feature=1 / =0 (binary features)"]
    for k in ANSWER_FEATURES + C0_FEATURES:
        v = np.array([ft[k] for ft in feats])
        extra = ""
        if set(np.unique(v)) <= {0.0, 1.0} and 0 < v.mean() < 1:
            extra = f"   {100 * c1[v == 1].mean():5.1f}% / {100 * c1[v == 0].mean():5.1f}%"
        lines.append(f"{k:<16} {v.mean():6.3f}{extra}")
    text = "\n".join(lines)
    print(text)
    out.with_name("escalation_feats_summary.txt").write_text(text + "\n", encoding="utf-8")
    if missing:
        print("\nWARNING: some passages are missing - was the corpus rebuilt after labelling? Tell Claude.")


if __name__ == "__main__":
    main()
