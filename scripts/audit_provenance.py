"""Split-integrity audit: does any evaluation question also appear (exactly or in meaning) in data used to build the system?

  python scripts/audit_provenance.py                 # GPU recommended for the semantic check (BGE-M3); no API calls
  python scripts/audit_provenance.py --no_semantic   # exact + near-exact checks only (fast, CPU)

BUILD sets (anything a model, threshold or choice was fitted on):
  dialect_train, dialect_dev, dialect_test_ragb   ArabicRAGB splits (dialect head training / its internal checks)
  dialect_dev_ardqa                               ArDQA dev (dialect head: epoch and temperature chosen here)
  complexity_train, complexity_dev                ArabicaQA / MAFQA complexity splits (if present)
  router_labels                                   the 1,955 outcome-labelled training questions (Router v1, Adaptive-RAG)
  calibration_667                                 ArDQA questions used for Router v2 and for choosing k
EVALUATION sets:
  test_ardqa, test_pilot, test_heldout            the 1,000-question end-to-end test, by part
  confirm_arabicaqa                               the confirmatory set (if present)
For every (evaluation, build) pair it reports, on both the Egyptian and the MSA text of each evaluation question:
  exact      identical after Arabic normalisation (hamza/alef/ya/ta-marbuta, diacritics, punctuation, spaces)
  near       character 3-gram Jaccard >= 0.85 (same question with small edits)
  semantic   BGE-M3 cosine >= 0.92 (likely the same question reworded) -> these need a human look
  passage    the evaluation question's gold passage is a gold passage of a build question (same source document)
Writes runs_audit/provenance_report.txt and runs_audit/provenance_flags.csv (every flagged pair, highest score
first, both texts side by side) so a person can confirm or dismiss each semantic flag.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

R_NOTE = []
FINAL_COL = "FINAL: rewrite + C1 on reranked top-k, 'I don't know' -> C2"


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def text_of(r):
    for k in ("question", "q", "judge_q", "q_msa", "q_egy"):
        if isinstance(r.get(k), str) and r[k].strip():
            return r[k]
    return ""


def gold_of(r):
    return list(r.get("gold_passage_ids") or r.get("gold_pids") or r.get("gold") or [])


def grams(s, n=3):
    s = f" {s} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def collect(a, normalize):
    """-> build: name -> list of (id, text, gold); evals: name -> list of (id, {'egy','msa'}, gold)."""
    build, evals = {}, {}
    for n in ("dialect_train", "dialect_dev", "dialect_test_ragb", "dialect_dev_ardqa", "complexity_train", "complexity_dev"):
        rows = load_jsonl(Path(a.splits) / f"{n}.jsonl")
        if rows:
            build[n] = [(r.get("id"), text_of(r), gold_of(r)) for r in rows]
    lab = load_jsonl(a.labels)
    held = set()
    hp = Path(a.heldout_ids)
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        held = set(h if isinstance(h, list) else h.get("ids", h.keys()))
    if lab:                                      # the 300 held-out questions were labelled but removed from training
        build["router_labels"] = [(r.get("id") or r.get("key"), text_of(r), gold_of(r)) for r in lab
                                  if (r.get("id") or r.get("key")) not in held]
        R_NOTE.append(f"router_labels: {len(lab)} labelled, {len(lab) - len(build['router_labels'])} held-out ids removed "
                      f"({hp} {'found' if hp.exists() else 'NOT FOUND: held-out questions will show as exact leaks'})")
    test = load_jsonl(a.test)
    test_ids = set()
    if Path(a.per_question).exists():
        with open(a.per_question, encoding="utf-8-sig", newline="") as f:
            test_ids = {r["id"] for r in csv.DictReader(f)}
    if not test_ids:
        print("WARNING: runs_final/per_question.csv not found: treating every egy_test question as a test question")
        test_ids = {r["id"] for r in test}
    calib = [r for r in test if r["set"] == "ardqa" and r["id"] not in test_ids]
    if calib:
        build["calibration_667"] = [(r["id"], r["q_msa"], gold_of(r)) for r in calib] + \
                                   [(r["id"] + "#egy", r["q_egy"], gold_of(r)) for r in calib]
    for part in ("ardqa", "pilot", "heldout"):
        rows = [r for r in test if r["set"] == part and r["id"] in test_ids]
        evals[f"test_{part}"] = [(r["id"], {"egy": r["q_egy"], "msa": r["q_msa"]}, gold_of(r)) for r in rows]
    conf = load_jsonl(a.confirm)
    if conf:
        evals["confirm_arabicaqa"] = [(r["id"], {"egy": r["q_egy"], "msa": r["q_msa"]}, gold_of(r)) for r in conf]
    return build, evals, len(calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--per_question", default="runs_final/per_question.csv")
    ap.add_argument("--heldout_ids", default="data/egy_test/heldout_ids.json")
    ap.add_argument("--confirm", default="data/confirm_arabicaqa/confirm_test.jsonl")
    ap.add_argument("--near", type=float, default=0.85)
    ap.add_argument("--semantic", type=float, default=0.92)
    ap.add_argument("--no_semantic", action="store_true")
    ap.add_argument("--out", default="runs_audit")
    a = ap.parse_args()
    from arroute.retrieval.text import normalize_ar
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalize_ar(s or ""))).strip()   # noqa: E731

    build, evals, n_cal = collect(a, normalize_ar)
    build = {k: v for k, v in build.items() if v}
    R = ["SPLIT-INTEGRITY AUDIT", "", "build sets: " + ", ".join(f"{k} ({len(v):,})" for k, v in build.items()),
         "evaluation sets: " + ", ".join(f"{k} ({len(v):,})" for k, v in evals.items())] + R_NOTE + [""]
    flags = []
    enc = None
    if not a.no_semantic:
        from arroute.retrieval.index import bge_m3_encoder
        enc = bge_m3_encoder()
    bemb = {}
    for bn, rows in build.items():
        texts = [norm(t) for _, t, _ in rows]
        exact = defaultdict(list)
        for i, t in enumerate(texts):
            exact[t].append(i)
        gidx = defaultdict(list)
        for i, (_, _, g) in enumerate(rows):
            for p in g:
                gidx[p].append(i)
        gram_index = defaultdict(set)
        bgrams = [grams(t) for t in texts]
        for i, g in enumerate(bgrams):
            for x in list(g)[:]:
                gram_index[x].add(i)
        if enc is not None:
            E = np.asarray(enc([t for _, t, _ in rows]), dtype=np.float32)
            bemb[bn] = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        for en, erows in evals.items():
            c = {"exact": 0, "near": 0, "semantic": 0, "passage": 0}
            hit_q = set()
            for eid, txt, gold in erows:
                for var, raw in txt.items():
                    t = norm(raw)
                    if t in exact:
                        c["exact"] += 1
                        hit_q.add(eid)
                        j = exact[t][0]
                        flags.append([en, bn, "exact", var, 1.0, eid, raw, rows[j][0], rows[j][1]])
                        continue
                    g = grams(t)
                    cand = defaultdict(int)
                    for x in g:
                        for j in gram_index.get(x, ()):
                            cand[j] += 1
                    best, bj = 0.0, None
                    for j, inter in cand.items():
                        jac = inter / (len(g) + len(bgrams[j]) - inter)
                        if jac > best:
                            best, bj = jac, j
                    if best >= a.near:
                        c["near"] += 1
                        hit_q.add(eid)
                        flags.append([en, bn, "near", var, round(best, 3), eid, raw, rows[bj][0], rows[bj][1]])
                shared = [p for p in gold if p in gidx]
                if shared:
                    c["passage"] += 1
                    flags.append([en, bn, "passage", "-", 1.0, eid, txt["msa"], rows[gidx[shared[0]][0]][0],
                                  rows[gidx[shared[0]][0]][1]])
            if enc is not None and erows:
                for var in ("egy", "msa"):
                    Q = np.asarray(enc([t[var] for _, t, _ in erows]), dtype=np.float32)
                    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
                    for s in range(0, len(Q), 256):
                        S = Q[s:s + 256] @ bemb[bn].T
                        j = S.argmax(1)
                        for i, (jj, sc) in enumerate(zip(j, S[np.arange(len(j)), j])):
                            if sc >= a.semantic:
                                eid, txt, _ = erows[s + i]
                                c["semantic"] += 1
                                hit_q.add(eid)
                                flags.append([en, bn, "semantic", var, round(float(sc), 3), eid, txt[var], rows[jj][0], rows[jj][1]])
            R.append(f"{en:18s} vs {bn:18s} exact {c['exact']:4d}  near {c['near']:4d}  semantic {c['semantic']:4d}"
                     f"  shared gold passage {c['passage']:4d}  questions flagged {len(hit_q)}")
        R.append("")
    R.append("Exact/near hits are leaks unless the build set is not used by the final system (see paper Table of decisions).")
    R.append("Semantic hits are candidates: read provenance_flags.csv and mark each as same question / different question.")
    R.append("Shared gold passages are expected where the same corpus document is asked about twice; they are not a")
    R.append("question leak, but report how many test questions share a source document with training questions.")
    text = "\n".join(R)
    print(text)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "provenance_report.txt").write_text(text + "\n", encoding="utf-8")
    flags.sort(key=lambda r: (r[2] != "exact", r[2] != "near", -r[4]))
    with (o / "provenance_flags.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_set", "build_set", "kind", "variant", "score", "eval_id", "eval_text", "build_id", "build_text",
                    "same_question (y/n)"])
        w.writerows(r + [""] for r in flags)
    print(f"\nwritten {o / 'provenance_report.txt'} and {o / 'provenance_flags.csv'} ({len(flags)} flags)")


if __name__ == "__main__":
    main()
