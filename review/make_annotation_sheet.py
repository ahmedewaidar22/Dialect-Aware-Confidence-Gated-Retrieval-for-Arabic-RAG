"""Build blind annotation sheets for validating the LLM judge.

Input: one row per graded answer (JSONL or CSV) with the question, the reference answer, the system's answer, the
judge's verdict and, optionally, a set name. The sample is stratified by (set, judge verdict) so that both correct
and incorrect verdicts are well represented (kappa cannot be estimated if one class is almost empty).

  python scripts/make_annotation_sheet.py --input runs_phase8/graded.jsonl --id-field key --question-field judge_q \
      --reference-field refs --answer-field answer --judge-field correct --set-field set --n 300 --out annotation

Writes  annotation_A.csv, annotation_B.csv   (give one to each annotator; same rows, DIFFERENT random order,
                                              judge verdict and system name hidden; they fill the `label` column 1/0)
        annotation_key.csv                    (id, set, judge verdict: keep private until both have finished)
Annotators grade: is the answer correct given the reference (1) or not (0)? Paraphrases and equivalent forms count as
correct; "I don't know" counts as incorrect. Write the rule in the paper's appendix.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def as_text(v):
    if isinstance(v, (list, tuple)):
        return " | ".join(map(str, v))
    return "" if v is None else str(v)


def as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "correct"}
    return bool(v)


def stratified(rows, n, seed, key):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    quota = max(1, n // max(1, len(groups)))
    picked = []
    for g in groups.values():
        picked += g[:quota]
    rest = [r for g in groups.values() for r in g[quota:]]
    rng.shuffle(rest)
    picked += rest[:max(0, n - len(picked))]
    return picked[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--reference-field", default="reference")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--judge-field", default="correct")
    ap.add_argument("--set-field", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="annotation")
    a = ap.parse_args(argv)
    rows = read_rows(a.input)
    sf = (lambda r: r.get(a.set_field, "all")) if a.set_field else (lambda r: "all")   # noqa: E731
    sample = stratified(rows, a.n, a.seed, lambda r: (sf(r), as_bool(r[a.judge_field])))
    fields = ["id", "question", "reference", "answer", "label", "note"]
    for tag, seed in (("A", a.seed + 1), ("B", a.seed + 2)):
        order = sample[:]
        random.Random(seed).shuffle(order)
        with open(f"{a.out}_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in order:
                w.writerow({"id": r[a.id_field], "question": as_text(r[a.question_field]),
                            "reference": as_text(r[a.reference_field]), "answer": as_text(r[a.answer_field]),
                            "label": "", "note": ""})
    with open(f"{a.out}_key.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "set", "judge"])
        for r in sample:
            w.writerow([r[a.id_field], sf(r), int(as_bool(r[a.judge_field]))])
    print(f"{len(sample)} answers sampled -> {a.out}_A.csv, {a.out}_B.csv, {a.out}_key.csv")


if __name__ == "__main__":
    main()
