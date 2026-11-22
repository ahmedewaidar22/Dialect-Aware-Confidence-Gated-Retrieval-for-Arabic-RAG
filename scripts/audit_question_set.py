"""Quality gate for a confirmatory question set. Run it BEFORE any system sees the questions.

  python scripts/audit_question_set.py --questions data/confirm/confirm_test.jsonl
  python scripts/audit_question_set.py --questions data/confirm2/confirm_test.jsonl --strict   # exit code 1 if it fails

A question set fails the gate if, after removing flagged items, fewer than --min (default 500) clean questions remain,
or if more than --max_template_share (default 0.10) of the questions start with the same three words (a sign that the
questions were produced from a template rather than written). Flags per question:
  template        starts with a 3-word opening shared by more than --max_template_share of the set
  no_entity       the question has no content word besides question words (e.g. "اتولد  امتى؟")
  truncated       ends on a preposition, conjunction or open bracket before the question mark
  duplicate       the same question text appears more than once
  ref_long        reference answer longer than 12 words (not a short factual answer)
  ref_cut         reference ends in the middle of a phrase (open bracket or quote, trailing preposition)
  ref_is_opening  reference of 4+ words that is the opening of the gold passage (the title sentence, not an answer)
Writes <questions>.audit.csv with one row per question and its flags, so the set can be repaired by hand.
These rules are fixed here, before the next set exists; do not change them after seeing system results.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
QWORDS = {"إيه", "ايه", "مين", "امتى", "إمتى", "فين", "إزاي", "ازاي", "كام", "ليه", "اتولد", "اللي", "هو", "هي", "ما", "من",
          "متى", "أين", "كيف", "كم", "لماذا", "عدد", "صدر", "اتعرض", "صدر/اتعرض", "في", "عن", "المعلومات", "المذكورة"}
CUT_END = r"(في|عن|بن|أو|من|إلى|على|و|هو|هي|لعام|\(|«|\"|؛|:)\s*؟?\s*$"


def words(s):
    return re.findall(r"[^\s؟?،,.:؛!\"()«»{}\[\]/]+", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--min", type=int, default=500)
    ap.add_argument("--max_template_share", type=float, default=0.10)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    items = [json.loads(x) for x in Path(a.questions).read_text(encoding="utf-8").splitlines() if x.strip()]
    passages = {}
    cp = Path(a.corpus) / "passages.jsonl"
    need = {p for it in items for p in it.get("gold_pids") or []}
    if cp.exists() and need:
        with cp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p["id"] in need:
                        passages[p["id"]] = p["text"]
    q_of = lambda it: (it.get("q_egy") or "").strip()                       # noqa: E731
    openings = Counter(" ".join(words(q_of(it))[:3]) for it in items)
    dups = Counter(q_of(it) for it in items)
    n = len(items)
    rows = []
    for it in items:
        q = q_of(it)
        ref = str((it.get("refs") or [it.get("answer") or ""])[0] or "").strip()
        flags = []
        op = " ".join(words(q)[:3])
        if openings[op] > max(3, a.max_template_share * n):
            flags.append("template")
        if not [w for w in words(q) if w not in QWORDS]:
            flags.append("no_entity")
        if re.search(CUT_END, q.rstrip("؟? ")) or re.search(CUT_END, q):
            flags.append("truncated")
        if dups[q] > 1:
            flags.append("duplicate")
        if len(words(ref)) > 12:
            flags.append("ref_long")
        if re.search(r"(\(|«|\"|:|في|عن|بن|من)\s*$", ref) or ref.count("(") > ref.count(")") or ref.count('"') % 2:
            flags.append("ref_cut")
        gold = " ".join(passages.get(p, "") for p in it.get("gold_pids") or [])
        if gold and len(words(ref)) >= 4 and gold.strip().startswith(ref[:30]):
            flags.append("ref_is_opening")
        rows.append({"id": it.get("id"), "flags": " ".join(flags), "q_egy": q, "reference": ref})
    clean = [r for r in rows if not r["flags"]]
    top = openings.most_common(8)
    print(f"{n} questions | clean (no flag): {len(clean)}")
    print("flag counts:", dict(Counter(f for r in rows for f in r["flags"].split())))
    print("most common 3-word openings:")
    for op, k in top:
        print(f"  {100 * k / n:5.1f}%  {op}")
    share = top[0][1] / n if top else 0
    ok = len(clean) >= a.min and share <= a.max_template_share
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}  (clean >= {a.min}: {len(clean) >= a.min}; "
          f"top opening share {100 * share:.1f}% <= {100 * a.max_template_share:.0f}%: {share <= a.max_template_share})")
    out = Path(str(a.questions) + ".audit.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["id", "flags", "q_egy", "reference"])
        w.writeheader()
        w.writerows(rows)
    print(f"per-question flags written to {out}")
    if a.strict and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
