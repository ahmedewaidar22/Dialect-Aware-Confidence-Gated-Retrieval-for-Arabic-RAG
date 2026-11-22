"""Step 2: turn the filled writer sheet into the confirmatory question file.

  python scripts/build_confirm_set.py --sheet data/confirm/passages_to_write.csv --out data/confirm/confirm_test.jsonl

Keeps rows where q_egy, q_msa and answer are all filled. Writes one JSON line per question:
  id, set="confirm", q_egy, q_msa, answer, refs, gold_pids.
Refuses to continue if fewer than --min rows are usable (default 500): collect more before running anything.
"""
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", default="data/confirm/confirm_test.jsonl")
    ap.add_argument("--min", type=int, default=500)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.sheet, encoding="utf-8-sig", newline="")))
    keep = [r for r in rows if all((r.get(k) or "").strip() for k in ("q_egy", "q_msa", "answer"))]
    print(f"{len(keep)} usable of {len(rows)} rows")
    if len(keep) < a.min:
        raise SystemExit(f"need at least {a.min}; collect more questions first")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps({"id": f"confirm:{r['id']}", "set": "confirm", "q_egy": r["q_egy"].strip(),
                                "q_msa": r["q_msa"].strip(), "answer": r["answer"].strip(),
                                "refs": [r["answer"].strip()], "gold_pids": [r["id"]]}, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
