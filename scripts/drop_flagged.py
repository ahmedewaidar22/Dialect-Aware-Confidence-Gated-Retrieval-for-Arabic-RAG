"""Turn the provenance audit's flags into id lists for a worst-case check (drop every flagged question).

  python scripts/drop_flagged.py                       # flags against the sets the compared systems were built on
  python scripts/drop_flagged.py --confirmed_only      # only rows you marked 'y' in same_question (y/n)

Relevant build sets (default): dialect_train, dialect_dev_ardqa (dialect head), router_labels (Router v1, Adaptive-RAG,
k), calibration_667 (Router v2, k). complexity_* splits are not used by any compared system and are ignored.
Kinds: exact, near, semantic (shared gold passages are not dropped; they are reported separately).
Writes runs_audit/leaked_ids_test.txt (for subset_results.py --drop) and, for the confirmatory set,
runs_confirm2/results_no_flagged.jsonl (run analyze_confirmatory.py on it unchanged).
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default="runs_audit/provenance_flags.csv")
    ap.add_argument("--build_sets", default="dialect_train,dialect_dev_ardqa,router_labels,calibration_667")
    ap.add_argument("--kinds", default="exact,near,semantic")
    ap.add_argument("--confirmed_only", action="store_true")
    ap.add_argument("--confirm_results", default="runs_confirm2/results.jsonl")
    a = ap.parse_args()
    bs, kinds = set(a.build_sets.split(",")), set(a.kinds.split(","))
    test, conf = set(), set()
    with open(a.flags, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["build_set"] not in bs or r["kind"] not in kinds:
                continue
            if a.confirmed_only and r.get("same_question (y/n)", "").strip().lower() not in ("y", "yes"):
                continue
            (conf if r["eval_set"] == "confirm_arabicaqa" else test).add(r["eval_id"])
    Path("runs_audit").mkdir(exist_ok=True)
    Path("runs_audit/leaked_ids_test.txt").write_text("\n".join(sorted(test)) + "\n", encoding="utf-8")
    print(f"test questions to drop: {len(test)} -> runs_audit/leaked_ids_test.txt")
    cr = Path(a.confirm_results)
    if cr.exists():
        rows = [x for x in cr.read_text(encoding="utf-8").splitlines() if x.strip()]
        keep = [x for x in rows if json.loads(x)["id"] not in conf]
        outp = cr.with_name("results_no_flagged.jsonl")
        outp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        print(f"confirmatory questions dropped: {len(rows) - len(keep)} of {len(rows)} -> {outp}")


if __name__ == "__main__":
    main()
