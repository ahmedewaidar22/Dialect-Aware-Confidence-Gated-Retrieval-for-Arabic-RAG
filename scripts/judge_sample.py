"""Validate the LLM judge against YOUR grades.

  python scripts/judge_sample.py export --run main_mini --n 100
      -> data/labels/<run>/judge_sample.csv  (fill the column 'your_verdict' with y or n; the judge's verdict
         is hidden in judge_sample_key.json so it cannot influence you)
  python scripts/judge_sample.py agree --run main_mini
      -> agreement, Cohen's kappa, and every disagreement listed
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path
from random import Random

sys.stdout.reconfigure(encoding="utf-8")


def export(run_dir, n, seed):
    recs = [json.loads(x) for x in (run_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [(r, s, res) for r in recs for s, res in r["results"].items() if res["judge"] != "rule"]
    rule = [(r, s, res) for r in recs for s, res in r["results"].items() if res["judge"] == "rule"]
    rng = Random(seed)
    rng.shuffle(items)
    rng.shuffle(rule)
    k_rule = min(len(rule), n // 5)                              # 20% rule-based, to check the rules too
    pick = items[:n - k_rule] + rule[:k_rule]
    rng.shuffle(pick)
    key = {}
    with (run_dir / "judge_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "question", "reference_answer", "model_answer", "your_verdict (y/n)"])
        for j, (r, s, res) in enumerate(pick):
            item = f"{j:03d}"
            key[item] = {"id": r["id"], "strategy": s, "judge": res["correct"], "method": res["judge"]}
            w.writerow([item, r["question"], " / ".join(r["refs"]), res["answer"], ""])
    (run_dir / "judge_sample_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(pick)} items to {run_dir / 'judge_sample.csv'}  ({k_rule} rule-based, {len(pick) - k_rule} LLM-judged)")


YES = {"y", "yes", "1", "true", "نعم", "ص"}
NO = {"n", "no", "0", "false", "لا", "خ"}


def read_graded_csv(path):
    """Excel on Windows may save with ';' or tabs, in a legacy encoding, and turn item '007' into 7."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1256", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    try:
        dialect = csv.Sniffer().sniff(first, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    return [{(k or "").strip().lstrip("\ufeff"): (v or "") for k, v in r.items()} for r in rows]


def agree(run_dir):
    key = json.loads((run_dir / "judge_sample_key.json").read_text(encoding="utf-8"))
    path = run_dir / "judge_sample.csv"
    rows = read_graded_csv(path)
    if not rows:
        raise SystemExit(f"{path} is empty")
    col = next((c for c in rows[0] if "verdict" in c.lower()), None)
    item_col = next((c for c in rows[0] if c.lower() == "item"), None)
    if not col or not item_col:
        raise SystemExit(f"could not find the 'item' and 'your_verdict' columns in {path}; columns seen: {list(rows[0])}")
    pairs, unknown = [], []
    for row in rows:
        v = row[col].strip().lower()
        try:
            item = f"{int(float(row[item_col])):03d}"            # '7' or '7.0' -> '007'
        except ValueError:
            continue
        if item in key and v in YES | NO:
            pairs.append((v in YES, key[item]["judge"], row))
        elif v:
            unknown.append(v)
    if unknown:
        print(f"ignored {len(unknown)} verdict(s) that are not y/n: {sorted(set(unknown))[:5]}")
    if not pairs:
        raise SystemExit(f"No graded rows in {path}. Open it, type y or n in the 'your_verdict (y/n)' column for "
                         f"each row (y = the model answer means the same as the reference), save, and run again.")
    if len(pairs) < len(rows):
        print(f"note: {len(pairs)} of {len(rows)} rows graded; agreement uses the graded rows only")
    n = len(pairs)
    acc = sum(h == j for h, j, _ in pairs) / n
    ph, pj = sum(h for h, _, _ in pairs) / n, sum(j for _, j, _ in pairs) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"graded: {n} | agreement: {100 * acc:.1f}% | Cohen's kappa: {kappa:.2f}")
    print(f"you said correct: {100 * ph:.1f}% | judge said correct: {100 * pj:.1f}%")
    for h, j, row in pairs:
        if h != j:
            print(f"  DISAGREE item {row['item']}: you={'y' if h else 'n'} judge={'y' if j else 'n'} | "
                  f"ref: {row['reference_answer'][:60]} | answer: {row['model_answer'][:60]}")
    return acc, kappa


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "agree"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--root", default="data/labels")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run_dir = Path(a.root) / a.run
    return export(run_dir, a.n, a.seed) if a.cmd == "export" else agree(run_dir)


if __name__ == "__main__":
    main()
