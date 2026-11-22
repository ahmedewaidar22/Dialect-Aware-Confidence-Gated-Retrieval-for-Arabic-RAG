"""Phase 2a: aggregate runs/*/metrics.json into mean +- std tables next to the cheap baselines.

  python scripts/summarize_runs.py --task dialect
  python scripts/summarize_runs.py --task complexity
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def ms(vals, nd=3):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "-"
    return f"{v.mean():.{nd}f}" if len(v) == 1 else f"{v.mean():.{nd}f}+-{v.std(ddof=1):.{nd}f}"


def load_groups(runs, task):
    groups = defaultdict(list)
    for f in sorted(Path(runs).glob(f"{task}__*/metrics.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        key = r["config"]["model"].split("/")[-1] + (" +ArDQA-dev" if r["config"].get("add_ardqa_dev") else "")
        groups[key].append(r)
    return groups


def summarize(task, runs, splits):
    base_f = Path(splits) / "baselines.json"
    base = json.loads(base_f.read_text(encoding="utf-8")) if base_f.exists() else {}
    groups = load_groups(runs, task)
    if not groups:
        print(f"no runs found for task '{task}' in {runs}/")
        return
    for key, rs in groups.items():
        names = list(rs[0]["eval"])
        print(f"\n=== {task} | {key} | {len(rs)} seed(s) | best epochs {[r['best_epoch'] for r in rs]}"
              f" | temperature {ms([r['temperature'] for r in rs], 2)} ===")
        if task == "dialect":
            print(f"  {'eval set':<34}{'n':>6}  {'accuracy':<14}{'F1(egy)':<14}{'ECE raw':<8}{'ECE cal':<9}"
                  f"| n-gram acc | heuristic acc")
            for n in names:
                g = lambda k, m="raw": [r["eval"][n][m][k] for r in rs]
                b = base.get("dialect", {}).get(n.split(" [")[0], {})
                print(f"  {n:<34}{rs[0]['eval'][n]['raw']['n']:>6}  {ms(g('accuracy')):<14}{ms(g('f1_pos')):<14}"
                      f"{ms(g('ece'), 3):<8}{ms(g('ece', 'calibrated'), 3):<9}"
                      f"| {b.get('ngram_acc', float('nan')):>10.3f} | {b.get('heuristic_acc', float('nan')):>13.3f}")
        else:
            print(f"  {'eval set':<30}{'n':>6}  {'bal.acc':<14}{'C2 recall':<14}{'FPR':<14}{'ECE raw':<8}{'ECE cal':<9}"
                  f"| n-gram bal.acc | words-only")
            for n in names:
                g = lambda k, m="raw": [r["eval"][n][m][k] for r in rs]
                b = base.get("complexity_clean" if "clean" in n else "complexity", {})
                print(f"  {n:<30}{rs[0]['eval'][n]['raw']['n']:>6}  {ms(g('balanced_accuracy')):<14}"
                      f"{ms(g('recall_pos')):<14}{ms(g('fpr')):<14}{ms(g('ece')):<8}{ms(g('ece', 'calibrated')):<9}"
                      f"| {b.get('ngram_bal_acc', float('nan')):>14.3f} | {b.get('words_bal_acc', float('nan')):>10.3f}")
            leg = "complexity_test"
            if leg in names:
                print("  legacy stress test, share predicted C2 by source (C1 sources: lower is better; C2 source: higher):")
                srcs = rs[0]["eval"][leg]["raw"].get("pos_rate_by_source", {})
                truth = {"arabicaqa": "C1 native", "mafqa_sub": "C1 fatwa", "mafqa": "C2 fatwa"}
                for s_ in srcs:
                    vals = [r["eval"][leg]["raw"]["pos_rate_by_source"].get(s_) for r in rs]
                    print(f"    {s_:<12}({truth.get(s_, '?'):<10}) predicted C2: {ms(vals)}")
    print("\nECE = calibration error (0 is perfect); 'cal' = after temperature scaling fitted on the select set.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["dialect", "complexity"], required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--splits", default="data/splits")
    a = ap.parse_args()
    summarize(a.task, a.runs, a.splits)
