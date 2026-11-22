"""Phase 4a report: compare router runs with always-C0/C1/C2 and the oracle, using the stored outcomes.

  python scripts/eval_router.py                         # every run in runs_router/
  python scripts/eval_router.py --runs runs_router/ngram runs_router/bert-base-arabertv02_seed0

No API calls: whether each strategy was right, and what it cost, is already in labels.jsonl.
Writes router_report.txt, router_report.json and router_curve.png next to the runs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.outcome import load_outcomes, save_json      # noqa: E402
from arroute.router.route_report import build_report, load_run, plot_curve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels/main_full/labels.jsonl")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--root", default="runs_router")
    ap.add_argument("--none_as", default=None, help="only affects the label-quality section")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default: <root>/router_report")
    a = ap.parse_args()

    recs = load_outcomes(a.labels)
    dirs = a.runs or sorted(str(p.parent) for p in Path(a.root).glob("*/oof.npz"))
    if not dirs:
        raise SystemExit(f"no runs found under {a.root} - run scripts/train_router.py first")
    runs = [load_run(d, recs) for d in dirs]
    text, js = build_report(recs, runs, none_as=a.none_as, reps=a.reps)
    print(text)
    stem = Path(a.out or Path(a.root) / "router_report")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    save_json(js, stem.with_suffix(".json"))
    if plot_curve(recs, runs, stem.with_name(stem.name.replace("report", "curve")).with_suffix(".png")):
        print(f"\nplot: {stem.with_name(stem.name.replace('report', 'curve')).with_suffix('.png')}")
    print(f"report: {stem.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
