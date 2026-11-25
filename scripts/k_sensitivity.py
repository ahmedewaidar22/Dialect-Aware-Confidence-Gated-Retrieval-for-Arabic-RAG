"""Would k have been chosen differently WITHOUT the 667 calibration questions? (no API calls, reads runs_v2/rows.jsonl)

  python scripts/k_sensitivity.py

The audit found that most ArDQA test questions share their gold passage with a calibration question, and k was chosen
on training + calibration questions. This re-applies the SAME rule used in router_v2.choose_k (smallest k >= 2 whose
reranked top-k finds the gold passage as often as hybrid top-5, within 0.5 points) to three question pools:
training only, calibration only, and both (the original choice). If training-only also gives k = 2, the calibration
questions did not decide k.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def choose(fit, reranked):
    hit = lambda ps, g: bool(set(ps) & set(g))                      # noqa: E731
    hyb5 = sum(hit(r["cands"][:5], r["gold"]) for r in fit) / len(fit)
    rows, best = [], None
    for k in range(1, 6):
        rr = sum(hit(reranked(r["cands"], r["rr20"], k), r["gold"]) for r in fit) / len(fit)
        rows.append((k, rr))
        if best is None and k >= 2 and rr >= hyb5 - 0.005:
            best = k
    return hyb5, rows, best or 5


def main():
    from arroute.router.v2 import reranked
    rows = [json.loads(x) for x in Path("runs_v2/rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    fit = [r for r in rows if r.get("split") in ("train", "calib") and r.get("gold") and r.get("rr20")]
    for name, pool in (("training only", [r for r in fit if r["split"] == "train"]),
                       ("calibration only", [r for r in fit if r["split"] == "calib"]),
                       ("training + calibration (original)", fit)):
        if not pool:
            print(f"{name}: no questions")
            continue
        hyb5, table, k = choose(pool, reranked)
        print(f"{name}: {len(pool)} questions | hybrid top-5 finds gold {100 * hyb5:.1f}%")
        print("   " + "  ".join(f"k={kk}: {100 * v:.1f}%" for kk, v in table) + f"   -> chosen k = {k}")


if __name__ == "__main__":
    main()
