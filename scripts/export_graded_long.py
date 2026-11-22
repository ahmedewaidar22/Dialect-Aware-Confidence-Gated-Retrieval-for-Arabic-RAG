"""Export every system's answer to every test question, with the judge's verdict, for the human judge validation.

  python scripts/export_graded_long.py            # run from the repo root, after Phases 8 and 9 are done

Writes runs_final/graded_long.csv with one row per (question, system):
  qid, system, question, reference, answer, judge, set
Systems: standard_rag, always_c2, adaptive_rag, router_v1_quality, final   (Egyptian input, the 1,000 test questions).
Reads only saved results (no API calls, no cost). It mirrors scripts/final_report.py, and prints a consistency check
against the verdicts that final_report.py uses, so a mismatch is visible before anything is annotated.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import eval_phase8 as E                              # noqa: E402
import final_report as F                             # noqa: E402
from arroute.router.final_eval import run_policy     # noqa: E402
from arroute.router.outcome import IDK               # noqa: E402

STRAT = {0: "C0", 1: "C1", 2: "C2"}
SYSTEM_NAMES = {"standard_rag": "Standard RAG (C1, hybrid top-5)", "always_c2": "Iterative RAG (always C2)",
                "adaptive_rag": "Adaptive-RAG (text classifier)", "router_v1_quality": "Router v1 (quality)",
                "final": F.FINAL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8", default="runs_phase8")
    ap.add_argument("--v2", default="runs_v2")
    ap.add_argument("--router7", default="runs_router7")
    ap.add_argument("--models", default="models")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--out", default="runs_final/graded_long.csv")
    a = ap.parse_args()

    ids, A, an8, st8, an9, sets, pol = F.build(a)
    items = {json.loads(x)["id"]: json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()}
    S, _ = F.systems("egy", ids, A, an8, st8, an9, pol)
    t, r = A["egy_t"], A["egy_r"]
    route_q = run_policy(r, "router", thresholds=pol["thresholds"]["quality"])["route"]
    adaptive = t["probs3"].argmax(1)

    def res(i, var, strat):
        st = st8[f"{i}|{var}"]
        return an8[E.akey(i, st)]["results"][strat]

    rows = []
    for j, i in enumerate(ids):
        it = items[i]
        q = it["q_egy"]
        ref = " | ".join(str(x) for x in (it.get("refs") or [it.get("answer")]) if x)
        picks = {"standard_rag": res(i, "egy_t", "C1"), "always_c2": res(i, "egy_t", "C2"),
                 "adaptive_rag": res(i, "egy_t", STRAT[int(adaptive[j])])}
        rt = str(route_q[j])
        picks["router_v1_quality"] = res(i, "egy_r", "C0" if rt == "C0" else "C1" if rt == "C1" else "C2")
        c1r = an9[f"test|{i}|egy_r"]["results"]["C1R"]
        idk = IDK in (c1r.get("answer") or "") or not (c1r.get("answer") or "").strip()
        picks["final"] = res(i, "egy_r", "C2") if idk else c1r
        for name, x in picks.items():
            rows.append({"qid": i, "system": name, "question": q, "reference": ref, "answer": x.get("answer") or "",
                         "judge": int(bool(x.get("correct"))), "set": sets[j]})

    print("consistency with final_report.py (number of questions where the verdict differs):")
    for name, long_name in SYSTEM_NAMES.items():
        mine = np.array([r_["judge"] for r_ in rows if r_["system"] == name], bool)
        print(f"  {name:20s} {int((mine != S[long_name][0].astype(bool)).sum())} of {len(mine)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
