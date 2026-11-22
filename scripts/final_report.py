"""Phase 10: ONE consolidated results report for the thesis (free, local, no API calls).

  python scripts/final_report.py

Reads the saved results of Phases 7-9 (runs_router7, runs_phase8, runs_v2, models/*) and writes runs_final/:
  thesis_tables.md        every results table in Markdown (paste into the thesis), with 95% CIs and p-values
  thesis_results.json     the same numbers, machine-readable
  per_question.csv        one row per test question x system (for your own checks / error analysis)
  cost_accuracy_egy.png   cost vs accuracy, Egyptian input (thesis figure)
  cost_accuracy_msa.png   the same for MSA input
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

import eval_phase8 as E                                             # noqa: E402

from arroute.router import v2 as V                                  # noqa: E402
from arroute.router.final_eval import arrays, paired, run_policy    # noqa: E402
from arroute.router.metrics import wilson                           # noqa: E402
from arroute.router.outcome import IDK, PRICE_IN, PRICE_OUT, fixed_frontier, frontier_acc, oracle_choice   # noqa: E402

SETS = ("ardqa", "pilot", "heldout")
IDK_KEY = "'I don't know' only"
FINAL = "FINAL: rewrite + C1 on reranked top-k, 'I don't know' -> C2"


def set_of(item_id, items):
    if item_id in items:
        return items[item_id]["set"]
    return "heldout" if ":held:" in item_id else ("ardqa" if ":ardqa:" in item_id else "pilot")


def build(a):
    p8, p9 = Path(a.phase8), Path(a.v2)
    st8, an8 = E.read_jsonl(p8 / "states.jsonl", "key"), E.read_jsonl(p8 / "answers.jsonl", "akey")
    rs8 = E.read_jsonl(p8 / "rescue.jsonl", "akey")
    an9 = E.read_jsonl(p9 / "answers.jsonl", "key")
    items = {}
    if Path(a.test).exists():
        items = {json.loads(x)["id"]: json.loads(x) for x in Path(a.test).read_text(encoding="utf-8").splitlines() if x.strip()}
    ids = sorted({k.split("|")[0] for k in st8})
    ids = [i for i in ids if all(f"{i}|{v}" in st8 and E.akey(i, st8[f"{i}|{v}"]) in an8 for v in E.VARIANTS)
           and all(f"test|{i}|{v}" in an9 for v in ("egy_r", "msa_r"))]
    if not ids:
        raise SystemExit("no test question has both Phase 8 and Phase 9 results")
    U = {v: [E.unit(st8[f"{i}|{v}"], an8[E.akey(i, st8[f"{i}|{v}"])], rs8.get(E.akey(i, st8[f"{i}|{v}"])),
                    (PRICE_IN, PRICE_OUT)) for i in ids] for v in E.VARIANTS}
    A = {v: arrays(U[v]) for v in E.VARIANTS}
    sets = np.array([set_of(i, items) for i in ids])
    pol = json.loads((Path(a.models) / "complexity_router" / "policy.json").read_text(encoding="utf-8"))
    return ids, A, an8, st8, an9, sets, pol


def systems(lang, ids, A, an8, st8, an9, pol):
    """-> ordered dict name -> (hit, cost, calls) per question, all on the SAME questions."""
    t, r = A[f"{lang}_t"], A[f"{lang}_r"]
    ix = np.arange(len(ids))
    S = {}
    S["No retrieval (C0)"] = (t["correct"][:, 0], t["cost"][:, 0], t["calls"][:, 0])
    S["Standard RAG (C1, hybrid top-5)"] = (t["correct"][:, 1], t["cost"][:, 1], t["calls"][:, 1])
    S["Iterative RAG (always C2)"] = (t["correct"][:, 2], t["cost"][:, 2], t["calls"][:, 2])
    ch = t["probs3"].argmax(1)
    S["Adaptive-RAG (text classifier)"] = (t["correct"][ix, ch], t["cost"][ix, ch], t["calls"][ix, ch])
    for m in ("match_c1", "quality"):
        p = run_policy(r, "router", thresholds=pol["thresholds"][m])
        S[f"Router v1 ({m})"] = (p["hit"], p["cost"], p["calls"])
    rw, rwc = r["rewrite_cost"], r["rewrote"]
    S["rewrite + C1 (hybrid top-5)"] = (r["correct"][:, 1], r["cost"][:, 1] + rw, r["calls"][:, 1] + rwc)
    idk_h = np.array([IDK in (an8[E.akey(i, st8[f"{i}|{lang}_r"])]["results"]["C1"]["answer"] or "") for i in ids])
    S["rewrite + C1 (hybrid top-5), 'I don't know' -> C2"] = (
        np.where(idk_h, r["correct"][:, 2], r["correct"][:, 1]), r["cost"][:, 1] + idk_h * r["cost"][:, 2] + rw,
        r["calls"][:, 1] + idk_h * r["calls"][:, 2] + rwc)
    c1r = [an9[f"test|{i}|{lang}_r"]["results"]["C1R"] for i in ids]
    h1 = np.array([x["correct"] for x in c1r], float)
    k1 = np.array([V.tok_cost(x) for x in c1r])
    idk = np.array([IDK in (x["answer"] or "") or not (x["answer"] or "").strip() for x in c1r])
    S["rewrite + C1 on reranked top-k"] = (h1, k1 + rw, 1 + rwc)
    S[FINAL] = (np.where(idk, r["correct"][:, 2], h1), k1 + idk * r["cost"][:, 2] + rw, 1 + idk * r["calls"][:, 2] + rwc)
    orc = oracle_choice(r["correct"])
    S["Oracle routing over C0/C1/C2 (upper bound)"] = (r["correct"][ix, orc], r["cost"][ix, orc] + rw, r["calls"][ix, orc] + rwc)
    return S, idk


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def table(S, sets, hull, base_names, extra=None):
    n = len(sets)
    head = ["System", "Accuracy", "95% CI", "$ / 1k q", "LLM calls / q"] + [f"Δ vs {b}" for b in base_names]
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = []
    for name, (h, c, k) in S.items():
        acc = h.mean()
        lo, hi = wilson(int(h.sum()), n)
        cells = [f"**{name}**" if name == FINAL else name, f"{100 * acc:.1f}%", f"{100 * lo:.1f}–{100 * hi:.1f}",
                 f"{1000 * c.mean():.3f}", f"{k.mean():.2f}"]
        cmp = {}
        for b in base_names:
            if b == name:
                cells.append("–")
                continue
            d, dlo, dhi, p = paired(h.astype(bool), S[b][0].astype(bool))
            cells.append(f"{100 * d:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}] p={fmt_p(p)}")
            cmp[b] = {"diff": d, "ci": [dlo, dhi], "p": p}
        L.append("| " + " | ".join(cells) + " |")
        rows.append({"system": name, "accuracy": float(acc), "ci": [lo, hi], "usd_per_1k": float(1000 * c.mean()),
                     "calls": float(k.mean()), "vs_mix": float(acc - frontier_acc(hull, 1000 * c.mean())),
                     "by_set": {s: {"accuracy": float(h[sets == s].mean()), "usd_per_1k": float(1000 * c[sets == s].mean())}
                                for s in SETS if (sets == s).any()}, "compare": cmp})
    for name, x in (extra or {}).items():
        L.append(f"| {name} | {100 * x['accuracy']:.1f}% | {100 * x['ci'][0]:.1f}–{100 * x['ci'][1]:.1f} | "
                 f"{x['usd_per_1k']:.3f} | {x['calls']:.2f} | " + " | ".join("see Phase 9 report" for _ in base_names) + " |")
    return L, rows


def figure(rows, hull, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange, gray = "#2a78d6", "#eb6834", "#8a8984"
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    hx, hy = [c for c, _ in hull], [100 * a for _, a in hull]
    ax.plot(hx, hy, color=gray, lw=1.5, ls=(0, (4, 3)), zorder=1)
    mid = len(hx) // 2
    ax.annotate("random mix of C0 / C1 / C2\n(no rewrite, no rerank)", ((hx[mid - 1] + hx[mid]) / 2, (hy[mid - 1] + hy[mid]) / 2),
                xytext=(10, -22), textcoords="offset points", ha="left", fontsize=7.5, color=muted)
    # (label, colour, label offset in points, horizontal alignment): offsets keep neighbouring labels apart
    show = {"Standard RAG (C1, hybrid top-5)": ("Standard RAG", gray, (7, 5), "left"),
            "Adaptive-RAG (text classifier)": ("Adaptive-RAG", gray, (-7, 5), "right"),
            "Iterative RAG (always C2)": ("Always C2", gray, (-7, 6), "right"),
            "No retrieval (C0)": ("No retrieval", gray, (7, -3), "left"),
            "Router v1 (match_c1)": ("Router v1 (match_c1)", orange, (7, -10), "left"),
            "Router v1 (quality)": ("Router v1 (quality)", orange, (-7, -13), "right"),
            "rewrite + C1 on reranked top-k": ("rewrite + rerank top-k", blue, (7, -3), "left"),
            FINAL: ("FINAL system", blue, (9, 4), "left")}
    for r in rows:
        if r["system"] not in show:
            continue
        lab, col, off, ha = show[r["system"]]
        x, y = r["usd_per_1k"], 100 * r["accuracy"]
        big = r["system"] == FINAL
        ax.scatter([x], [y], s=110 if big else 54, color=col, edgecolor=surface, linewidth=2, zorder=3)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, ha=ha, zorder=4,
                    color=ink if big else muted, fontweight="bold" if big else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc=surface, ec="none", alpha=0.9))
    ax.set_xlim(0, max(1.0, max(r["usd_per_1k"] for r in rows) * 1.08))
    ax.set_xlabel("LLM cost, $ per 1,000 questions (lower is better)", fontsize=9, color=muted)
    ax.set_ylabel("Accuracy, % (higher is better)", fontsize=9, color=muted)
    ax.set_title(title, fontsize=11, color=ink, loc="left")
    ax.grid(True, color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=t, markersize=7) for c, t in
                       ((gray, "baselines"), (orange, "learned routers (v1)"), (blue, "this thesis: final system"))],
              loc="lower right", fontsize=8, frameon=False, labelcolor=muted)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8", default="runs_phase8")
    ap.add_argument("--v2", default="runs_v2")
    ap.add_argument("--router7", default="runs_router7")
    ap.add_argument("--models", default="models")
    ap.add_argument("--test", default="data/egy_test/egy_test.jsonl")
    ap.add_argument("--out", default="runs_final")
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, A, an8, st8, an9, sets, pol = build(a)
    v2f = Path(a.v2) / "v2_report.json"
    v2js = json.loads(v2f.read_text(encoding="utf-8")) if v2f.exists() else {}
    k = v2js.get("k") or (json.loads((Path(a.v2) / "v2_config.json").read_text(encoding="utf-8"))["k"]
                          if (Path(a.v2) / "v2_config.json").exists() else "?")
    n = len(ids)
    J = {"n": n, "k": k, "sets": {s: int((sets == s).sum()) for s in SETS}}
    M = ["# Results tables (generated by scripts/final_report.py)", "",
         f"Test set: {n:,} questions never used for training or tuning ("
         + ", ".join(f"{s} {int((sets == s).sum())}" for s in SETS) + "). Generator and judge: gpt-4.1-mini. "
         f"Cost = generator tokens at ${PRICE_IN}/${PRICE_OUT} per 1M (incl. the rewrite call). "
         "95% CI: Wilson (accuracy), paired bootstrap (differences). p: exact McNemar (paired). "
         f"Reranked top-k: k = {k}, chosen on training questions only.", ""]

    # ---- 1. dialect head
    dm = Path(a.models) / "dialect_head" / "meta.json"
    if dm.exists():
        ev = json.loads(dm.read_text(encoding="utf-8")).get("eval", {})
        M += ["## Table 1. Dialect head on the Egyptian test set", "",
              "| Test set | n | Egyptian recognised | MSA kept as MSA |", "|---|---|---|---|"]
        for s in ("ardqa", "pilot", "heldout", "ALL"):
            e = ev.get(f"egy_test_{s}")
            if e:
                M.append(f"| {s} | {e['n']} | {100 * e['egy_recall']:.1f}% | {100 * e['msa_specificity']:.1f}% |")
        M.append("")
        J["dialect"] = {s: ev.get(f"egy_test_{s}") for s in ("ardqa", "pilot", "heldout", "ALL")}

    # ---- 2. retrieval query plan (Phase 7)
    qp = Path(a.router7) / "query_plan_report.json"
    if qp.exists():
        q = json.loads(qp.read_text(encoding="utf-8"))["plans"]
        M += ["## Table 2. Retrieval: gold passage in the hybrid top 5, mixed Egyptian + MSA traffic (1,667 questions x 2)", "",
              "| Query plan | Egyptian input | MSA input | All | Searches / q |", "|---|---|---|---|---|"]
        for plan in q.get("ALL inputs", {}):
            eg, ms = q["Egyptian inputs"][plan]["hybrid"], q["MSA inputs"][plan]["hybrid"]
            M.append(f"| {plan} | {100 * eg:.1f}% | {100 * ms:.1f}% "
                     f"| {100 * q['ALL inputs'][plan]['hybrid']:.1f}% | {q['ALL inputs'][plan]['searches']:.2f} |")
        M.append("")
        J["retrieval"] = q

    # ---- 3/4. end to end, Egyptian and MSA (then 5/6)
    T56 = []
    for lang, title, num in (("egy", "Egyptian input", 3), ("msa", "MSA input (same questions)", 4)):
        S, idk = systems(lang, ids, A, an8, st8, an9, pol)
        hull = fixed_frontier(A[f"{lang}_t"]["correct"], A[f"{lang}_t"]["cost"], A[f"{lang}_t"]["calls"])
        extra = {}
        for x in v2js.get("systems", []):
            if x["section"].startswith("EGYPTIAN" if lang == "egy" else "MSA") and x["system"] in ("ROUTER V2 - quality",
                                                                                                  "ROUTER V2 - budget"):
                extra["Router v2, learned cascade (" + x["system"].split(" - ")[1] + ")"] = x
        L, rows = table(S, sets, hull, ["Standard RAG (C1, hybrid top-5)", "Iterative RAG (always C2)"], extra)
        M += [f"## Table {num}. End-to-end answer accuracy and cost, {title}", ""] + L + [
              "", f"The final system escalated to C2 on {100 * idk.mean():.1f}% of questions.", ""]
        J[lang] = {"systems": rows, "escalated": float(idk.mean()),
                   "hull": [[float(c), float(x)] for c, x in hull]}
        d, lo, hi, p = paired(S[FINAL][0].astype(bool), S["Router v1 (quality)"][0].astype(bool))
        J[lang]["final_vs_router_v1_quality"] = {"diff": d, "ci": [lo, hi], "p": p}
        if not a.no_figures:
            try:
                figure(rows, hull, out / f"cost_accuracy_{lang}.png", f"Accuracy vs cost - {title}, {n:,} test questions")
            except ImportError:
                print("matplotlib is not installed: figures skipped (pip install matplotlib)")
        if lang == "egy":
            # ---- 5. per test set (appended after Table 4)
            T56 += ["## Table 5. By test set, Egyptian input (accuracy / $ per 1k questions)", "",
                  "| System | ArDQA (native) | Pilot (native) | Held-out (translated) |", "|---|---|---|---|"]
            for name in ("Standard RAG (C1, hybrid top-5)", "Iterative RAG (always C2)", "Adaptive-RAG (text classifier)",
                         "Router v1 (quality)", "rewrite + C1 on reranked top-k", FINAL):
                h, c, _ = S[name]
                T56.append(f"| {name} | " + " | ".join(f"{100 * h[sets == s].mean():.1f}% / {1000 * c[sets == s].mean():.3f}"
                                                     for s in SETS) + " |")
            T56.append("")
            # ---- 6. ablation ladder
            ladder = [("Standard RAG (C1, hybrid top-5)", "start"), ("rewrite + C1 (hybrid top-5)", "+ dialect rewrite"),
                      ("rewrite + C1 on reranked top-k", f"+ rerank, top-{k} passages"),
                      (FINAL, "+ 'I don't know' -> C2")]
            T56 += ["## Table 6. What each part adds (Egyptian input, paired with the previous step)", "",
                  "| Step | Accuracy | $ / 1k q | Δ accuracy vs previous step | Δ cost |", "|---|---|---|---|---|"]
            prev = None
            J["ablation"] = []
            for name, step in ladder:
                h, c, _ = S[name]
                if prev is None:
                    T56.append(f"| {step} ({name}) | {100 * h.mean():.1f}% | {1000 * c.mean():.3f} | – | – |")
                else:
                    d, lo, hi, p = paired(h.astype(bool), prev[0].astype(bool))
                    T56.append(f"| {step} | {100 * h.mean():.1f}% | {1000 * c.mean():.3f} | {100 * d:+.1f} "
                             f"[{100 * lo:+.1f}, {100 * hi:+.1f}] p={fmt_p(p)} | {100 * (c.mean() / prev[1].mean() - 1):+.0f}% |")
                J["ablation"].append({"step": step, "system": name, "accuracy": float(h.mean()), "usd": float(1000 * c.mean())})
                prev = (h, c)
            T56.append("")

    M += T56
    # ---- 7. confirmation on the calibration questions (not test)
    cal = [x["results"] for kk, x in an9.items() if kk.startswith("calib|") and "C2" in x["results"]]
    if cal:
        c1 = np.array([x["C1R"]["correct"] for x in cal], float)
        c2 = np.array([x["C2"]["correct"] for x in cal], float)
        k1 = np.array([V.tok_cost(x["C1R"]) for x in cal])
        k2 = np.array([V.tok_cost(x["C2"]) for x in cal])
        i1 = np.array([IDK in (x["C1R"]["answer"] or "") or not (x["C1R"]["answer"] or "").strip() for x in cal])
        fin = np.where(i1, c2, c1)
        M += [f"## Table 7. Same ranking on {len(cal)} native Egyptian ArDQA questions that are NOT in the test set", "",
              "| System | Accuracy | $ / 1k q |", "|---|---|---|",
              f"| rewrite + C1 on reranked top-{k} | {100 * c1.mean():.1f}% | {1000 * k1.mean():.3f} |",
              f"| final system | {100 * fin.mean():.1f}% | {1000 * (k1 + i1 * k2).mean():.3f} |",
              f"| always C2 | {100 * c2.mean():.1f}% | {1000 * k2.mean():.3f} |", ""]
        J["calibration"] = {"n": len(cal), "c1r": float(c1.mean()), "final": float(fin.mean()), "c2": float(c2.mean())}

    # ---- 8. detectors (Phase 9)
    if v2js.get("auc"):
        M += ["## Table 8. Can the system tell when an answer is right? (AUC on the test set, Egyptian input)", "",
              "| Test set | v1 text head: 'C0 is enough' | v2: 'C0 is right' | v2: 'C1 is right' | 'I don't know' alone |",
              "|---|---|---|---|---|"]
        for s, d in v2js["auc"].get("egy", {}).items():
            M.append(f"| {s} | {d['v1 text head P(C0)']:.2f} | {d['det0 (C0 right)']:.2f} | {d['det1 (C1R right)']:.2f} "
                     f"| {d[IDK_KEY]:.2f} |")
        M.append("")

    (out / "thesis_tables.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    (out / "thesis_results.json").write_text(json.dumps(J, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    with (out / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        S_e, _ = systems("egy", ids, A, an8, st8, an9, pol)
        names = list(S_e)
        w.writerow(["id", "set"] + [f"{nm} | correct" for nm in names] + [f"{nm} | usd" for nm in names])
        for j, i in enumerate(ids):
            w.writerow([i, sets[j]] + [int(S_e[nm][0][j]) for nm in names] + [f"{S_e[nm][1][j]:.6f}" for nm in names])
    print("\n".join(M))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
