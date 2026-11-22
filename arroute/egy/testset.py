"""Phase 5: the Egyptian test set (torch-free helpers).

Three parts, all with an Egyptian question AND its MSA twin, a reference answer and the gold passage(s):
  ardqa    - ArDQA test: native, human-written Egyptian questions (single-hop)
  pilot    - your 200 validated pilot pairs (single-hop); reference answers extracted from the context
  heldout  - 300 labelled questions (ArabicaQA + HotpotQA, incl. multi-hop) set aside from router
             training and rewritten into Egyptian by the LLM; you check 100 of them
"""
import re
from collections import defaultdict

import numpy as np

from ..retrieval.text import normalize_ar, tokenize_ar

# Words that are typical of Egyptian and (almost) never used in MSA questions, after normalize_ar().
EGY_MARKERS = {"ايه", "ازاي", "ازاى", "فين", "امتي", "امتى", "مين", "ليه", "كام", "ده", "دي", "دا", "دول",
               "اللي", "عشان", "علشان", "بتاع", "بتاعه", "بتاعت", "بتوع", "مش", "بقي", "بقى", "كده", "اوي",
               "خالص", "لسه", "ابتدا", "هيه", "انهي", "انهى", "ازاى", "اتولد", "اتبني", "اتعمل", "اتكتب",
               "اتأسس", "اتاسس", "اتنشر", "اتجوز", "اتوفي", "اتوفى", "اتعرض", "اتصور", "اتلعب"}
_DIGITS = re.compile(r"[0-9٠-٩]+")
_AR2EN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", normalize_ar(s or "").lower())).strip()


def egy_markers(text):
    return sorted({t for t in tokenize_ar(text) if t in EGY_MARKERS})


def numbers(text):
    return set(_DIGITS.findall((text or "").translate(_AR2EN)))


def check_translation(msa, egy, answer=None):
    """Problems with an MSA -> Egyptian rewrite (empty list = looks fine)."""
    problems = []
    if not egy or sum("؀" <= c <= "ۿ" for c in egy) < 3:
        problems.append("not Arabic")
        return problems
    if norm(egy) == norm(msa):
        problems.append("unchanged")
    if not egy_markers(egy):
        problems.append("no Egyptian marker word")
    if numbers(msa) - numbers(egy):
        problems.append("number lost: " + ",".join(sorted(numbers(msa) - numbers(egy))))
    a = norm(answer)
    if a and len(a) >= 3 and f" {a} " in f" {norm(egy)} " and f" {a} " not in f" {norm(msa)} ":
        problems.append("answer leaked")
    if len(egy) > 2.5 * max(len(msa), 10):
        problems.append("much longer than the original")
    return problems


def answer_in_context(answer, context):
    a = norm(answer)
    return bool(a) and f" {a} " in f" {norm(context)} "


def select_heldout(recs, n=300, seed=7):
    """Stratified by (source, label, comparison); identical question texts are kept together."""
    groups = defaultdict(list)
    for r in recs:
        groups[norm(r["question"])].append(r["id"])
    strata = defaultdict(list)
    for key, ids in groups.items():
        r = next(x for x in recs if x["id"] == ids[0])
        strata[(r["source"], r["label"], r.get("qtype") == "comparison")].append(key)
    total = sum(len(v) for v in strata.values())
    rng = np.random.RandomState(seed)
    quota = {k: n * len(v) / total for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[:n - sum(take.values())]:
        take[k] += 1                                    # largest remainder: exactly n groups
    chosen = []
    for k in sorted(strata):
        keys = sorted(strata[k])
        for i in rng.choice(len(keys), min(take[k], len(keys)), replace=False):
            chosen += groups[keys[i]]
    return sorted(chosen)


def review_sample(items, n, seed=11, by="source"):
    """Stratified sample (by `by`) of n items for human review."""
    groups = defaultdict(list)
    for it in items:
        groups[it[by]].append(it)
    rng = np.random.RandomState(seed)
    out = []
    for k in sorted(groups):
        g = groups[k]
        m = round(n * len(g) / len(items))
        out += [g[i] for i in sorted(rng.choice(len(g), min(m, len(g)), replace=False))]
    return out[:n]


YES = {"y", "yes", "1", "true", "ok", "نعم", "ص", "صح"}
NO = {"n", "no", "0", "false", "لا", "خ", "غلط"}


def apply_review(items, rows, row_map):
    """rows: reviewed CSV rows; row_map: row -> (item id, field). Returns (items, stats)."""
    by_id = {it["id"]: it for it in items}
    stats = defaultdict(lambda: {"ok": 0, "fixed": 0, "rejected": 0, "blank": 0})
    for r in rows:
        row = (r.get("row") or "").strip()
        if row not in row_map:
            continue
        iid, field = row_map[row]
        it, kind = by_id[iid], ("translation" if field == "q_egy" else "answer")
        ok, fix = (r.get("ok") or "").strip().lower(), (r.get("fix") or "").strip()
        if ok in YES and not fix:
            it.setdefault("review", {})[field] = "ok"
            stats[kind]["ok"] += 1
        elif fix:
            it[field] = fix
            if field == "answer":
                it["refs"] = [fix]
            it.setdefault("review", {})[field] = "fixed"
            stats[kind]["fixed"] += 1
        elif ok in NO:
            it.setdefault("review", {})[field] = "rejected"
            stats[kind]["rejected"] += 1
        else:
            stats[kind]["blank"] += 1
    return items, {k: dict(v) for k, v in stats.items()}
