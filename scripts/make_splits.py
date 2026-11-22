"""Phase 1, step 2: leakage-safe splits.  data/processed/*.jsonl  ->  data/splits/*.jsonl

Rules (each one comes from a problem seen in the build report):
  * ArabicRAGB: its official split shares 56% of passages across train/test -> re-split BY PASSAGE.
  * ArabicaQA: official splits are passage-disjoint, but 284 identical questions cross splits
    -> drop from dev/test any question text that is in train.
  * MAFQA originals + their sub-questions: split BY original_index so they never straddle splits.
  * ArDQA (human-translated) and the pilot are EXTERNAL tests: never trained on.
Fails loudly (assert) if any leakage remains.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles/redirects default to a legacy code page
from arroute.data import read_jsonl, write_jsonl  # noqa: E402

PASSAGE_REF = re.compile(r"(الفقرة|الفقره|النص|المقطع|الوثيقة)")


def group_split(items, group_fn, fractions, seed):
    """Assign whole groups to splits, always to the split furthest below its target size."""
    groups = defaultdict(list)
    for e in items:
        groups[group_fn(e)].append(e)
    keys = sorted(groups)
    Random(seed).shuffle(keys)
    out, counts, total = [[] for _ in fractions], [0] * len(fractions), len(items)
    for k in keys:
        i = max(range(len(fractions)), key=lambda j: fractions[j] * total - counts[j])
        out[i].extend(groups[k])
        counts[i] += len(groups[k])
    return out


def dedupe(items):
    seen, out = set(), []
    for e in items:
        if e.question not in seen:
            seen.add(e.question)
            out.append(e)
    return out


def drop_overlap(items, reference):
    """Drop items whose question text occurs in `reference`. Group-aware: if one member of a
    parallel group (e.g. one of ArDQA's 5 varieties) overlaps, the WHOLE group goes, so that
    surviving groups keep their MSA/dialect counterparts for paired comparisons."""
    ref = {e.question for e in reference}
    bad = {e.parallel_group for e in items if e.parallel_group and e.question in ref}
    kept = [e for e in items if e.question not in ref
            and not (e.parallel_group and e.parallel_group in bad)]
    return kept, len(items) - len(kept)


def subsample(items, cap, seed):
    if cap and len(items) > cap:
        items = list(items)
        Random(seed).shuffle(items)
        items = items[:cap]
    return items


def assert_disjoint(splits, key, what):
    names = list(splits)
    sets = {n: {key(e) for e in splits[n]} for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sets[a] & sets[b]
            assert not both, f"LEAK ({what}): {len(both)} shared between {a} and {b}, e.g. {next(iter(both))}"


def pgroup(e):
    return e.gold_passage_ids[0] if e.gold_passage_ids else e.id


def build_dialect(P, seed):
    rag = dedupe([e for e in P["arabicragb"] if not PASSAGE_REF.search(e.question)])
    train, dev, test = group_split(rag, pgroup, (0.8, 0.1, 0.1), seed)
    out = {"dialect_train": train, "dialect_dev": dev, "dialect_test_ragb": test}
    ext = {"dialect_dev_ardqa": [e for e in P["ardqa"] if e.split == "dev"],
           "dialect_test_ardqa": [e for e in P["ardqa"] if e.split == "test"],
           "dialect_test_pilot": list(P["pilot"])}
    seen = train + dev + test                          # every ArabicRAGB split
    for name, items in ext.items():                    # dict order: ardqa dev, ardqa test, pilot
        out[name], dropped = drop_overlap(items, seen)
        if dropped:
            print(f"  {name}: dropped {dropped} examples whose question text already occurs earlier")
        seen = seen + out[name]                        # later external sets are checked against earlier ones
    assert_disjoint({k: out[k] for k in ("dialect_train", "dialect_dev", "dialect_test_ragb")}, pgroup, "passage")
    assert_disjoint(out, lambda e: e.question, "question text")
    return out


def build_complexity(P, seed, cap_c1_train, cap_c1_eval):
    ara = {s: [e for e in P["arabicaqa"] if e.split == s] for s in ("train", "dev", "test")}
    ara["train"] = dedupe(ara["train"])
    ara["dev"], n1 = drop_overlap(dedupe(ara["dev"]), ara["train"])
    ara["test"], n2 = drop_overlap(dedupe(ara["test"]), ara["train"] + ara["dev"])
    print(f"  arabicaqa: removed {n1} dev and {n2} test questions that duplicate earlier splits")
    ara["train"] = subsample(ara["train"], cap_c1_train, seed)
    ara["dev"] = subsample(ara["dev"], cap_c1_eval, seed)
    ara["test"] = subsample(ara["test"], cap_c1_eval, seed)

    maf = P["mafqa"] + P["mafqa_sub"]                    # split by original_index, together
    m_tr, m_dv, m_te = group_split(maf, lambda e: e.parallel_group, (0.7, 0.15, 0.15), seed)

    out = {"complexity_train": ara["train"] + m_tr, "complexity_dev": ara["dev"] + m_dv,
           "complexity_test": ara["test"] + m_te}
    assert_disjoint({"train": m_tr, "dev": m_dv, "test": m_te}, lambda e: e.parallel_group, "MAFQA group")
    assert_disjoint(out, lambda e: e.question, "question text")
    return out


def length_match(c1, c2, seed, cap=40):
    """Equalise the word-count histogram of the two classes: in every length bin keep the same number
    of questions from each class (random choice). Afterwards length carries no class information."""
    rng = Random(seed)
    by1, by2 = defaultdict(list), defaultdict(list)
    for e in c1:
        by1[min(len(e.question.split()), cap)].append(e)
    for e in c2:
        by2[min(len(e.question.split()), cap)].append(e)
    k1, k2 = [], []
    for n in sorted(set(by1) | set(by2)):
        a, b = list(by1.get(n, [])), list(by2.get(n, []))
        k = min(len(a), len(b))
        rng.shuffle(a)
        rng.shuffle(b)
        k1 += a[:k]
        k2 += b[:k]
    return k1, k2


def _hist(items, edges=(6, 8, 10, 12, 14, 17, 10_000)):
    out, lo = [], 0
    for hi in edges:
        out.append(sum(lo <= len(e.question.split()) <= hi for e in items))
        lo = hi + 1
    return out


def build_complexity_clean(P, seed, fractions=(0.7, 0.15, 0.15)):
    """C1 = translated NQ-Open, C2 = translated HotpotQA: same domain, same translation pipeline,
    then length-matched. UNVALIDATED until the translation review CSV has been checked."""
    c1, c2 = dedupe(P.get("nq_ar", [])), dedupe(P.get("hotpot_ar", []))
    if not c1 or not c2:
        print("  (translated sets not present: skipping complexity_clean)")
        return {}
    both = {e.question for e in c1} & {e.question for e in c2}      # same text, two labels = conflict
    if both:
        print(f"  removed {len(both)} questions that appear in BOTH classes (label conflict)")
        c1, c2 = [e for e in c1 if e.question not in both], [e for e in c2 if e.question not in both]
    print(f"  before matching: C1 {len(c1):,} | C2 {len(c2):,} | words<=6/8/10/12/14/17/18+")
    print(f"    C1 {_hist(c1)}\n    C2 {_hist(c2)}")
    m1, m2 = length_match(c1, c2, seed)
    print(f"  after matching : C1 {len(m1):,} | C2 {len(m2):,}\n    C1 {_hist(m1)}\n    C2 {_hist(m2)}")
    parts = {}
    for name, items in (("C1", m1), ("C2", m2)):
        items = list(items)
        Random(seed).shuffle(items)
        n = len(items)
        a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
        parts[name] = (items[:a], items[a:b], items[b:])
    out = {f"complexity_clean_{sp}": parts["C1"][i] + parts["C2"][i]
           for i, sp in enumerate(("train", "dev", "test"))}
    assert_disjoint(out, lambda e: e.question, "question text")
    return out


def summarize(name, items):
    cnt = Counter((e.source, e.variety if name.startswith('dialect') else e.complexity) for e in items)
    print(f"  {name:<22} {len(items):>7,}   " + ", ".join(f"{s}/{c}:{n:,}" for (s, c), n in sorted(cnt.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap_c1_train", type=int, default=20000)
    ap.add_argument("--cap_c1_eval", type=int, default=2000)
    a = ap.parse_args()
    P = {n: read_jsonl(Path(a.processed) / f"{n}.jsonl")
         for n in ("arabicaqa", "arabicragb", "ardqa", "mafqa", "mafqa_sub", "pilot")}
    for n in ("hotpot_ar", "nq_ar"):                       # optional: only if translation was run
        f = Path(a.processed) / f"{n}.jsonl"
        if f.exists():
            P[n] = read_jsonl(f)
    print("DIALECT splits")
    d = build_dialect(P, a.seed)
    print("COMPLEXITY splits")
    c = build_complexity(P, a.seed, a.cap_c1_train, a.cap_c1_eval)
    print("COMPLEXITY (clean, length-matched) splits")
    cc = build_complexity_clean(P, a.seed)
    print("\nWritten:")
    for name, items in {**d, **c, **cc}.items():
        write_jsonl(items, Path(a.out) / f"{name}.jsonl")
        summarize(name, items)
    print("\nAll leakage assertions passed.")


if __name__ == "__main__":
    main()
