"""Phase 1, step 1: raw datasets -> unified JSONL in data/processed/.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max_arabicaqa 20000     # subsample ArabicaQA (seeded)
Prints per-source statistics and runs leakage / sanity checks. Nothing is hidden:
any loader that fails prints WHY and the others still run.
"""
import argparse
import random
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.data import LOADERS, write_jsonl  # noqa: E402
from arroute.data.loaders import PASSAGE_REF  # noqa: E402


def stats(name, exs, passages):
    print(f"\n=== {name}: {len(exs):,} examples | {len(passages):,} passages ===")
    for label, key in (("variety", lambda e: e.variety), ("complexity", lambda e: e.complexity),
                       ("split", lambda e: e.split or "-"), ("domain", lambda e: e.domain)):
        print(f"  {label:<10}", dict(Counter(key(e) for e in exs).most_common()))
    missing = sum(1 for e in exs for g in e.gold_passage_ids if g not in passages)
    print(f"  gold passage ids missing from passages: {missing}")


def length_stats(name, exs):
    """Question length by source: a big gap between classes is a shortcut a router can learn."""
    w = sorted(len(e.question.split()) for e in exs)
    print(f"  question length (words): median {statistics.median(w):.0f} | "
          f"mean {statistics.mean(w):.1f} | p90 {w[int(0.9 * (len(w) - 1))]}")


def passage_leak(name, exs):
    """Gold passages that are used by examples in more than one original split."""
    by = defaultdict(set)
    for e in exs:
        if e.split:
            for g in e.gold_passage_ids:
                by[g].add(e.split)
    if by:
        n = sum(1 for s in by.values() if len(s) > 1)
        print(f"  gold passages shared across splits: {n:,} of {len(by):,}")


def passage_refs(name, exs):
    """Questions that point at 'the passage' cannot work as standalone retrieval queries, and a
    skewed rate across dialects would let a classifier cheat on generation-template style."""
    pat = PASSAGE_REF
    by = defaultdict(lambda: [0, 0])
    for e in exs:
        by[e.variety][1] += 1
        by[e.variety][0] += bool(pat.search(e.question))
    rates = {v: f"{100 * a / b:.1f}%" for v, (a, b) in sorted(by.items()) if b}
    if any(a for a, _ in by.values()):
        print(f"  questions mentioning 'the passage/text' by variety: {rates}")


def leakage(name, exs):
    """Identical question text appearing in more than one original split."""
    by_text = defaultdict(set)
    for e in exs:
        by_text[e.question].add(e.split)
    dup = sum(1 for s in by_text.values() if len(s) > 1)
    if dup:
        print(f"  WARNING leakage: {dup} identical questions occur in >1 split ({name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max_arabicaqa", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)

    ok, failed = [], []
    for name, fn in LOADERS.items():
        try:
            exs, passages = fn(Path(args.raw))
            if name == "arabicaqa" and args.max_arabicaqa and len(exs) > args.max_arabicaqa:
                random.Random(args.seed).shuffle(exs)
                exs = exs[:args.max_arabicaqa]
                keep = {g for e in exs for g in e.gold_passage_ids}
                passages = {k: v for k, v in passages.items() if k in keep}
                print(f"[arabicaqa] subsampled to {len(exs):,} (seed {args.seed})")
            ids = Counter(e.id for e in exs)
            dups = [i for i, c in ids.items() if c > 1]
            assert not dups, f"duplicate ids in {name}: {dups[:3]}"
            write_jsonl(exs, out / f"{name}.jsonl")
            write_jsonl(passages.values(), out / f"{name}_passages.jsonl")
            stats(name, exs, passages)
            leakage(name, exs)
            passage_refs(name, exs)
            length_stats(name, exs)
            passage_leak(name, exs)
            r = random.Random(args.seed).sample(exs, min(3, len(exs)))
            for e in r:
                print(f"  e.g. [{e.variety}/{e.complexity}] {e.question[:90]}")
            ok.append(name)
        except FileNotFoundError as ex:
            print(f"\n[{name}] not present, skipping ({ex})")
        except Exception as ex:                      # noqa: BLE001
            failed.append(name)
            print(f"\n!!! {name} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc(limit=2)

    print(f"\nOK: {ok}\nFAILED: {failed}")
    print(f"Files in {out}/")


if __name__ == "__main__":
    main()
