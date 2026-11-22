"""Build style-matched English->Arabic question sets for the complexity head.

  python scripts/translate_questions.py --kind hotpot --n 40      # C2 (multi-hop), cheap trial
  python scripts/translate_questions.py --kind nq     --n 40      # C1 (single-hop), cheap trial
  python scripts/translate_questions.py --kind hotpot --n 3000    # full run (same for nq)

Both kinds go through the IDENTICAL prompt and model, so "translated style" cannot separate C1 from C2.
Output: data/raw/translated/{hotpot,nq}_ar.jsonl  +  *_review.csv (100 rows for you to validate).
Requires: OPENAI_API_KEY and ARROUTE_LLM_MODEL (see arroute/llm.py). Every call is cached on disk.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

SOURCES = {   # kind -> (hf dataset, config, split, question col, label, filter)
    "hotpot": ("hotpotqa/hotpot_qa", "distractor", "train", "question", "C2"),
    "nq": ("google-research-datasets/nq_open", None, "train", "question", "C1"),
}

PROMPT = (
    "Translate each English question below into clear, natural Modern Standard Arabic.\n"
    "Rules: keep every question standalone and keep its meaning exactly; write proper names in their "
    "common Arabic form; do NOT answer the questions; do NOT add or drop information.\n"
    "Return ONLY a JSON array of strings, with exactly the same number of items and the same order.\n\n"
    "Questions:\n")


def looks_arabic(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum("\u0600" <= c <= "\u06FF" for c in letters) / len(letters) > 0.5


def parse_json_list(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(t[t.find("["):t.rfind("]") + 1])


def translate_batch(questions, complete_fn):
    """-> list of Arabic strings, or None if the model's answer is unusable."""
    try:
        out = parse_json_list(complete_fn(PROMPT + json.dumps(questions, ensure_ascii=False)))
    except Exception:                                    # noqa: BLE001
        return None
    if len(out) != len(questions) or not all(isinstance(x, str) and looks_arabic(x) for x in out):
        return None
    return [x.strip() for x in out]


def translate_all(questions, complete_fn, batch=20):
    """Batches first; a bad batch falls back to one-by-one so a single failure costs one question."""
    results, failed = {}, []
    for i in range(0, len(questions), batch):
        chunk = questions[i:i + batch]
        out = translate_batch(chunk, complete_fn)
        if out is None:
            out = []
            for q in chunk:
                single = translate_batch([q], complete_fn)
                out.append(single[0] if single else None)
        for q, a in zip(chunk, out):
            if a is None:
                failed.append(q)
            else:
                results[q] = a
    return results, failed


def sample_english(rows, n, seed, min_words=4, max_words=40):
    """rows: iterable of (question, meta dict). Dedupes, filters by length, seeded sample."""
    seen, pool = set(), []
    for q, meta in rows:
        q = " ".join(str(q).split())
        key = q.lower()
        if key in seen or not (min_words <= len(q.split()) <= max_words):
            continue
        seen.add(key)
        pool.append((q, meta))
    Random(seed).shuffle(pool)
    return pool[:n]


def hf_rows(kind, seed, want):
    from datasets import load_dataset
    name, cfg, split, col, _ = SOURCES[kind]
    ds = load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=seed, buffer_size=20000)
    got = 0
    for row in ds:
        if kind == "hotpot" and row.get("level") == "easy":     # keep medium/hard multi-hop
            continue
        yield row[col], {"type": row.get("type"), "level": row.get("level")}
        got += 1
        if got >= want:
            return


def write_outputs(kind, pairs, model, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"{kind}-{i}", "kind": kind, "question_en": q, "question_ar": ar, "model": model, **meta}
            for i, (q, ar, meta) in enumerate(pairs)]
    with (outdir / f"{kind}_ar.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (outdir / f"{kind}_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_en", "question_ar", "translation_ok (y/n)", "corrected_ar"])
        for r in Random(0).sample(rows, min(100, len(rows))):
            w.writerow([r["id"], r["question_en"], r["question_ar"], "", ""])
    return rows


def main(argv=None, complete_fn=None, rows=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=SOURCES, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="data/raw/translated")
    a = ap.parse_args(argv)
    if complete_fn is None:
        from arroute.llm import complete, DEFAULT_MODEL
        complete_fn, model = complete, DEFAULT_MODEL
    else:
        model = "fake"
    src_rows = rows if rows is not None else hf_rows(a.kind, a.seed, a.n * 4)
    sample = sample_english(src_rows, a.n, a.seed)
    print(f"[{a.kind}] sampled {len(sample)} English questions; translating with model={model} ...")
    done, failed = translate_all([q for q, _ in sample], complete_fn, a.batch)
    pairs = [(q, done[q], meta) for q, meta in sample if q in done]
    rows_out = write_outputs(a.kind, pairs, model, a.out)

    en = [len(r["question_en"].split()) for r in rows_out]
    ar = [len(r["question_ar"].split()) for r in rows_out]
    print(f"[{a.kind}] translated {len(rows_out)} | failed {len(failed)} | words EN median {statistics.median(en):.0f}"
          f" | words AR median {statistics.median(ar):.0f}, p90 {sorted(ar)[int(0.9 * (len(ar) - 1))]}")
    for r in Random(1).sample(rows_out, min(5, len(rows_out))):
        print(f"  EN: {r['question_en']}\n  AR: {r['question_ar']}\n")
    print(f"Review file: {Path(a.out) / (a.kind + '_review.csv')}  (check ~100 rows, mark translation_ok y/n)")
    return rows_out


if __name__ == "__main__":
    main()
