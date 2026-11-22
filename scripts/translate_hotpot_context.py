"""Phase 3a: translate HotpotQA items WITH their 10 paragraphs (2 gold + 8 hard distractors) into Arabic.

One API call per item, so a name is spelled the same way in the question, the answer and the paragraphs.
  python scripts/translate_hotpot_context.py --n 10                 # trial (a few cents)
  python scripts/translate_hotpot_context.py --n 1000 --workers 8   # full run; cached, safe to re-run
Output: data/raw/translated/hotpot_ctx_ar.jsonl, hotpot_ctx_review.csv, hotpot_ctx_trial.txt
Needs OPENAI_API_KEY and ARROUTE_LLM_MODEL.
"""
import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

PROMPT = (
    "Translate this question-answering item from English into clear Modern Standard Arabic.\n"
    "Rules:\n"
    "- Translate the question, the answer, and EVERY paragraph (title and text).\n"
    "- Write each person, place, work and organisation name the SAME way everywhere in the item.\n"
    "- Keep all facts, numbers and dates exactly. Do not add, drop or explain anything. Do not answer.\n"
    "- Return ONLY a JSON object: {\"question\": str, \"answer\": str, \"paragraphs\": "
    "[{\"title\": str, \"text\": str}, ...]} with the paragraphs in the same order and the same count.\n\n"
    "Item:\n")


def looks_arabic(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum("\u0600" <= c <= "\u06FF" for c in letters) / len(letters) > 0.5


def _norm_ar(s):
    s = re.sub("[\u064B-\u065F\u0670]", "", s)
    s = re.sub("[إأآا]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip().lower()


def leaks_answer(q_en, a_en, q_ar, a_ar):
    """True if the Arabic question contains the answer although the English question does not."""
    a_en_l = a_en.strip().lower()
    na = _norm_ar(a_ar)
    if a_en_l in ("yes", "no") or len(na) < 3:
        return False
    return a_en_l not in q_en.lower() and na in _norm_ar(q_ar)


def parse_json_object(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(t[t.find("{"):t.rfind("}") + 1])


def item_from_row(row):
    """HF 'hotpot_qa' rows store context/supporting_facts as dicts of parallel lists; the original JSON
    stores lists of [title, sentences]. Accept both."""
    ctx, sf = row["context"], row["supporting_facts"]
    if isinstance(ctx, dict):
        titles, sents = ctx["title"], ctx["sentences"]
    else:
        titles, sents = [c[0] for c in ctx], [c[1] for c in ctx]
    gold = set(sf["title"]) if isinstance(sf, dict) else {x[0] for x in sf}
    paras = [{"title": t, "text": " ".join(s).strip(), "gold": t in gold} for t, s in zip(titles, sents)]
    return {"id": row["id"], "question": row["question"], "answer": row["answer"], "type": row.get("type"),
            "level": row.get("level"), "paragraphs": [p for p in paras if p["text"]]}


def build_prompt(item, attempt=1):
    payload = {"question": item["question"], "answer": item["answer"],
               "paragraphs": [{"title": p["title"], "text": p["text"]} for p in item["paragraphs"]]}
    tail = "" if attempt == 1 else ("\n\n(Second attempt: return the JSON object exactly as specified. "
                                    "Translate the English question ONLY: the Arabic question must NOT contain, "
                                    "name or hint at the answer.)")
    return PROMPT + json.dumps(payload, ensure_ascii=False) + tail


def validate(item, out):
    if not isinstance(out, dict) or not isinstance(out.get("paragraphs"), list):
        return None
    paras = out["paragraphs"]
    if len(paras) != len(item["paragraphs"]):
        return None
    q, a = str(out.get("question", "")).strip(), str(out.get("answer", "")).strip()
    if not q or not a or not looks_arabic(q):
        return None
    if leaks_answer(item["question"], item["answer"], q, a):          # the answer was copied into the question
        return None
    clean = []
    for p in paras:
        if not isinstance(p, dict) or not str(p.get("text", "")).strip() or not looks_arabic(str(p["text"])):
            return None
        clean.append({"title": str(p.get("title", "")).strip(), "text": str(p["text"]).strip()})
    yn = {"yes": "نعم", "no": "لا"}.get(item["answer"].strip().lower())
    return {"question": q, "answer": yn or a, "paragraphs": clean}


def translate_item(item, complete_fn):
    for attempt in (1, 2):
        try:
            res = validate(item, parse_json_object(complete_fn(build_prompt(item, attempt))))
        except Exception:                                   # noqa: BLE001
            res = None
        if res:
            return res
    return None


def to_record(item, tr, model):
    return {"id": item["id"], "type": item["type"], "level": item["level"], "model": model,
            "question_en": item["question"], "answer_en": item["answer"],
            "question_ar": tr["question"], "answer_ar": tr["answer"],
            "paragraphs": [{"title_en": p["title"], "text_en": p["text"], "gold": p["gold"],
                            "title_ar": t["title"], "text_ar": t["text"]}
                           for p, t in zip(item["paragraphs"], tr["paragraphs"])]}


def hf_rows(seed):
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    for row in ds.shuffle(seed=seed, buffer_size=10_000):
        if row.get("level") != "easy":                     # medium/hard only
            yield row


def write_outputs(records, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hotpot_ctx_ar.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "hotpot_ctx_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "question_en", "question_ar", "answer_en", "answer_ar", "gold_paragraph_en",
                    "gold_paragraph_ar", "ok (y/n)", "note"])
        for r in Random(0).sample(records, min(30, len(records))):
            g = next(p for p in r["paragraphs"] if p["gold"])
            w.writerow([r["id"], r["question_en"], r["question_ar"], r["answer_en"], r["answer_ar"],
                        g["text_en"], g["text_ar"], "", ""])
    with (out / "hotpot_ctx_trial.txt").open("w", encoding="utf-8") as f:
        for r in records[:10]:
            g = next(p for p in r["paragraphs"] if p["gold"])
            f.write(f"Q  EN: {r['question_en']}\nQ  AR: {r['question_ar']}\nA  EN: {r['answer_en']}   |   "
                    f"A  AR: {r['answer_ar']}\nP  EN: {g['text_en'][:300]}\nP  AR: {g['text_ar'][:300]}\n\n")


def main(argv=None, complete_fn=None, rows=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/raw/translated")
    a = ap.parse_args(argv)
    if complete_fn is None:
        from arroute.llm import DEFAULT_MODEL, complete
        complete_fn, model = complete, DEFAULT_MODEL
    else:
        model = "fake"
    src = rows if rows is not None else hf_rows(a.seed)
    items = []
    for row in src:
        it = item_from_row(row)
        if sum(p["gold"] for p in it["paragraphs"]) >= 2:   # need both supporting paragraphs
            items.append(it)
        if len(items) >= a.n:
            break
    print(f"translating {len(items)} items ({sum(len(i['paragraphs']) for i in items)} paragraphs) "
          f"with model={model}, {a.workers} workers ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda it: translate_item(it, complete_fn), items))
    records = [to_record(it, tr, model) for it, tr in zip(items, results) if tr]
    failed = len(items) - len(records)
    write_outputs(records, a.out)
    print(f"done in {time.time() - t0:.0f}s | translated {len(records)} | failed {failed}")
    print(f"examples to read: {Path(a.out) / 'hotpot_ctx_trial.txt'}  (open in Notepad)")
    return records


if __name__ == "__main__":
    main()
