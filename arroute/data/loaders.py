"""Loaders: raw dataset files -> (list[QAExample], dict[passage_id -> Passage]).

Each loader takes the raw root (data/raw) and finds its own subfolder.
Formats here come from the real inspect_report.txt, not guesses -- except where a
comment says UNVERIFIED (ArabicRAGB column names), which fail loudly if wrong.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .schema import Passage, QAExample

VARIETIES = ("egy", "glf", "lev", "mgr", "msa")
# questions that point at "the passage/text/attachment" cannot work as standalone retrieval queries
PASSAGE_REF = re.compile(r"(الفقرة|الفقره|النص|المقطع|الوثيقة|المرفق)")


def strip_wrap(text):
    """Remove wrapper braces seen on some MAFQA sub-questions ({...}); they would be a class tell."""
    return text.strip("{} ").strip()


def clean(text):
    """Whitespace cleanup only. Letter normalization (alef/yaa/...) is done at retrieval time."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def pid(text):
    """Stable passage id = hash of the cleaned text (same text -> same id, any source)."""
    return "p_" + hashlib.sha1(clean(text).encode("utf-8")).hexdigest()[:16]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# ArabicaQA (SQuAD-format MRC): single-hop, MSA, Wikipedia  -> complexity C1
# ----------------------------------------------------------------------------
def load_arabicaqa(raw, include_impossible=False):
    base = Path(raw) / "ArabicaQA" / "MRC"
    examples, passages, missing_ctx = [], {}, 0
    seen_split = {}                                       # raw id -> split first seen in
    cross_split_ids = 0
    empty_q = 0
    for fname, split in (("train.json", "train"), ("val.json", "dev"), ("test.json", "test")):
        for art in _read_json(base / fname)["data"]:
            title = art.get("title", "")
            for para in art["paragraphs"]:
                ctx = clean(para.get("context"))
                if not ctx:
                    missing_ctx += len(para.get("qas", []))
                    continue
                p = pid(ctx)
                passages.setdefault(p, Passage(p, ctx, "arabicaqa", title))
                for qa in para["qas"]:
                    if qa.get("is_impossible") and not include_impossible:
                        continue                      # unanswerable != C0; exclude
                    question = clean(qa.get("question"))
                    if not question:                      # real data has a few blank questions
                        empty_q += 1
                        continue
                    answers = qa.get("answers") or []
                    if seen_split.setdefault(qa["id"], split) != split:
                        cross_split_ids += 1              # same raw id in two splits = leakage signal
                    examples.append(QAExample(
                        id=f"arabicaqa:{split}:{qa['id']}", question=question,
                        variety="msa", complexity="C1",
                        answer=clean(answers[0].get("text")) if answers else None,
                        gold_passage_ids=[p], source="arabicaqa", domain="wikipedia",
                        split=split))
    if missing_ctx:
        print(f"[arabicaqa] WARNING: {missing_ctx} questions had no 'context' and were skipped")
    if empty_q:
        print(f"[arabicaqa] skipped {empty_q} questions that were empty after cleaning")
    if cross_split_ids:
        print(f"[arabicaqa] WARNING: {cross_split_ids} raw question ids appear in more than one split")
    if not examples:
        raise ValueError("[arabicaqa] 0 examples loaded: 'context' key may be missing in MRC files")
    return examples, passages


# ----------------------------------------------------------------------------
# ArDQA: parallel QA over 5 varieties. File = {D}-{split}-context-{cv}-questions-{qv}.json
# The same qas[].id appears in every file => it is the alignment key.
# We read only the MSA-context files: one shared MSA gold passage per item, and
# questions in all 5 varieties => exactly the pilot setup (dialect question over MSA corpus).
# ----------------------------------------------------------------------------
_ARDQA_FILE = re.compile(r"context-(egy|glf|lev|mgr|msa)-questions-(egy|glf|lev|mgr|msa)\.json$")
_ARDQA_DOMAINS = {"ArDQA-SQuAD": "squad", "ArDQA-Vlogs": "vlogs", "ArDQA-Narratives": "narratives"}


def _ardqa_rows(path):
    """Flatten one ArDQA file. qas[].id is NOT unique and the 25 files are NOT perfectly parallel
    (e.g. 160 vs 154 questions), so items are matched on (title, MSA-context, id, occurrence)."""
    rows, occ = [], Counter()
    for art in _read_json(path)["data"]:
        title = art.get("title", "")
        for para in art["paragraphs"]:
            ctx = clean(para["context"])
            cp = pid(ctx)
            for qa in para["qas"]:
                base = (title, cp, str(qa["id"]))
                k = occ[base]
                occ[base] += 1                          # k distinguishes repeated ids
                ans = qa.get("answers") or []
                rows.append({"key": base + (k,), "id": str(qa["id"]), "title": title, "ctx": ctx,
                             "question": clean(qa["question"]),
                             "answer": clean(ans[0].get("text")) if ans else None})
    return rows


def load_ardqa(raw):
    base = Path(raw) / "ArDQA"
    examples, passages, empty_q = [], {}, 0
    for dom_dir, domain in _ARDQA_DOMAINS.items():
        d = base / dom_dir
        if not d.exists():
            print(f"[ardqa] WARNING: {dom_dir} not found, skipping")
            continue
        by_split = {}                                   # split -> {qvar: rows}
        for f in sorted(d.rglob("*.json")):
            m = _ARDQA_FILE.search(f.name)
            if m and m.group(1) == "msa":               # MSA context = one shared gold passage
                by_split.setdefault(f.parent.name, {})[m.group(2)] = _ardqa_rows(f)
        for split, by_var in by_split.items():
            maps = {v: {r["key"]: r for r in rows} for v, rows in by_var.items()}
            all_keys = sorted(set().union(*[set(m) for m in maps.values()]), key=str)
            coverage = sum(len(m) for m in maps.values()) / (len(all_keys) * len(maps))
            sizes = {v: len(r) for v, r in sorted(by_var.items())}
            missing = {v: len(all_keys) - len(m) for v, m in sorted(maps.items())}
            print(f"[ardqa] {domain}/{split}: {len(all_keys)} items | file sizes {sizes} | "
                  f"missing per variety {missing} | coverage {coverage:.1%}")
            if coverage < 0.8:                          # systematic misalignment, not a few gaps
                worst = max(missing, key=missing.get)
                ex = [k for k in all_keys if k not in maps[worst]][:3]
                raise ValueError(f"[ardqa] {domain}/{split}: coverage {coverage:.2f} < 0.8, the "
                                 f"files do not share the same items. Unmatched in {worst}: {ex}")
            ref = by_var.get("msa") or next(iter(by_var.values()))
            rep = sum(c - 1 for c in Counter(r["id"] for r in ref).values() if c > 1)
            dup = len(ref) - len({(r["ctx"], r["question"]) for r in ref})
            if rep or dup:
                print(f"[ardqa]   {rep} repeated ids, {dup} exact duplicate (context, question) "
                      f"items in the MSA file; dedupe later if needed")
            for key in all_keys:
                group = f"ardqa:{domain}:{split}:" + hashlib.sha1(repr(key).encode()).hexdigest()[:12]
                for v, mp in maps.items():
                    r = mp.get(key)
                    if r is None:
                        continue
                    if not r["question"]:
                        empty_q += 1
                        continue
                    gp = pid(r["ctx"])
                    passages.setdefault(gp, Passage(gp, r["ctx"], "ardqa", r["title"]))
                    examples.append(QAExample(
                        id=f"{group}:{v}", question=r["question"], variety=v, complexity=None,
                        answer=r["answer"], gold_passage_ids=[gp], source="ardqa",
                        domain=domain, parallel_group=group, split=split,
                        extra={"ardqa_id": r["id"]}))
    if empty_q:
        print(f"[ardqa] skipped {empty_q} empty questions")
    if not examples:
        raise ValueError("[ardqa] 0 examples: expected files named *-context-msa-questions-*.json")
    return examples, passages


# ----------------------------------------------------------------------------
# ArabicRAGB (HF dataset on disk). UNVERIFIED column names -> auto-detected, loud failure.
# Its 'complexity' is reasoning-within-one-passage, NOT retrieval complexity, so we do
# not map it to C0/C1/C2; the raw value is kept in extra['ragb_complexity'].
# ----------------------------------------------------------------------------
_DIALECT_MAP = [("msa", "msa"), ("modern", "msa"), ("standard", "msa"), ("egy", "egy"),
                ("cairo", "egy"), ("gulf", "glf"), ("glf", "glf"), ("khalij", "glf"),
                ("lev", "lev"), ("shami", "lev"), ("magh", "mgr"), ("mgr", "mgr"),
                ("moroc", "mgr"), ("alger", "mgr"), ("tunis", "mgr")]


def map_dialect(value):
    s = str(value).strip().lower()
    for needle, label in _DIALECT_MAP:
        if needle in s:
            return label
    return None


def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_arabicragb(raw):
    from datasets import load_from_disk
    ds = load_from_disk(str(Path(raw) / "ArabicRAGB"))
    examples, passages, unknown = [], {}, Counter()
    for split in ds.keys():
        cols = ds[split].column_names
        q_col = _pick(cols, ["query", "question"])
        d_col = _pick(cols, ["query_dialect", "dialect"])
        c_col = _pick(cols, ["query_complexity", "complexity"])
        p_col = _pick(cols, ["passage_text", "passage", "context", "document", "text"])
        t_col = _pick(cols, ["passage_title"])
        if not (q_col and d_col):
            raise ValueError(f"[arabicragb] cannot find query/dialect columns in {cols}")
        for i, row in enumerate(ds[split]):
            variety = map_dialect(row[d_col])
            if variety is None:
                unknown[row[d_col]] += 1
                continue
            gold = []
            if p_col and clean(row[p_col]):
                p = pid(row[p_col])
                passages.setdefault(p, Passage(p, clean(row[p_col]), "arabicragb",
                                               clean(row[t_col]) if t_col else ""))
                gold = [p]
            examples.append(QAExample(
                id=f"arabicragb:{split}:{i}", question=clean(row[q_col]), variety=variety,
                complexity=None, gold_passage_ids=gold, source="arabicragb",
                domain="mixed", split=split,
                extra={"ragb_complexity": row[c_col] if c_col else None,
                       "ragb_dialect_raw": str(row[d_col]),
                       "ragb_id": str(row.get("id")),
                       "ragb_category": str(row.get("source_category"))}))
    if unknown:
        raise ValueError(f"[arabicragb] unmapped dialect values: {dict(unknown)} -> extend _DIALECT_MAP")
    if not p_col:
        print(f"[arabicragb] NOTE: no passage column among {cols}; gold passages left empty")
    return examples, passages


# ----------------------------------------------------------------------------
# MAFQA: list of {original_index, context, question, decomposition, a_gen}. Fatwa, C2.
# Only ~388 items -> NOT enough alone (domain shortcut risk). 'decomposition' kept raw.
# ----------------------------------------------------------------------------
def load_mafqa(raw):
    base = Path(raw) / "MAFQA"
    files = sorted(base.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no .json in {base}")
    items = _read_json(files[0])
    examples, passages, occ = [], {}, Counter()
    for it in items:
        oi = it["original_index"]
        k = occ[oi]
        occ[oi] += 1                                    # several items share one original_index
        ctx = clean(it.get("context"))
        p = pid(ctx) if ctx else None
        if p:
            passages.setdefault(p, Passage(p, ctx, "mafqa"))
        ans = it.get("a_gen")
        examples.append(QAExample(
            id=f"mafqa:{oi}:{k}", question=strip_wrap(clean(it["question"])),
            variety="msa", complexity="C2", answer=clean(ans) if isinstance(ans, str) else None,
            gold_passage_ids=[p] if p else [], source="mafqa", domain="fatwa",
            parallel_group=f"mafqa:{oi}",               # same source fatwa -> keep in ONE split
            extra={"decomposition": it.get("decomposition"), "original_index": oi}))
    uq = len({e.question for e in examples})
    print(f"[mafqa] {len(examples)} items | {len(occ)} distinct original_index | "
          f"{uq} distinct questions")
    return examples, passages


def load_mafqa_sub(raw):
    """MAFQA sub-questions sq1/sq2 are SINGLE-hop fatwa questions (sq1 from passage p_a, sq2 from p_b):
    C1 examples in the same domain as the C2 originals, which weakens the 'fatwa => C2' shortcut.
    NOTE: they are shorter/cleaner than the original scenario questions, so a length shortcut
    may replace the domain shortcut -- check with a length-only baseline in Phase 2."""
    files = sorted((Path(raw) / "MAFQA").glob("*.json"))
    if not files:
        raise FileNotFoundError("no MAFQA .json")
    examples, passages, occ = [], {}, Counter()
    for it in _read_json(files[0]):
        oi = it["original_index"]
        k = occ[oi]
        occ[oi] += 1
        dec = it.get("decomposition")
        if not isinstance(dec, dict):
            continue
        for i in (1, 2):                                # sq3 is built from BOTH passages: excluded
            sq = strip_wrap(clean(dec.get(f"sq{i}")))
            if not sq:
                continue
            gold_text = {1: dec.get("p_a"), 2: dec.get("p_b")}.get(i)
            gold = []
            if clean(gold_text):
                p = pid(gold_text)
                passages.setdefault(p, Passage(p, clean(gold_text), "mafqa_sub"))
                gold = [p]
            examples.append(QAExample(
                id=f"mafqa_sub:{oi}:{k}:sq{i}", question=sq, variety="msa", complexity="C1",
                answer=clean(dec.get(f"sa{i}")) or None, gold_passage_ids=gold,
                source="mafqa_sub", domain="fatwa", parallel_group=f"mafqa:{oi}",
                extra={"parent": f"mafqa:{oi}:{k}", "sq_index": i}))
    return examples, passages


# ----------------------------------------------------------------------------
# Pilot set: 200 validated pairs -> 400 examples (msa + egy). Held-out regression set.
# ----------------------------------------------------------------------------
def load_pilot(raw):
    f = next((Path(raw) / "pilot").glob("*.json"))
    examples, passages = [], {}
    for it in _read_json(f):
        p = pid(it["context"])
        passages.setdefault(p, Passage(p, clean(it["context"]), "pilot"))
        for field_, variety in (("question_msa", "msa"), ("question_egy", "egy")):
            examples.append(QAExample(
                id=f"pilot:{it['id']}:{variety}", question=clean(it[field_]), variety=variety,
                complexity="C1", gold_passage_ids=[p], source="pilot", domain="general",
                parallel_group=f"pilot:{it['id']}", split="heldout"))
    return examples, passages


def _load_translated(raw, kind, label, source):
    f = Path(raw) / "translated" / f"{kind}_ar.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} not present (run scripts/translate_questions.py --kind {kind})")
    examples = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = clean(r["question_ar"])
        if q:
            examples.append(QAExample(
                id=f"{source}:{r['id']}", question=q, variety="msa", complexity=label, source=source,
                domain="wikipedia", extra={"question_en": r["question_en"], "type": r.get("type"),
                                           "level": r.get("level"), "model": r.get("model")}))
    return examples, {}


def load_hotpot_ar(raw):          # translated HotpotQA (medium/hard) -> multi-hop, C2
    return _load_translated(raw, "hotpot", "C2", "hotpot_ar")


def load_nq_ar(raw):              # translated NQ-Open -> single-hop, C1 (same pipeline => same style)
    return _load_translated(raw, "nq", "C1", "nq_ar")


def load_hotpot_ctx_ar(raw):
    """Translated HotpotQA items WITH their 10 paragraphs: questions to be OUTCOME-labelled, plus corpus passages.
    Passage id = hash of the ENGLISH paragraph, so a paragraph shared by several items is one corpus passage."""
    f = Path(raw) / "translated" / "hotpot_ctx_ar.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} not present (run scripts/translate_hotpot_context.py)")
    examples, passages = [], {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gold = []
        for p in r["paragraphs"]:
            pid_ = "h_" + hashlib.sha1((p["title_en"] + "\n" + p["text_en"]).encode("utf-8")).hexdigest()[:16]
            text = clean(f"{p['title_ar']}. {p['text_ar']}" if p["title_ar"] else p["text_ar"])
            passages.setdefault(pid_, Passage(pid_, text, "hotpot_ctx_ar", clean(p["title_ar"])))
            if p["gold"]:
                gold.append(pid_)
        q = clean(r["question_ar"])
        if q:
            examples.append(QAExample(
                id=f"hotpot_ctx:{r['id']}", question=q, variety="msa", complexity=None, answer=clean(r["answer_ar"]),
                gold_passage_ids=gold, source="hotpot_ctx_ar", domain="wikipedia",
                extra={"question_en": r["question_en"], "answer_en": r["answer_en"], "type": r.get("type"),
                       "level": r.get("level"), "model": r.get("model")}))
    return examples, passages


LOADERS = {
    "arabicaqa": load_arabicaqa, "ardqa": load_ardqa, "arabicragb": load_arabicragb,
    "mafqa": load_mafqa, "mafqa_sub": load_mafqa_sub, "pilot": load_pilot,
    "hotpot_ar": load_hotpot_ar, "nq_ar": load_nq_ar, "hotpot_ctx_ar": load_hotpot_ctx_ar,
}
