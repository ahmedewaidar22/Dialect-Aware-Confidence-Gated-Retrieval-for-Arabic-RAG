"""Phase 6: Egyptian -> MSA query rewriting (torch-free).

  rewrite_llm(q, gen)   the rewriter used by the router (an LLM prompt; cached, cheap)
  rewrite_rules(q)      a free word-substitution baseline (is an LLM needed at all?)
  check_rewrite(...)    automatic sanity checks on a rewrite
"""
import re

from .testset import egy_markers, norm, numbers

P_REWRITE = ("أعد صياغة السؤال التالي من العامية المصرية إلى اللغة العربية الفصحى، مع الحفاظ على المعنى تمامًا "
             "وعلى كل الأسماء والأرقام والتواريخ كما هي. إذا كان السؤال مكتوبًا بالفصحى بالفعل فأعده كما هو. "
             "لا تجب عن السؤال، ولا تضف أو تحذف أي معلومة، ولا تشرح. أعد السؤال المُعاد صياغته فقط.\n\n"
             "السؤال: {q}")

# Phase 6b: change as little as possible, so the keywords that search relies on survive
P_REWRITE_MIN = ("حوّل السؤال التالي من العامية المصرية إلى الفصحى بأقل تغيير ممكن: استبدل الكلمات العامية فقط "
                 "(مثل: إيه، فين، إمتى، إزاي، مين، ليه، كام، اللي، ده، دي، مش، بتاع، والأفعال مثل بيلعب واتولد) "
                 "بما يقابلها في الفصحى، واترك كل الكلمات الأخرى والأسماء والأرقام كما هي حرفيًا وبنفس الترتيب قدر "
                 "الإمكان. إذا كان السؤال بالفصحى فأعده كما هو حرفيًا. لا تجب عن السؤال ولا تشرح. "
                 "أعد السؤال فقط.\n\nالسؤال: {q}")

# Egyptian function words -> MSA (applied to whole words; a deliberately simple, transparent baseline)
RULES = {
    "ايه": "ما", "إيه": "ما", "ازاي": "كيف", "إزاي": "كيف", "ازاى": "كيف", "فين": "أين", "امتى": "متى",
    "إمتى": "متى", "امتي": "متى", "مين": "من", "ليه": "لماذا", "كام": "كم", "اللي": "الذي", "ده": "هذا",
    "دا": "هذا", "دي": "هذه", "دول": "هؤلاء", "مش": "ليس", "عشان": "لأن", "علشان": "لأن", "بتاع": "الخاص ب",
    "بتاعة": "الخاصة ب", "بتاعت": "الخاصة ب", "بس": "فقط", "دلوقتي": "الآن", "كده": "هكذا", "اوي": "جدا",
    "أوي": "جدا", "تاني": "ثاني", "التاني": "الثاني", "تانية": "ثانية", "الاتنين": "كلاهما", "اتنين": "اثنان",
    "ولا": "أم", "انهي": "أي", "إنهي": "أي", "لسه": "ما زال", "عايز": "يريد", "عاوز": "يريد", "فيه": "يوجد",
}
_BI = re.compile(r"^(و?)ب([يتن])(\w{2,})$")            # bi-imperfect: بيلعب -> يلعب, وبتعمل -> وتعمل
_AT = re.compile(r"^(و?)ات(\w{3,})$")                    # Egyptian passive: اتولد -> ولد (crude, on purpose)
_KEEP_AT = {"اتحاد", "اتجاه", "اتفاق", "اتفاقية", "اتصال", "اتساع", "اتهام", "اتباع", "اتخاذ"}


def rewrite_rules(q):
    out = []
    for tok in q.split():
        core = tok.strip("؟?،,.!")
        tail = tok[len(core):] if tok.startswith(core) else ""
        pre = ""
        if core.startswith("و") and core[1:] in RULES:
            pre, core = "و", core[1:]
        if core in RULES:
            out.append(pre + RULES[core] + tail)
            continue
        m = _BI.match(pre + core)
        if m and (m.group(2) + m.group(3)) not in RULES:
            out.append(m.group(1) + m.group(2) + m.group(3) + tail)
            continue
        m = _AT.match(pre + core)
        if m and (pre + core).lstrip("و") not in _KEEP_AT and not (pre + core).lstrip("و").startswith("اتح"):
            out.append(m.group(1) + m.group(2) + tail)
            continue
        out.append(tok)
    return " ".join(out)


def rewrite_llm_usage(q, gen):
    """-> (rewrite, usage dict, was_cached). gen: {"fn", "model", "budget", optional "prompt" (default P_REWRITE),
    optional "params" (default temperature 0, max 200 output tokens; a reasoning model needs e.g.
    {"reasoning": {"effort": "low"}, "max_output_tokens": 2000}), optional "extra" params}."""
    params = {**gen.get("params", {"temperature": 0, "max_output_tokens": 200}), **gen.get("extra", {})}
    text, usage, cached = gen["fn"](gen.get("prompt", P_REWRITE).format(q=q), model=gen["model"], budget=gen["budget"],
                                    **params)
    t = (text or "").strip().strip('"').strip()
    t = re.sub(r"^(السؤال(?: المعاد صياغته)?\s*[:：]\s*)", "", t)
    return (t.splitlines()[0].strip() if t else q), usage or {}, cached


def rewrite_llm(q, gen):
    """Same as rewrite_llm_usage, text only (identical prompt and params, so the cache is shared)."""
    return rewrite_llm_usage(q, gen)[0]


def check_rewrite(src, out, answer=None):
    """Problems with a rewrite (empty = fine)."""
    p = []
    if not out.strip():
        return ["empty"]
    if numbers(src) - numbers(out):
        p.append("number lost")
    a = norm(answer)
    if a and len(a) >= 3 and f" {a} " in f" {norm(out)} " and f" {a} " not in f" {norm(src)} ":
        p.append("answer added")
    if len(out) > 2.0 * max(len(src), 15):
        p.append("much longer")
    if egy_markers(out):
        p.append("still Egyptian words")
    return p
