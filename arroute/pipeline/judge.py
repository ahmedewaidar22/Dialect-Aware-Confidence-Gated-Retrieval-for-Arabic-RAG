"""Is an answer correct? Cheap rules first, then an LLM judge that sees the reference answer(s).

Validate the judge against human grades (scripts/judge_sample.py) before trusting the labels.
"""
import re

from arroute.retrieval.text import normalize_ar

from .strategies import IDK, parse_action

P_JUDGE = ("You are grading an answer to a question by comparing it with the reference answer.\n\n"
           "Question: {q}\nReference answer: {ref}\nModel answer: {ans}\n\n"
           "The model answer is CORRECT if it states the same key fact as the reference. Names may be spelled, "
           "transliterated or written in another script differently; numbers and dates must match. It is "
           "INCORRECT if it is wrong, says it does not know, or is too vague to contain the key fact.\n"
           'Reply with ONLY a JSON object: {{"correct": true}} or {{"correct": false}}')

_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_YES, _NO = {"نعم", "yes"}, {"لا", "no"}


def norm(s):
    s = normalize_ar(str(s or "")).lower().translate(_DIG)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def rule_verdict(answer, refs):
    """True / False when a simple rule is decisive, None when the LLM judge must decide."""
    a = norm(answer)
    if not a or norm(IDK) in a:
        return False
    for r in refs:
        rn = norm(r)
        if rn in _YES | _NO:                                   # yes/no question: compare the first word
            first = a.split()[0]
            if first in _YES | _NO:
                return (first in _YES) == (rn in _YES)
            return None                                         # e.g. "both are cities": let the LLM decide
    for r in refs:
        rn = norm(r)
        if 2 <= len(rn) <= 60 and re.search(rf"(^|\s){re.escape(rn)}(\s|$)", a):
            return True                                         # reference appears as whole words in the answer
    return None


def judge(q, answer, refs, judge_gen):
    """-> (correct: bool, method: 'rule' | 'llm' | 'llm_unparsed', usage dict)"""
    refs = [r for r in refs if r and str(r).strip()]
    v = rule_verdict(answer, refs)
    if v is not None:
        return v, "rule", {}
    text, usage, _ = judge_gen.ask(P_JUDGE.format(q=q, ref=" / ".join(refs), ans=answer))
    d = parse_action(text)
    if d is None or "correct" not in d:
        return False, "llm_unparsed", usage
    return bool(d["correct"]), "llm", usage
