"""The three answering strategies the router chooses between.

  C0  no retrieval   : the generator answers from its own knowledge (1 LLM call, 0 searches)
  C1  single-step    : search once with the question, answer from the top-k passages (1 call, 1 search)
  C2  iterative      : search, then let the generator ask for up to (max_searches-1) more searches, then answer

Each returns a dict: answer, calls, searches, passages (ids), input/output tokens, cached calls, trace.
The generator is any object with ask(prompt) -> (text, usage, was_cached).
"""
import json
import re

IDK = "لا أعرف"
MAX_PASSAGE_CHARS = 1500

P_C0 = ("Answer the question from your own knowledge.\n"
        "Reply with ONLY the short answer in Arabic (a name, number, date, or short phrase). "
        f"If you do not know, reply exactly: {IDK}\n\nQuestion: {{q}}")

P_C1 = ("Answer the question using the passages below.\n"
        "Reply with ONLY the short answer in Arabic (a name, number, date, or short phrase). "
        f"If the passages do not contain the answer, reply exactly: {IDK}\n\n"
        "Passages:\n{passages}\n\nQuestion: {q}")

P_ITER = ("You answer a question that may need SEVERAL facts, by searching an Arabic document collection step by step.\n"
          "Many questions first need an intermediate fact (which film, person, band or place is meant) before the "
          "final answer can be looked up about THAT entity.\n\n"
          "Passages found so far:\n{passages}\n\nSearches already made: {searches}\n\nQuestion: {q}\n\n"
          "If the passages contain the FINAL answer to the question (not only an intermediate fact), reply with ONLY:\n"
          '{{"action": "answer", "answer": "<short answer in Arabic>"}}\n'
          "Otherwise reply with ONLY this JSON object, naming the intermediate fact you found (if any) and searching "
          "for the ONE missing fact about it. Do not repeat a search already made:\n"
          '{{"action": "search", "found": "<intermediate fact, or empty>", "query": "<short Arabic query for the missing fact>"}}\n'
          "Searches left: {left}")

P_C2_FINAL = ("Answer the question by combining the facts in the passages below; it may need two or more of them "
              "(first an intermediate fact, then the final answer about it). Give the FINAL answer, not an intermediate fact.\n"
              "Reply with ONLY the short answer in Arabic (a name, number, date, or short phrase). "
              f"If the passages do not contain the answer, reply exactly: {IDK}\n\n"
              "Passages:\n{passages}\n\nQuestion: {q}")


def fmt_passages(texts):
    return "\n".join(f"[{i + 1}] {t[:MAX_PASSAGE_CHARS]}" for i, t in enumerate(texts))


def clean_answer(text):
    t = (text or "").strip().strip('"').strip()
    t = re.sub(r"^(الإجابة|الجواب|Answer)\s*[:：]\s*", "", t, flags=re.I)
    return t.splitlines()[0].strip() if t else ""


def parse_action(text):
    try:
        t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
        d = json.loads(t[t.find("{"):t.rfind("}") + 1])
        return d if isinstance(d, dict) else None
    except Exception:                                         # noqa: BLE001
        return None


def _new(name):
    return {"strategy": name, "answer": "", "calls": 0, "searches": 0, "passages": [], "input_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0, "cached_calls": 0, "incomplete": False, "trace": []}


def _ask(res, gen, prompt):
    text, usage, cached = gen.ask(prompt)
    res["calls"] += 1
    res["cached_calls"] += int(bool(cached))
    for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
        res[k] += int(usage.get(k, 0) or 0)
    res["incomplete"] = res["incomplete"] or bool(usage.get("incomplete"))
    return text


def run_c0(q, gen):
    res = _new("C0")
    res["answer"] = clean_answer(_ask(res, gen, P_C0.format(q=q)))
    return res


def run_c1(q, gen, retriever, k=5, method="hybrid"):
    res = _new("C1")
    pids = retriever.search(q, k=k, method=method)
    res["searches"], res["passages"] = 1, list(pids)
    texts = [retriever.text(p) for p in pids]
    res["answer"] = clean_answer(_ask(res, gen, P_C1.format(q=q, passages=fmt_passages(texts))))
    return res


def run_c2(q, gen, retriever, k=5, method="hybrid", max_searches=3, max_passages=10):
    res = _new("C2")
    pool = list(retriever.search(q, k=k, method=method))
    res["searches"] = 1
    queries = [q]
    for step in range(max_searches - 1):
        view = pool if len(pool) <= max_passages else pool[:5] + pool[-(max_passages - 5):]
        act = parse_action(_ask(res, gen, P_ITER.format(
            q=q, passages=fmt_passages([retriever.text(p) for p in view]),
            searches=" | ".join(queries), left=max_searches - 1 - step)))
        if act and act.get("action") == "answer" and str(act.get("answer", "")).strip():
            res["answer"] = clean_answer(str(act["answer"]))
            res["trace"].append({"step": step, "action": "answer"})
            break
        if act and act.get("action") == "search" and str(act.get("query", "")).strip():
            query = str(act["query"]).strip()
            queries.append(query)
            new = [p for p in retriever.search(query, k=k, method=method) if p not in pool]
            pool += new
            res["searches"] += 1
            res["trace"].append({"step": step, "action": "search", "found": str(act.get("found", "")),
                                 "query": query, "new": len(new)})
            if not new:                                   # search found nothing new: answer now, don't loop
                break
            continue
        res["trace"].append({"step": step, "action": "unparseable"})
        break
    if not res["answer"]:                                     # out of searches (or unparseable): answer now
        view = pool if len(pool) <= max_passages else pool[:5] + pool[-(max_passages - 5):]
        res["answer"] = clean_answer(_ask(res, gen, P_C2_FINAL.format(
            q=q, passages=fmt_passages([retriever.text(p) for p in view]))))
    res["passages"] = pool
    return res


STRATEGIES = {"C0": run_c0, "C1": run_c1, "C2": run_c2}
