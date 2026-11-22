"""Phase 3b tests: budget guard, cached client, judge rules, strategies, and a full labelling run (all fakes)."""
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arroute import llm                                                  # noqa: E402
from arroute.data import QAExample, write_jsonl                          # noqa: E402
from arroute.pipeline.judge import judge, rule_verdict                   # noqa: E402
from arroute.pipeline.strategies import IDK, run_c0, run_c1, run_c2      # noqa: E402
from arroute.retrieval.index import BM25Index, Retriever                 # noqa: E402
import compare_labels                                                    # noqa: E402
import judge_sample                                                      # noqa: E402
import label_outcomes                                                    # noqa: E402

PASSAGES = {"p1": "تم بناء السد العالي في أسوان بين عامي 1960 و1970",
            "p2": "القاهرة عاصمة مصر وأكبر مدنها",
            "p3": "يبلغ طول نهر النيل حوالي 6650 كيلومترا",
            "p4": "المشروع الضخم على النيل في الجنوب هو أحد أشهر المشروعات"}


def retriever():
    ids = list(PASSAGES)
    return Retriever(BM25Index(ids, [PASSAGES[i] for i in ids]), passages=dict(PASSAGES))


# ---------------------------------------------------------------- budget + cached client
def test_budget_reserves_worst_case_and_refuses():
    b = llm.Budget(0.01, {"m": (1.0, 10.0)})
    r = b.reserve("m", "x" * 200, 100)                     # 100 tok in * $1/M + 100 out * $10/M = 0.0011
    assert abs(r - 0.0011) < 1e-9
    b.settle("m", r, {"input_tokens": 50, "output_tokens": 20})
    assert abs(b.spent - (50 * 1 + 20 * 10) / 1e6) < 1e-12
    with pytest.raises(llm.BudgetExceeded):
        b.reserve("m", "x" * 200, 2000)                       # worst case 0.0201 > remaining
    with pytest.raises(ValueError):
        b.price("unknown-model")


class FakeClient:
    def __init__(self, status="completed"):
        self.calls, self.status = 0, status
        self.responses = SimpleNamespace(create=self.create)

    def create(self, model, input, **params):
        self.calls += 1
        return SimpleNamespace(output_text="القاهرة", status=self.status, usage=SimpleNamespace(
            input_tokens=120, output_tokens=30, input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=12)))


def test_cached_client_ledger_and_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    fake = FakeClient()
    monkeypatch.setattr(llm, "_client", fake)
    b = llm.Budget(1.0, {"m": (2.0, 10.0)})
    t, u, cached = llm.complete_with_usage("سؤال", model="m", budget=b, max_output_tokens=100)
    assert t == "القاهرة" and not cached and u["reasoning_tokens"] == 12
    assert abs(b.spent - (120 * 2 + 30 * 10) / 1e6) < 1e-12
    t2, _, cached2 = llm.complete_with_usage("سؤال", model="m", budget=b, max_output_tokens=100)
    assert cached2 and fake.calls == 1 and abs(b.spent - 0.00054) < 1e-12        # cache hit: no new spend
    led = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(led) == 1 and json.loads(led[0])["cost_usd"] > 0
    assert llm.complete("سؤال", model="m", max_output_tokens=100) == "القاهرة"


def test_incomplete_answers_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    fake = FakeClient(status="incomplete")
    monkeypatch.setattr(llm, "_client", fake)
    llm.complete_with_usage("q", model="m")
    llm.complete_with_usage("q", model="m")
    assert fake.calls == 2


# ---------------------------------------------------------------- judge
def test_judge_rules():
    assert rule_verdict("القاهرة", ["القاهرة"]) is True
    assert rule_verdict("عاصمتها القاهرة.", ["القاهره"]) is True                   # normalisation
    assert rule_verdict("18", ["8"]) is None                                        # no partial-number match
    assert rule_verdict(IDK, ["القاهرة"]) is False and rule_verdict("", ["x"]) is False
    assert rule_verdict("نعم، كلاهما", ["نعم"]) is True and rule_verdict("لا", ["نعم"]) is False
    assert rule_verdict("كلاهما مدينتان", ["نعم"]) is None                           # LLM decides
    assert rule_verdict("٦٦٥٠ كم", ["6650"]) is True                                # Arabic-Indic digits


def test_judge_falls_back_to_llm():
    class J:
        def __init__(self, reply):
            self.reply = reply

        def ask(self, p):
            return self.reply, {"input_tokens": 9}, False
    assert judge("q", "تينا تيرنر", ["Tina Turner"], J('{"correct": true}'))[:2] == (True, "llm")
    assert judge("q", "شيء", ["x"], J("garbage"))[:2] == (False, "llm_unparsed")


# ---------------------------------------------------------------- strategies
class ScriptedGen:
    """Behaves like a tiny model: knows the capital, reads passage [1], searches when 'أسوان' is missing."""
    def __init__(self):
        self.prompts = []

    def ask(self, p):
        self.prompts.append(p)
        u = {"input_tokens": len(p) // 3, "output_tokens": 10}
        if "grading an answer" in p:
            return '{"correct": false}', u, False
        if "from your own knowledge" in p:
            return ("القاهرة" if "عاصمة مصر" in p else IDK), u, False
        if "Searches left" in p:
            if "أسوان" in p:
                return '{"action": "answer", "answer": "أسوان"}', u, False
            return '{"action": "search", "query": "بناء السد العالي"}', u, False
        first = p.split("[1] ", 1)[1].split("\n")[0] if "[1] " in p else IDK
        return first, u, False


def test_strategies():
    r, g = retriever(), ScriptedGen()
    assert run_c0("ما هي عاصمة مصر؟", g)["answer"] == "القاهرة"
    c1 = run_c1("كم يبلغ طول نهر النيل؟", g, r, k=2, method="bm25")
    assert c1["searches"] == 1 and "6650" in c1["answer"] and c1["passages"][0] == "p3"
    c2 = run_c2("أين يقع المشروع الضخم على النيل؟", g, r, k=1, method="bm25")
    assert c2["answer"] == "أسوان" and c2["searches"] == 2 and "p1" in c2["passages"]
    assert c2["trace"][0]["action"] == "search" and c2["calls"] == 2


# ---------------------------------------------------------------- full labelling run
def _processed(tmp):
    P = Path(tmp) / "processed"
    write_jsonl([QAExample("a1", "ما هي عاصمة مصر؟", "msa", "C1", answer="القاهرة", gold_passage_ids=["p2"],
                           source="arabicaqa", split="test"),
                 QAExample("a2", "كم يبلغ طول نهر النيل؟", "msa", "C1", answer="6650 كيلومترا", gold_passage_ids=["p3"],
                           source="arabicaqa", split="test")], P / "arabicaqa.jsonl")
    write_jsonl([QAExample("h1", "أين يقع المشروع الضخم على النيل؟", "msa", None, answer="أسوان",
                           gold_passage_ids=["p4", "p1"], source="hotpot_ctx_ar", extra={"answer_en": "Aswan"})],
                P / "hotpot_ctx_ar.jsonl")
    return P


def fake_complete(counter):
    gen = ScriptedGen()

    def f(prompt, model=None, budget=None, **params):
        counter.append(model)
        reserved = budget.reserve(model, prompt, params.get("max_output_tokens")) if budget else 0
        text, usage, cached = gen.ask(prompt)
        if budget:
            budget.settle(model, reserved, usage)
        return text, usage, cached
    return f


ARGS = ["--gen_model", "gm", "--judge_model", "jm", "--price", "gm=1,1", "--price", "jm=1,1",
        "--method", "bm25", "--k", "1", "--workers", "2"]


def test_label_run_resume_and_budget_stop(tmp_path):
    P, calls = _processed(tmp_path), []
    recs = label_outcomes.main(["--run", "r1", "--budget_usd", "1", "--processed", str(P),
                                "--out", str(tmp_path / "labels")] + ARGS,
                               complete_fn=fake_complete(calls), retriever=retriever())
    labels = {r["id"]: r["label"] for r in recs}
    assert labels == {"a1": "C0", "a2": "C1", "h1": "C2"}
    h1 = next(r for r in recs if r["id"] == "h1")["results"]
    assert set(h1) == {"C0", "C1", "C2"} and h1["C2"]["gold_found"] == 1.0          # found both gold passages
    summary = (tmp_path / "labels" / "r1" / "summary.txt").read_text(encoding="utf-8")
    assert "questions labelled: 3" in summary and "STOPPED" not in summary
    n_before = len(calls)
    label_outcomes.main(["--run", "r1", "--budget_usd", "1", "--processed", str(P),
                         "--out", str(tmp_path / "labels")] + ARGS, complete_fn=fake_complete(calls), retriever=retriever())
    assert len(calls) == n_before                                                  # resumed: nothing re-run
    stopped = label_outcomes.main(["--run", "r2", "--budget_usd", "0.000001", "--processed", str(P),
                                   "--out", str(tmp_path / "labels")] + ARGS,
                                  complete_fn=fake_complete([]), retriever=retriever())
    assert stopped == [] and "STOPPED" in (tmp_path / "labels" / "r2" / "summary.txt").read_text(encoding="utf-8")


def test_compare_and_judge_validation(tmp_path, capsys):
    P = _processed(tmp_path)
    label_outcomes.main(["--run", "a", "--budget_usd", "1", "--processed", str(P), "--out", str(tmp_path / "L"),
                         "--mode", "full"] + ARGS, complete_fn=fake_complete([]), retriever=retriever())
    shutil.copytree(tmp_path / "L" / "a", tmp_path / "L" / "b")
    assert compare_labels.main("a", "b", root=str(tmp_path / "L")) == 1.0
    judge_sample.main(["export", "--run", "a", "--n", "6", "--root", str(tmp_path / "L")])
    f = tmp_path / "L" / "a" / "judge_sample.csv"
    rows = list(csv.reader(f.open(encoding="utf-8-sig")))
    key = json.loads((tmp_path / "L" / "a" / "judge_sample_key.json").read_text(encoding="utf-8"))
    for row in rows[1:]:                                       # a grader who agrees with the judge everywhere
        row[4] = "y" if key[row[0]]["judge"] else "n"
    with f.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows)
    acc, _ = judge_sample.main(["agree", "--run", "a", "--root", str(tmp_path / "L")])
    assert acc == 1.0


def test_c2_stops_when_a_search_adds_nothing():
    class Repeater:
        def __init__(self):
            self.n = 0

        def ask(self, p):
            self.n += 1
            if "Searches left" in p:
                return '{"action": "search", "query": "النيل"}', {}, False
            return "القاهرة", {}, False
    r, g = retriever(), Repeater()
    res = run_c2("سؤال عن النيل", g, r, k=4, method="bm25", max_searches=3)      # k=4: first search gets all 4
    assert res["searches"] == 2 and res["calls"] == 2 and res["trace"][0]["new"] == 0


def test_degenerate_references_filtered_and_types_recorded(tmp_path):
    assert not label_outcomes.usable_ref(".") and not label_outcomes.usable_ref("") and label_outcomes.usable_ref("15")
    P = _processed(tmp_path)
    write_jsonl([QAExample("a1", "ما هي عاصمة مصر؟", "msa", "C1", answer="القاهرة", gold_passage_ids=["p2"],
                           source="arabicaqa", split="test"),
                 QAExample("bad", "ما دور التلال؟", "msa", "C1", answer=".", gold_passage_ids=["p1"],
                           source="arabicaqa", split="test")], P / "arabicaqa.jsonl")
    write_jsonl([QAExample("h2", "هل كلاهما مجلات؟", "msa", None, answer="نعم", gold_passage_ids=["p1"],
                           source="hotpot_ctx_ar", extra={"answer_en": "yes", "type": "comparison"})],
                P / "hotpot_ctx_ar.jsonl")
    recs = label_outcomes.main(["--run", "t", "--budget_usd", "1", "--processed", str(P), "--out", str(tmp_path / "L")]
                               + ARGS, complete_fn=fake_complete([]), retriever=retriever())
    ids = {r["id"] for r in recs}
    assert "bad" not in ids and {"a1", "h2"} <= ids
    h2 = next(r for r in recs if r["id"] == "h2")
    assert h2["qtype"] == "comparison" and h2["yesno"] is True
    assert "by question type" in (tmp_path / "L" / "t" / "summary.txt").read_text(encoding="utf-8")


class FakeLocal:
    """OpenAI-compatible local server (Chat Completions), e.g. Ollama."""
    def __init__(self, content="<think>أفكر...</think>القاهرة", finish="stop"):
        self.kwargs, self.content, self.finish = [], content, finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kw):
        self.kwargs.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content),
                                                        finish_reason=self.finish)],
                               usage=SimpleNamespace(prompt_tokens=50, completion_tokens=7))


def test_local_backend_maps_params_strips_think_and_is_free(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    url = "http://localhost:11434/v1"
    fake = FakeLocal()
    monkeypatch.setitem(llm._local_clients, url, fake)
    b = llm.Budget(0.0, {"qwen3.5:9b": (0.0, 0.0)})             # zero budget is fine: local costs nothing
    out, usage, cached = llm.complete_with_usage("سؤال", model="qwen3.5:9b", budget=b, base_url=url,
                                                 temperature=0, max_output_tokens=256)
    assert out == "القاهرة" and usage["input_tokens"] == 50 and not cached
    assert fake.kwargs[0] == {"max_tokens": 256, "temperature": 0}                 # mapped for Chat Completions
    assert llm.complete_with_usage("سؤال", model="qwen3.5:9b", budget=b, base_url=url,
                                   temperature=0, max_output_tokens=256)[2] is True   # cached
    cut = FakeLocal(content="<think>still thinking", finish="length")
    monkeypatch.setitem(llm._local_clients, url, cut)
    out2, u2, _ = llm.complete_with_usage("آخر", model="qwen3.5:9b", base_url=url, max_output_tokens=8)
    assert out2 == "" and u2["incomplete"] is True                                 # truncated mid-think: not cached


def test_label_run_with_local_generator_routes_only_the_generator(tmp_path):
    P, seen = _processed(tmp_path), []
    gen = ScriptedGen()

    def f(prompt, model=None, budget=None, **params):
        seen.append((model, params.get("base_url")))
        reserved = budget.reserve(model, prompt, params.get("max_output_tokens")) if budget else 0
        text, usage, cached = gen.ask(prompt)
        budget.settle(model, reserved, usage)
        return text, usage, cached
    args = ["--run", "loc", "--budget_usd", "1", "--processed", str(P), "--out", str(tmp_path / "L"),
            "--gen_model", "qwen3.5:9b", "--gen_base_url", "http://localhost:11434/v1", "--judge_model", "jm",
            "--price", "jm=1,1", "--method", "bm25", "--k", "1", "--workers", "1"]
    recs = label_outcomes.main(args, complete_fn=f, retriever=retriever())
    assert {r["label"] for r in recs} == {"C0", "C1", "C2"}
    assert {b for m, b in seen if m == "qwen3.5:9b"} == {"http://localhost:11434/v1"}
    assert {b for m, b in seen if m == "jm"} == {None}                              # judge stays on the API
    assert "local http://localhost:11434/v1" in (tmp_path / "L" / "loc" / "summary.txt").read_text(encoding="utf-8")


def test_c2_prompt_lists_previous_searches_and_final_prompt_combines():
    class Recorder:
        def __init__(self):
            self.prompts = []

        def ask(self, p):
            self.prompts.append(p)
            if "Searches left" in p:
                return '{"action": "search", "found": "السد العالي", "query": "موقع السد العالي"}', {}, False
            return "أسوان", {}, False
    g = Recorder()
    res = run_c2("أين يقع المشروع الضخم على النيل؟", g, retriever(), k=1, method="bm25", max_searches=2)
    assert "Searches already made: أين يقع المشروع الضخم على النيل؟" in g.prompts[0]
    assert "FINAL answer, not an intermediate fact" in g.prompts[-1]           # the C2-specific final prompt
    assert res["trace"][0]["found"] == "السد العالي" and res["answer"] == "أسوان"


def test_run_settings_cannot_be_mixed(tmp_path):
    P = _processed(tmp_path)
    base = ["--run", "same", "--budget_usd", "1", "--processed", str(P), "--out", str(tmp_path / "L")] + ARGS
    label_outcomes.main(base, complete_fn=fake_complete([]), retriever=retriever())
    with pytest.raises(SystemExit, match="different settings"):
        label_outcomes.main(base + ["--mode", "full"], complete_fn=fake_complete([]), retriever=retriever())


def test_compare_reports_per_strategy_accuracy(tmp_path, capsys):
    P = _processed(tmp_path)
    for run, mode in (("cas", "cascade"), ("ful", "full")):
        label_outcomes.main(["--run", run, "--mode", mode, "--budget_usd", "1", "--processed", str(P),
                             "--out", str(tmp_path / "L")] + ARGS, complete_fn=fake_complete([]), retriever=retriever())
    compare_labels.main("cas", "ful", root=str(tmp_path / "L"))
    out = capsys.readouterr().out
    assert "strategy accuracy on questions where BOTH runs ran that strategy" in out and "C2: n=" in out


class RateLimitError(Exception):          # same class NAME as openai.RateLimitError
    pass


def test_rate_limits_are_waited_out(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    fake = FakeClient()
    calls = {"n": 0}
    real = fake.create

    def flaky(model, input, **params):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RateLimitError("429 tokens per min")
        return real(model, input, **params)
    fake.responses = SimpleNamespace(create=flaky)
    monkeypatch.setattr(llm, "_client", fake)
    b = llm.Budget(1.0, {"m": (2.0, 10.0)})
    out, _, _ = llm.complete_with_usage("q", model="m", budget=b, max_output_tokens=50)
    assert out == "القاهرة" and calls["n"] == 3
    assert abs(b.spent - (120 * 2 + 30 * 10) / 1e6) < 1e-12           # charged once, reservation released


def test_other_errors_fail_fast_and_release_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    fake = FakeClient()

    def boom(model, input, **params):
        raise ValueError("bad request")
    fake.responses = SimpleNamespace(create=boom)
    monkeypatch.setattr(llm, "_client", fake)
    b = llm.Budget(1.0, {"m": (2.0, 10.0)})
    with pytest.raises(ValueError):
        llm.complete_with_usage("q", model="m", budget=b, max_output_tokens=50)
    assert abs(b.spent) < 1e-12


def test_one_failing_question_does_not_stop_the_run(tmp_path):
    P, gen = _processed(tmp_path), ScriptedGen()
    state = {"fail": True}

    def f(prompt, model=None, budget=None, **params):
        if state["fail"] and "طول نهر النيل" in prompt:
            raise RuntimeError("timeout")
        return gen.ask(prompt)
    args = ["--run", "x", "--budget_usd", "1", "--processed", str(P), "--out", str(tmp_path / "L")] + ARGS
    recs = label_outcomes.main(args, complete_fn=f, retriever=retriever())
    assert {r["id"] for r in recs} == {"a1", "h1"}                          # a2 skipped, run finished
    out_f = tmp_path / "L" / "x" / "labels.jsonl"
    out_f.write_text(out_f.read_text(encoding="utf-8") + '{"id": "cut-of', encoding="utf-8")   # hard-stop remnant
    state["fail"] = False
    recs = label_outcomes.main(args, complete_fn=f, retriever=retriever())
    assert {r["id"] for r in recs} == {"a1", "a2", "h1"}                    # re-run finished the skipped one


def _graded_run(tmp_path):
    P = _processed(tmp_path)
    label_outcomes.main(["--run", "g", "--mode", "full", "--budget_usd", "1", "--processed", str(P),
                         "--out", str(tmp_path / "L")] + ARGS, complete_fn=fake_complete([]), retriever=retriever())
    judge_sample.main(["export", "--run", "g", "--n", "6", "--root", str(tmp_path / "L")])
    run = tmp_path / "L" / "g"
    return run, json.loads((run / "judge_sample_key.json").read_text(encoding="utf-8"))


def test_agree_survives_excel_style_csv(tmp_path):
    run, key = _graded_run(tmp_path)
    rows = list(csv.reader((run / "judge_sample.csv").open(encoding="utf-8-sig")))
    lines = [";".join(["item", "question", "reference_answer", "model_answer", "your_verdict (y/n)"])]
    for r in rows[1:]:                                        # semicolons, item '000' -> '0', ' Y ' verdicts, cp1256
        lines.append(";".join([str(int(r[0])), r[1], r[2], r[3], " Y " if key[r[0]]["judge"] else "n"]))
    (run / "judge_sample.csv").write_bytes("\r\n".join(lines).encode("cp1256", errors="replace"))
    acc, _ = judge_sample.main(["agree", "--run", "g", "--root", str(tmp_path / "L")])
    assert acc == 1.0


def test_agree_explains_an_ungraded_file(tmp_path):
    run, _ = _graded_run(tmp_path)
    with pytest.raises(SystemExit, match="No graded rows"):
        judge_sample.main(["agree", "--run", "g", "--root", str(tmp_path / "L")])
