"""Turn data/splits/*.jsonl into the two binary tasks (torch-free)."""
from dataclasses import dataclass, field
from pathlib import Path

from arroute.data import read_jsonl

DIALECT_EVALS = ["dialect_dev", "dialect_test_ragb", "dialect_dev_ardqa", "dialect_test_ardqa", "dialect_test_pilot"]
COMPLEXITY_EVALS = ["complexity_clean_dev", "complexity_clean_test", "complexity_test"]
LEGACY = "complexity_test"          # mixed ArabicaQA/MAFQA test: domain+length confounded, a STRESS test only


@dataclass
class Task:
    name: str
    train_texts: list
    train_y: list
    evals: dict = field(default_factory=dict)   # name -> (texts, y, sources)
    select: str = ""                            # eval set used for epoch choice + temperature fit
    max_len: int = 64
    pos_name: str = ""


def _pack(examples, label):
    return ([e.question for e in examples], [label(e) for e in examples], [e.source for e in examples])


def rd_any(sd, name):
    f = Path(sd) / f"{name}.jsonl"
    return read_jsonl(f) if f.exists() else []


def load_task(task, splits_dir, add_ardqa_dev=False):
    sd = Path(splits_dir)
    rd = lambda n: read_jsonl(sd / f"{n}.jsonl")
    if task == "dialect":
        keep = lambda ex: [e for e in ex if e.variety in ("msa", "egy")]
        label = lambda e: int(e.variety == "egy")
        train = keep(rd("dialect_train"))
        evals = {n: keep(rd(n)) for n in DIALECT_EVALS}
        select = "dialect_dev_ardqa"            # human-translated dev: the only informative dev set
        if add_ardqa_dev:                       # use its data for training; then it cannot guide selection
            train = train + evals["dialect_dev_ardqa"]
            evals["dialect_dev_ardqa [IN TRAINING]"] = evals.pop("dialect_dev_ardqa")
            select = "dialect_dev"
        return Task("dialect", *_pack(train, label)[:2], {n: _pack(v, label) for n, v in evals.items()},
                    select, 64, "Egyptian")
    if task == "complexity":
        keep = lambda ex: [e for e in ex if e.complexity in ("C1", "C2")]
        label = lambda e: int(e.complexity == "C2")
        train = keep(rd("complexity_clean_train"))
        if not train:
            raise ValueError("complexity_clean_train is empty: run translate_questions.py and make_splits.py")
        evals = {n: _pack(keep(rd(n)), label) for n in COMPLEXITY_EVALS}
        # Single-hop BY CONSTRUCTION (pilot facts; ArDQA extractive QA), in MSA and in Egyptian:
        # every positive prediction is a false positive, and FPR(egy) vs FPR(msa) measures dialect shift.
        for src_file, src_name in (("dialect_test_pilot", "pilot"), ("dialect_test_ardqa", "ardqa")):
            ex = rd_any(sd, src_file)
            for v in ("msa", "egy"):
                sub = [e for e in ex if e.variety == v]
                if sub:
                    evals[f"c1_by_design_{src_name}_{v}"] = (
                        [e.question for e in sub], [0] * len(sub), [e.source for e in sub])
        return Task("complexity", *_pack(train, label)[:2], evals, "complexity_clean_dev", 128, "C2")
    raise ValueError(f"unknown task {task}")
