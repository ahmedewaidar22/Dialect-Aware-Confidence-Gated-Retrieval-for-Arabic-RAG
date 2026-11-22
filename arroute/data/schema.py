"""Unified schema: every dataset is converted into QAExample (+ Passage for corpora)."""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

COMPLEXITY = {"C0", "C1", "C2"}          # no-retrieval / single-hop / multi-hop
VARIETY = {"msa", "egy", "glf", "lev", "mgr"}


@dataclass
class QAExample:
    id: str                               # globally unique, e.g. "arabicaqa:000123"
    question: str
    variety: str                          # one of VARIETY
    complexity: Optional[str]             # one of COMPLEXITY, or None if unlabeled
    answer: Optional[str] = None
    gold_passage_ids: List[str] = field(default_factory=list)
    source: str = ""                      # dataset name -> needed for the shortcut check
    domain: str = ""                      # e.g. "wikipedia", "fatwa", "vlogs"
    parallel_group: Optional[str] = None  # items that must NEVER be split across train/test
    label_type: str = "silver"            # "silver" (derived) or "gold" (hand-labeled)
    split: str = ""                       # the dataset's ORIGINAL split, if it has one
    extra: dict = field(default_factory=dict)   # raw labels / fields worth keeping

    def validate(self):
        assert self.question.strip(), f"{self.id}: empty question"
        assert self.variety in VARIETY, f"{self.id}: bad variety {self.variety}"
        assert self.complexity is None or self.complexity in COMPLEXITY, \
            f"{self.id}: bad complexity {self.complexity}"
        return self


@dataclass
class Passage:
    id: str
    text: str
    source: str = ""
    title: str = ""


def write_jsonl(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)
    for it in items:                       # validate everything BEFORE touching the file
        if hasattr(it, "validate"):
            it.validate()
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def read_jsonl(path, cls=QAExample):
    with open(path, encoding="utf-8") as f:
        return [cls(**json.loads(line)) for line in f if line.strip()]
