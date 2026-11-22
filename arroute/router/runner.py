"""One training run (one seed) around a pluggable backend. Torch-free so it can be tested without a GPU.

backend.fit(train_texts, train_y, select_texts, select_y, seed) -> {"best_epoch": int, "history": [...]}
backend.logits(texts) -> numpy array (n, 2)
"""
import json
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate_probs, fit_temperature, softmax


def run_seed(task, seed, backend, out_dir, config):
    t0 = time.time()
    s_texts, s_y, _ = task.evals[task.select]
    fit_info = backend.fit(task.train_texts, task.train_y, s_texts, s_y, seed)
    sel_logits = backend.logits(s_texts)
    T = fit_temperature(sel_logits, np.asarray(s_y))
    result = {"config": config, "seed": seed, "task": task.name, "positive_class": task.pos_name,
              "select_set": task.select, "temperature": T, "best_epoch": fit_info["best_epoch"],
              "history": fit_info["history"], "n_train": len(task.train_texts), "eval": {}}
    logits_dump = {}
    for name, (texts, y, sources) in task.evals.items():
        lg = backend.logits(texts)
        logits_dump[name] = lg
        result["eval"][name] = {
            "raw": evaluate_probs(softmax(lg, 1.0), y, sources),
            "calibrated": evaluate_probs(softmax(lg, T), y, sources)}
    result["seconds"] = round(time.time() - t0, 1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / "logits.npz", **{k.replace(" ", "_"): v for k, v in logits_dump.items()})
    return result
