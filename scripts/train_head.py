"""Phase 2a: train ONE classification head (dialect OR complexity) with a transformer backbone.

  python scripts/train_head.py --task dialect    --model aubmindlab/bert-base-arabertv02 --seeds 0 1 2
  python scripts/train_head.py --task dialect    --model UBC-NLP/MARBERTv2               --seeds 0 1 2
  python scripts/train_head.py --task complexity --model aubmindlab/bert-base-arabertv02 --seeds 0 1 2 3 4

Writes runs/<task>__<model>__seed<k>/metrics.json (+ logits.npz). Model weights are NOT saved yet:
this phase answers 'which backbone / setup', the final router is trained afterwards.
Needs a GPU for sensible speed (works on CPU, slowly).
"""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from arroute.router.envcheck import TORCH_HELP, torch_at_least  # noqa: E402
from arroute.router.metrics import binary_metrics  # noqa: E402
from arroute.router.outcome import label_metrics  # noqa: E402
from arroute.router.runner import run_seed          # noqa: E402
from arroute.router.tasks import load_task          # noqa: E402


class TorchBackend:
    """The only torch-dependent code. Plain loop on purpose: no Trainer API to break between versions."""

    def __init__(self, model_name, max_len, epochs, lr, batch, num_labels=2, class_weights=None, select="f1"):
        import torch
        self.torch, self.model_name = torch, model_name
        self.max_len, self.epochs, self.lr, self.batch = max_len, epochs, lr, batch
        self.num_labels, self.class_weights = num_labels, class_weights
        self.select = select          # 'f1' (macro-F1 of argmax) or 'auc' (ranking quality; binary only)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp = self.device == "cuda"
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _enc(self, texts):
        return self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt").to(self.device)

    def logits(self, texts, batch=128):
        torch = self.torch
        self.model.eval()
        order = np.argsort([len(t) for t in texts])          # length-sorted batches: faster
        out = np.zeros((len(texts), self.num_labels), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                idx = order[i:i + batch]
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
                    lg = self.model(**self._enc([texts[j] for j in idx])).logits
                out[idx] = lg.float().cpu().numpy()
        return out

    def save(self, path):
        """Save the fine-tuned model + tokenizer (Hugging Face format) for the final router."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    def fit(self, train_texts, train_y, sel_texts, sel_y, seed):
        torch = self.torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        except ValueError as e:
            if "torch.load" in str(e):                     # old torch + a '.bin'-only model such as MARBERT
                raise SystemExit(TORCH_HELP) from e
            raise
        self.model.to(self.device)
        n = len(train_texts)
        total = math.ceil(n / self.batch) * self.epochs
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        try:                                                  # torch >= 2.3
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):                   # older torch
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        rng, y_all = np.random.RandomState(seed), np.asarray(train_y)
        cw = None if self.class_weights is None else torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
        best, history = {"f1": -1.0, "epoch": -1, "state": None}, []
        for epoch in range(self.epochs):
            self.model.train()
            perm, losses = rng.permutation(n), []
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                enc = self._enc([train_texts[j] for j in idx])
                labels = torch.tensor(y_all[idx], dtype=torch.long, device=self.device)
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
                    lg = self.model(**enc).logits
                loss = F.cross_entropy(lg.float(), labels, weight=cw)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            sel_lg = self.logits(sel_texts)
            pred = sel_lg.argmax(axis=1)
            if self.select == "auc" and self.num_labels == 2:
                from arroute.router.route_report import auc
                f1 = auc(sel_lg[:, 1] - sel_lg[:, 0], np.asarray(sel_y) == 1)
            else:
                f1 = (binary_metrics(sel_y, pred) if self.num_labels == 2
                      else label_metrics(sel_y, pred, self.num_labels))["macro_f1"]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "select_macro_f1": f1})
            print(f"    epoch {epoch}: train loss {np.mean(losses):.4f} | select-set {self.select} {f1:.4f}")
            if f1 > best["f1"]:
                best.update(f1=f1, epoch=epoch,
                            state={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()})
        self.model.load_state_dict(best["state"])
        return {"best_epoch": best["epoch"], "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["dialect", "complexity"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=None, help="default: 64 dialect, 128 complexity")
    ap.add_argument("--add_ardqa_dev", action="store_true", help="dialect only: also train on ArDQA dev")
    ap.add_argument("--force", action="store_true", help="re-run seeds that already have metrics.json")
    a = ap.parse_args()

    import torch
    import transformers
    if not torch_at_least(torch.__version__):
        print(f"WARNING: torch {torch.__version__} is older than 2.6; models with only '.bin' weights "
              f"(e.g. MARBERT) will not load.{TORCH_HELP}")
    task = load_task(a.task, a.splits, a.add_ardqa_dev)
    max_len = a.max_len or task.max_len
    slug = a.model.replace("/", "_") + ("__ardqadev" if a.add_ardqa_dev else "")
    print(f"[{a.task}] {a.model} | train {len(task.train_texts):,} | select on {task.select} | max_len {max_len}")
    backend = TorchBackend(a.model, max_len, a.epochs, a.lr, a.batch)
    for seed in a.seeds:
        outdir = Path(a.out) / f"{a.task}__{slug}__seed{seed}"
        if (outdir / "metrics.json").exists() and not a.force:
            print(f"  seed {seed}: already done, skipping (use --force to redo)")
            continue
        print(f"  seed {seed}")
        cfg = {"model": a.model, "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "max_len": max_len,
               "add_ardqa_dev": a.add_ardqa_dev, "torch": torch.__version__,
               "transformers": transformers.__version__}
        res = run_seed(task, seed, backend, outdir, cfg)
        print(f"    done in {res['seconds']}s | best epoch {res['best_epoch']} | T={res['temperature']:.2f}")


if __name__ == "__main__":
    main()
