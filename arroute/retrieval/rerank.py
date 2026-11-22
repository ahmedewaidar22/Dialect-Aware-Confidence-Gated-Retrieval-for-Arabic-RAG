"""Cross-encoder reranker wrapper (BAAI/bge-reranker-v2-m3): scores (question, passage) pairs."""
import numpy as np


class Reranker:
    def __init__(self, name, device=None, max_len=512, batch=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_len, self.batch = max_len, batch

    def score(self, pairs):
        out = np.zeros(len(pairs), np.float32)
        order = np.argsort([len(q) + len(p) for q, p in pairs])               # length-sorted: much faster
        with self.torch.no_grad():
            for i in range(0, len(pairs), self.batch):
                idx = order[i:i + self.batch]
                enc = self.tok([pairs[j][0] for j in idx], [pairs[j][1] for j in idx], padding=True,
                               truncation="only_second", max_length=self.max_len, return_tensors="pt").to(self.device)
                out[idx] = self.model(**enc).logits[:, 0].float().cpu().numpy()
        return out
