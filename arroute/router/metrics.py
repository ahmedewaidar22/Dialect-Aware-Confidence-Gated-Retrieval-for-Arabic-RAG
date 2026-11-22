"""Torch-free metrics: accuracy family, calibration (ECE + temperature scaling), Wilson intervals."""
import numpy as np


def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, T=1.0):
    p = softmax(logits, T)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    """Temperature scaling: the single T>0 minimising NLL on a held-out set (grid search, deterministic)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    return float(grid[int(np.argmin([nll(logits, y, T) for T in grid]))])


def ece(probs, y, n_bins=10):
    """Expected calibration error on the top-label confidence."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges, total = np.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def _f1(y, pred, cls):
    tp = int(((pred == cls) & (y == cls)).sum())
    fp = int(((pred == cls) & (y != cls)).sum())
    fn = int(((pred != cls) & (y == cls)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def binary_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    pos, neg = y == 1, y == 0
    recall = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    fpr = float((pred[neg] == 1).mean()) if neg.any() else float("nan")
    return {"n": int(len(y)), "n_pos": int(pos.sum()), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(np.nanmean([recall, 1 - fpr])),
            "macro_f1": float((_f1(y, pred, 0) + _f1(y, pred, 1)) / 2), "f1_pos": float(_f1(y, pred, 1)),
            "recall_pos": recall, "fpr": fpr}


def evaluate_probs(probs, y, sources=None):
    y = np.asarray(y)
    pred = probs.argmax(axis=1)
    out = binary_metrics(y, pred)
    out["ece"] = ece(probs, y)
    if sources is not None:
        sources = np.asarray(sources)
        out["pos_rate_by_source"] = {s: float(pred[sources == s].mean()) for s in sorted(set(sources.tolist()))}
    return out


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
