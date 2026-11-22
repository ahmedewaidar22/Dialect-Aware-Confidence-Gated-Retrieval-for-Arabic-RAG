"""Local check BEFORE training: is CUDA working, does BERT fit in VRAM, how fast is a training step?

Builds BERT-base-sized models with RANDOM weights (no download), runs the same AMP + GradScaler code
path as train_head.py at the worst-case size (batch 32 x 128 tokens), and prints peak VRAM and speed.
The vocab sizes mimic AraBERT-v0.2 (64k) and MARBERT-v2 (100k).
"""
import sys
import time

import torch

sys.path.insert(0, __import__('pathlib').Path(__file__).resolve().parent.parent.as_posix())
from arroute.router.envcheck import TORCH_HELP, torch_at_least  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if not torch_at_least(torch.__version__):
    print("WARNING: PyTorch is older than 2.6. AraBERT will work, but MARBERT will not load." + TORCH_HELP)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA. Reinstall torch with the CUDA wheel (setup_train.ps1) and check nvidia-smi.")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB (Windows may reserve some for the desktop)")

import torch.nn.functional as F                                             # noqa: E402
from transformers import BertConfig, BertForSequenceClassification         # noqa: E402

BATCH, LEN, STEPS = 32, 128, 5


def measure(vocab, label):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = BertForSequenceClassification(BertConfig(vocab_size=vocab, num_labels=2)).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    ids = torch.randint(5, vocab, (BATCH, LEN), device="cuda")
    mask = torch.ones_like(ids)
    labels = torch.randint(0, 2, (BATCH,), device="cuda")

    def step():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    step()                                                                   # warm-up
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / STEPS * 1000
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label}: {n_params:.0f}M params | peak VRAM {peak:.2f} GB | {ms:.0f} ms/step (batch {BATCH} x {LEN} tokens)")
    # upper-bound run times (real dialect runs use shorter sequences, so they are faster)
    print(f"   upper-bound time per run: dialect ~{195 * 4 * ms / 60000:.1f} min | "
          f"complexity ~{52 * 4 * ms / 60000:.1f} min   (4 epochs; plus model load and evaluation)")


for vocab_size, name in ((64_000, "AraBERT-sized"), (100_000, "MARBERT-sized")):
    measure(vocab_size, name)
print("\nIf peak VRAM is below your free VRAM, training will fit. If not, use  -Batch 16  in run_phase2a.ps1.")
