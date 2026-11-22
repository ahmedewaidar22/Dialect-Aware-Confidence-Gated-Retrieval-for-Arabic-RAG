"""Phase 0 smoke test: confirms CUDA works and the two key models fit on your GPU."""
import time
import torch

print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA. Reinstall torch with the CUDA wheel (see requirements.txt).")

dev = torch.cuda.get_device_properties(0)
print(f"GPU: {dev.name} | VRAM: {dev.total_memory / 1e9:.1f} GB")

# 1) Router backbone: AraBERT forward + backward pass in fp16 (simulates training)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = "aubmindlab/bert-base-arabertv02"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).cuda()
batch = tok(["امتى السد العالي اتبنى؟"] * 32, padding="max_length", max_length=128,
            truncation=True, return_tensors="pt").to("cuda")
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(**batch, labels=torch.zeros(32, dtype=torch.long, device="cuda")).loss
loss.backward()
print(f"AraBERT train step OK (batch 32) | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
del model, batch, loss
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# 2) Dense retriever: BGE-M3 encoding speed
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("BAAI/bge-m3", device="cuda")
st.half()
texts = ["تم بناء السد العالي في أسوان بين عامي 1960 و1970."] * 512
t = time.time()
st.encode(texts, batch_size=64, normalize_embeddings=True)
rate = len(texts) / (time.time() - t)
print(f"BGE-M3 OK | ~{rate:.0f} passages/sec | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("\nAll checks passed. Your GPU is ready for Phase 1.")
