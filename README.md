# AR-Route

Arabic question-answering RAG system with a dialect head (Egyptian vs. MSA), Egyptian→MSA rewriting, and a complexity router, plus a Django web app.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# install PyTorch first, matching your CUDA version (torch >= 2.6), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt -r requirements-web.txt
copy .env.example .env   # then put your OpenAI key in .env
```

## Models and corpus

The model weights (~516 MB each) and the retrieval corpus are on the Hugging Face Hub:
**https://huggingface.co/ahmedewaidar22/ar-route**

```powershell
python scripts\download_models.py --repo ahmedewaidar22/ar-route
```

This creates `models/dialect_head`, `models/final_router` and `data/corpus`.

## Run the web app

With Docker (NVIDIA GPU):

```powershell
docker compose up --build
```

Then open http://localhost:8000.

## Experiments

The `run_phase*.ps1` scripts reproduce each research phase; see [LOCAL_README.md](LOCAL_README.md).
