# AR-Route: Dialect-Aware, Confidence-Gated Retrieval for Arabic RAG

AR-Route is a retrieval-augmented generation (RAG) system for Arabic question answering that works on **Egyptian Arabic** as well as **Modern Standard Arabic (MSA)**. It:

1. **detects the dialect** of the question with a fine-tuned AraBERT classifier,
2. **rewrites Egyptian questions into MSA** before searching (the corpus is MSA),
3. runs **hybrid search** (BM25 + BGE-M3 dense) and **reranks** the candidates with a cross-encoder,
4. answers from only the **top-2 reranked passages** (cheap), and
5. **escalates to iterative multi-step retrieval** only when the first answer is "I don't know" (confidence gate).

On 1,000 held-out Egyptian test questions it reaches **78.7% accuracy at $0.456 per 1,000 questions**. Standard RAG gets 69.4% at $0.534, and always using iterative RAG gets 77.4% at $0.936. AR-Route is more accurate than standard RAG and costs about half as much as iterative RAG.

The repository contains the full research pipeline (phases 0–10), the final system, a Django web app with a Docker setup, tests, and the scripts used to produce every result table.

---

## Table of contents

- [How the final system works](#how-the-final-system-works)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Models and corpus (Hugging Face)](#models-and-corpus-hugging-face)
- [Configuration (.env)](#configuration-env)
- [Running the system](#running-the-system)
- [Web app and API](#web-app-and-api)
- [Docker](#docker)
- [Reproducing the research (phases)](#reproducing-the-research-phases)
- [Data](#data)
- [Cost control and caching](#cost-control-and-caching)
- [Tests](#tests)
- [Evaluation methodology](#evaluation-methodology)
- [Limitations](#limitations)

---

## How the final system works

```
question (Egyptian or MSA)
   │
   ▼
Dialect head (AraBERT, temperature-calibrated) ──► P(Egyptian)
   │   P ≥ 0.8  → rewrite to MSA with the LLM, search the rewrite
   │   P ≤ 0.2  → search the question as typed
   │   otherwise → search BOTH and merge the rankings (RRF)
   ▼
Hybrid retrieval: BM25 + BGE-M3 dense, reciprocal rank fusion → top-20 candidates
   ▼
Reranker (BAAI/bge-reranker-v2-m3) → keep the top-k = 2 passages
   ▼
C1: the LLM answers from those 2 passages (single call)
   ▼
Answer is "I don't know"? ── no ──► return answer
   │ yes
   ▼
C2: iterative RAG (the LLM can request extra searches) → return answer
```

The code is in [arroute/pipeline/final.py](arroute/pipeline/final.py) (`FinalSystem`, `load_final`). Every model is injected as a plain function, so the routing logic can be unit-tested without a GPU or torch.

**The three answering strategies** ([arroute/pipeline/strategies.py](arroute/pipeline/strategies.py)):

| Strategy | Description | Cost |
|---|---|---|
| **C0** | No retrieval: the LLM answers from its own knowledge | 1 call, 0 searches |
| **C1** | Single-step RAG: one search, answer from the top passages | 1 call, 1 search |
| **C2** | Iterative RAG: search, then the LLM may ask for more searches before answering | several calls |

**Fixed parameters** (frozen in [freeze_manifest.json](freeze_manifest.json)): `lo = 0.2`, `hi = 0.8`, `n_cands = 20`, `k = 2` ([runs_v2/v2_config.json](runs_v2/v2_config.json), chosen on training questions only), generator `gpt-4.1-mini`, temperature 0.

---

## Results

All numbers come from [scripts/final_report.py](scripts/final_report.py). The test set is 1,000 questions that were never used for training or tuning: 500 ArDQA (native Egyptian), 200 pilot (native Egyptian) and 300 held-out (translated). `gpt-4.1-mini` is both the generator and the judge.

### Dialect head

| Test set | n | Egyptian recognised | MSA kept as MSA |
|---|---|---|---|
| ArDQA | 1,167 | 96.3% | 99.1% |
| Pilot | 200 | 99.5% | 99.0% |
| Held-out | 300 | 98.0% | 97.0% |
| **All** | 1,667 | **97.0%** | **98.7%** |

### End-to-end accuracy and cost, Egyptian input

| System | Accuracy | 95% CI | $ / 1k q |
|---|---|---|---|
| No retrieval (C0) | 34.7% | 31.8–37.7 | 0.036 |
| Standard RAG (C1, hybrid top-5) | 69.4% | 66.5–72.2 | 0.534 |
| Iterative RAG (always C2) | 77.4% | 74.7–79.9 | 0.936 |
| Adaptive-RAG (text classifier) | 69.3% | 66.4–72.1 | 0.507 |
| Router v1 (quality) | 77.2% | 74.5–79.7 | 0.902 |
| Router v2, learned cascade (quality) | 78.5% | 75.8–80.9 | 0.722 |
| **FINAL (AR-Route)** | **78.7%** | 76.1–81.1 | **0.456** |
| Oracle routing over C0/C1/C2 (upper bound) | 82.9% | 80.4–85.1 | 0.383 |

The final system escalated to C2 on 8.3% of questions. On MSA input it reaches **83.3%**, against 76.3% for standard RAG.

### What each component adds (Egyptian input)

| Step | Accuracy | $ / 1k q | Δ accuracy |
|---|---|---|---|
| Standard RAG | 69.4% | 0.534 | – |
| + dialect rewrite | 71.5% | 0.615 | +2.1 (p = 0.062) |
| + rerank, top-2 passages | 75.2% | 0.302 | +3.7 (p = 0.001) |
| + "I don't know" → C2 | 78.7% | 0.456 | +3.5 (p < 0.001) |

### Validation

- **LLM judge vs. human grading:** 95% agreement on 100 items, Cohen's κ = 0.90.
- **Confirmatory test:** the pipeline was frozen first ([freeze_manifest.json](freeze_manifest.json)). Native speakers then wrote 561 new Egyptian questions, and the frozen system answered 459 of them correctly (81.8%).
- **Out-of-sample check:** on 667 native ArDQA questions outside the test set, the ranking held: final system 77.2%, always-C2 76.6%, at about one third of the cost.

The full tables are in `runs_final/thesis_tables.md`, which is generated locally and not committed.

---

## Repository layout

```
ar-route/
├── arroute/                  # the Python package
│   ├── data/                 # dataset loaders + unified schema (QAExample, Passage)
│   ├── egy/                  # Egyptian→MSA rewriting, Egyptian test-set helpers
│   ├── retrieval/            # BM25, dense (BGE-M3), hybrid (RRF), reranker, Arabic normalisation
│   ├── router/               # complexity routers (v1, v2), escalation, metrics, reports
│   ├── pipeline/             # strategies C0/C1/C2, LLM judge, full router, FINAL system
│   ├── llm.py                # cached LLM client with token accounting and a hard budget guard
│   └── free.py               # optional: free OpenAI-compatible gateway support
├── scripts/                  # one script per step (build data, train, evaluate, report, demo)
├── review/                   # reviewer-response tools: judge agreement, freeze manifest, latency
├── tests/                    # pytest suite (torch-free logic tests)
├── webapp/                   # Django app: UI + JSON API around the final system
├── data/                     # small committed data (labels, features, Egyptian test set, confirm sets)
├── runs_v2/v2_config.json    # chosen k for the final system
├── run_phase*.ps1            # PowerShell runners, one per research phase
├── setup.ps1 / setup_train.ps1
├── Dockerfile / docker-compose.yml
├── requirements*.txt
├── freeze_manifest.json      # hashes of the frozen final pipeline
└── LOCAL_README.md           # short phase-by-phase run notes
```

### Key modules

| File | Purpose |
|---|---|
| [arroute/pipeline/final.py](arroute/pipeline/final.py) | Final system: dialect plan → retrieve → rerank → C1 → escalate to C2 |
| [arroute/pipeline/strategies.py](arroute/pipeline/strategies.py) | C0 / C1 / C2 answering strategies |
| [arroute/pipeline/judge.py](arroute/pipeline/judge.py) | Answer correctness: cheap rules first, then an LLM judge with reference answers |
| [arroute/pipeline/arrouter.py](arroute/pipeline/arrouter.py) | Phase 7 full router (dialect + 3-stage complexity router) |
| [arroute/egy/rewrite.py](arroute/egy/rewrite.py) | `rewrite_llm` (LLM rewriter), `rewrite_rules` (rule baseline), rewrite sanity checks |
| [arroute/retrieval/index.py](arroute/retrieval/index.py) | Corpus, BM25, dense, hybrid retrieval, RRF |
| [arroute/retrieval/rerank.py](arroute/retrieval/rerank.py) | Cross-encoder reranker wrapper |
| [arroute/retrieval/text.py](arroute/retrieval/text.py) | Arabic normalisation and tokenisation |
| [arroute/router/policy.py](arroute/router/policy.py) | Router v1: 3-stage policy with operating points (budget / match_c1 / quality) |
| [arroute/router/v2.py](arroute/router/v2.py) | Router v2: confidence-aware cascade using token log-probabilities |
| [arroute/router/final_eval.py](arroute/router/final_eval.py) | Offline scoring of every system from stored outcomes |
| [arroute/router/metrics.py](arroute/router/metrics.py) | Accuracy, ECE, temperature scaling, Wilson intervals |
| [arroute/llm.py](arroute/llm.py) | Disk-cached LLM calls and a budget that reserves worst-case cost before each call |

---

## Installation

Requirements: Windows or Linux, Python 3.11, and an NVIDIA GPU (recommended for the dialect head, BGE-M3 and the reranker; CPU works but is slow).

```powershell
git clone https://github.com/ahmedewaidar22/Dialect-Aware-Confidence-Gated-Retrieval-for-Arabic-RAG.git
cd Dialect-Aware-Confidence-Gated-Retrieval-for-Arabic-RAG

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate

# 1) PyTorch first, matching your CUDA version (torch >= 2.6 required)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2) project requirements
pip install -r requirements.txt          # core: retrieval, models, LLM client
pip install -r requirements-web.txt      # web app: Django + gunicorn
pip install -r requirements-train.txt    # only if you want to retrain models
```

On Windows, `setup.ps1` and `setup_train.ps1` automate these steps.

---

## Models and corpus (Hugging Face)

The model weights (about 516 MB each) and the retrieval corpus are too large for GitHub. They are hosted at:

**https://huggingface.co/ahmedewaidar22/ar-route**

| Folder | Contents | Needed by |
|---|---|---|
| `dialect_head/` | AraBERT (`aubmindlab/bert-base-arabertv02`) fine-tuned for Egyptian vs. MSA, plus `meta.json` (temperature, lo/hi thresholds) | final system, web app |
| `final_router/` | Phase 4e complexity router (text head, passage readers, stage B/C models, policy) | router experiments |
| `corpus/` | 25,147 passages (`passages.jsonl`), BGE-M3 embeddings (`emb_bgem3.npy`), ids | final system, web app |

Download them into place:

```powershell
python scripts\download_models.py --repo ahmedewaidar22/ar-route
# or only what the final system needs:
python scripts\download_models.py --repo ahmedewaidar22/ar-route --only dialect_head corpus
```

This creates `models/dialect_head`, `models/final_router` and `data/corpus`. The BGE-M3 encoder and the `bge-reranker-v2-m3` reranker are downloaded from Hugging Face automatically the first time they are used.

---

## Configuration (.env)

Copy the template and fill it in:

```powershell
copy .env.example .env
```

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI key used for the generator, rewriter and judge | – |
| `ARROUTE_LLM_MODEL` | LLM model id | `gpt-4.1-mini` |
| `ARROUTE_BUDGET_USD` | Hard spending cap per process (web app) | `5` |
| `ARROUTE_LLM_BASE_URL` / `ARROUTE_LLM_API_KEY` | Optional: an OpenAI-compatible endpoint (read by the phase 8–10 runners) | OpenAI |
| `ARROUTE_LLM_CACHE` | LLM cache folder | `data/llm_cache` |
| `DJANGO_SECRET_KEY` | Django secret | `change-me` |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated host list | `localhost,127.0.0.1` |
| `DJANGO_DEBUG` | `1` for debug mode | `0` |
| `ARROUTE_MODELS` / `ARROUTE_CORPUS` | Override the models / corpus paths | `models`, `data/corpus` |

`.env` is gitignored. Never commit real keys.

---

## Running the system

Ask one question from the command line:

```powershell
python scripts\final_demo.py --question "مين اللي بنى السد العالي؟"
```

Run it on a sample of test questions (Egyptian + MSA, judged automatically):

```powershell
python scripts\final_demo.py --n 6 --budget_usd 0.20
```

For each question the script prints P(Egyptian) and the dialect decision, the search query, the two chosen passages, C1's answer, whether it escalated to C2, the final answer, the cost, the time taken and (for test questions) the judge's verdict.

From Python:

```python
from arroute.egy.rewrite import rewrite_llm
from arroute.llm import Budget, complete_logprobs, complete_with_usage
from arroute.pipeline.final import load_final

budget = Budget(0.50)
gen = {"fn": complete_with_usage, "model": "gpt-4.1-mini", "budget": budget}
system = load_final("models", "data/corpus", rewrite=lambda q: rewrite_llm(q, gen))
# see webapp/rag/services.py for the full wiring of ask_c1 / gen used by system.answer(...)
```

---

## Web app and API

A Django app in [webapp/](webapp/) wraps the final system. The models load once and questions are answered one at a time, because the GPU models are shared.

```powershell
cd webapp
python manage.py runserver 0.0.0.0:8000
```

Then open http://localhost:8000.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/ask` | POST | `{"question": "..."}` → answer JSON (question max 1,000 characters) |
| `/healthz` | GET | `{"ready": true}` once the models are loaded (otherwise 503) |

Example response fields: `answer`, `dialect`, `p_egy`, `query` (the rewritten search query), `passages` (id + text), `c1`, `escalated`, `calls`, `usd`, `seconds`, `budget_spent`.

```bash
curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question": "مين اللي بنى السد العالي؟"}'
```

---

## Docker

The image is based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` and serves the app with gunicorn on port 8000. The models and corpus are **mounted**, not copied, so download them first.

```powershell
python scripts\download_models.py --repo ahmedewaidar22/ar-route --only dialect_head corpus
copy .env.example .env    # fill in OPENAI_API_KEY
docker compose up --build
```

`docker-compose.yml` mounts `./models` and `./data/corpus` read-only and reuses your host Hugging Face cache. It keeps the LLM cache in a named volume and requests one NVIDIA GPU, so you need the NVIDIA Container Toolkit. The first start takes a few minutes while the models load. The health check allows 300 s.

---

## Reproducing the research (phases)

Each phase has a PowerShell runner (`run_phase*.ps1`) that runs the tests first, then the phase's scripts, and zips the results. Runners are safe to stop and re-run: finished work is saved, and LLM calls come from the cache.

| Phase | Runner | What it does | API cost |
|---|---|---|---|
| 0 | `scripts/download_data.py` | Download public datasets into `data/raw/` | free |
| 1 | `scripts/build_dataset.py`, `make_splits.py` | Unified schema, train/calibration/test splits | free |
| 2a | `run_phase2a.ps1` | Train text heads (dialect, complexity) on AraBERT/MARBERT | free (GPU) |
| 3a | `run_phase3a.ps1` | Build the corpus, BM25 + BGE-M3 index, retrieval check | small |
| 3b | `run_phase3b.ps1 -Stage …` | Outcome labels: run C0/C1/C2 on every question, LLM judge, judge agreement | capped |
| 4a | `run_phase4a.ps1` | Complexity router trained on outcome labels | free |
| 4b | `run_phase4b.ps1` | Two-stage router: text head + retrieval-confidence signals | free |
| 4c | `run_phase4c.ps1` | Escalate C1 → C2 when C1 looks unreliable | ~$1 |
| 4d | `run_phase4d.ps1` | Reranker features + passage-sufficiency reader | free |
| 4e | `run_phase4e.ps1` | Final complexity router (v1), saved to `models/final_router` | free |
| 5 | `run_phase5.ps1` | Build the Egyptian test set (ArDQA + pilot + held-out), human review | ~$0.30 |
| 6 / 6b | `run_phase6.ps1`, `run_phase6b.ps1` | Egyptian → MSA rewriting and how to use it (rewrite / both / as typed) | ~$0.10 |
| 7 | `run_phase7.ps1` | Full router: dialect head + rewriter + complexity router | small |
| 8 | `run_phase8.ps1` | End-to-end comparison on 1,000 Egyptian questions vs. standard RAG and Adaptive-RAG | ~$4–5 |
| 9 | `run_phase9.ps1` | Router v2: confidence-aware cascade with log-probabilities | ~$3–4 |
| 10 | `run_phase10.ps1` | Final system + consolidated report, tables and figures | free (demo: cents) |

Run a phase with:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1 -Smoke     # quick check first
powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1
powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1 -ReportOnly
```

See [LOCAL_README.md](LOCAL_README.md) for step-by-step notes per phase. [scripts/README_confirmatory.md](scripts/README_confirmatory.md) describes the confirmatory test protocol, and [review/README_review_scripts.md](review/README_review_scripts.md) covers judge agreement, latency and the freeze manifest.

---

## Data

### Sources

| Dataset | Role |
|---|---|
| **ArabicaQA** | MSA single-hop QA, main corpus source |
| **HotpotQA** (translated to Arabic) | Multi-hop questions + contexts |
| **NQ-Open** (translated) | Single-hop questions |
| **ArDQA** | Native Egyptian questions (test set) |
| **ArabicRAGB** (`HeshamHaroon/ArabicRAGB`) | Dialect split used to train the dialect head |
| **MAFQA** | Additional passages (manual download) |
| Pilot set | 200 validated Egyptian/MSA pairs |

The corpus has 25,147 passages: ArabicaQA 14,051, translated HotpotQA contexts 9,389, MAFQA 1,162, ArDQA 345, pilot 200.

### What is committed

The small data files are in the repo:
- `data/labels/` (outcome labels)
- `data/router_feats/` (router features)
- `data/egy_test/` (1,667-question Egyptian test set + held-out ids)
- `data/confirm*/` (confirmatory question sets)

These are **not committed**, because they are large or can be regenerated:
- `data/raw/` (download with `scripts/download_data.py`)
- `data/processed/`
- `data/splits/`
- `data/llm_cache/`
- `data/corpus/` (on Hugging Face)

---

## Cost control and caching

[arroute/llm.py](arroute/llm.py) wraps every LLM call:

- **Disk cache.** Calls are keyed by (model, prompt, params), so re-runs cost nothing. Parameters starting with `_` (for example `_repeat=1`) only change the cache key, which forces fresh answers.
- **Hard budget.** A `Budget` reserves the worst-case cost of a call (prompt estimate + `max_output_tokens`) *before* making it, and refuses the call if it could exceed the limit, even with parallel workers.
- **Ledger.** Spending is logged to `data/llm_cache/ledger.jsonl`.
- **Free endpoints.** [arroute/free.py](arroute/free.py) lets you use a free OpenAI-compatible gateway (set `ARROUTE_LLM_BASE_URL`).

---

## Tests

```powershell
python -m pytest tests -q
```

The tests cover the data loaders, splits, retrieval, rewriting, the routers (v1, two-stage, v2), escalation, the final system and the Phase 8 evaluation. Models are injected as plain functions, so most of the tests need no GPU, torch or API key.

---

## Evaluation methodology

- **Correctness.** Cheap rules first, then an LLM judge (`gpt-4.1-mini`) that sees the reference answers ([arroute/pipeline/judge.py](arroute/pipeline/judge.py)). It was validated against human grades (κ = 0.90).
- **Offline scoring.** During labelling, every question was answered by C0, C1 and C2. Any routing policy can therefore be scored offline for free by looking up stored outcomes and costs ([arroute/router/final_eval.py](arroute/router/final_eval.py)).
- **Statistics.** 95% Wilson intervals for accuracy, paired bootstrap for differences, exact McNemar tests for paired comparisons.
- **No test leakage.** The dialect-head thresholds, `k` and the router thresholds were all chosen on training/calibration questions only. The 300 held-out questions were excluded from router training.
- **Cost.** Generator tokens at $0.4 / $1.6 per 1M (input/output), including the rewrite call.
- **Pre-registration.** `freeze_manifest.json` stores hashes of the final pipeline. They were recorded before the confirmatory questions were written. Check them with `python review/freeze_manifest.py --verify freeze_manifest.json`.

---

## Limitations

- The dialect head distinguishes **Egyptian vs. MSA** only. Other dialects are untested.
- Answers depend on a commercial LLM (`gpt-4.1-mini`). Results can change as the provider updates the model.
- The held-out set is LLM-translated into Egyptian. The native sets (ArDQA, pilot, confirmatory) are the stronger evidence.
- Single-GPU, one-question-at-a-time serving. The web app is a demo, not a production deployment.
