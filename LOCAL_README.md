# AR-Route: local training (Windows + NVIDIA GPU)

1. Extract this folder (e.g. C:\ar-route).
2. Copy the 12 .jsonl files from Google Drive `MyDrive/ar-route/data/splits/` into `data\splits\`.
3. `powershell -ExecutionPolicy Bypass -File .\setup_train.ps1`   (one time, large download)
4. Open a NEW PowerShell, `cd` here, `.\.venv\Scripts\Activate.ps1`, then `python scripts\check_train_env.py`
5. `powershell -ExecutionPolicy Bypass -File .\run_phase2a.ps1`   (add `-Batch 16` if you run out of VRAM)
6. Send `phase2a_results.zip` to Claude.

Troubleshooting: if MARBERT fails with a torch.load / CVE-2025-32434 error, upgrade PyTorch:
`.\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126`

## Phase 3a (corpus + retrieval)
1. Copy the whole `MyDrive/ar-route/data/processed/` folder from Google Drive into `data\processed\`.
2. In PowerShell: `$env:OPENAI_API_KEY = "sk-..."` and `$env:ARROUTE_LLM_MODEL = "<same model ID as before>"`.
3. `powershell -ExecutionPolicy Bypass -File .\run_phase3a.ps1`  (it pauses after a 10-item trial so you can check it)
4. Send `phase3a_results.zip` to Claude.

## Phase 3b (outcome labels)
Set `$env:OPENAI_API_KEY` and `$env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"`, then run `run_phase3b.ps1` with
`-Stage trial`, then `label`, `judge-export`, `judge-agree`, `sensitivity` (send each zip to Claude).
Every stage has a hard spending cap; spending is logged in data\llm_cache\ledger.jsonl.

## Free local generator (optional)
1. Install Ollama from ollama.com (it runs in the background on Windows).
2. `ollama pull qwen3.5:9b`   (about 6.6 GB)
3. `powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage trial-local`
   Other models: add `-LocalModel <name>`. The judge stays on GPT-4.1 mini (cents), so runs stay comparable.

## Phase 4a (complexity router on outcome labels) - no API key, no cost
1. Make sure `data\labels\main_full\labels.jsonl` exists (written by `run_phase3b.ps1 -Stage label`).
2. `powershell -ExecutionPolicy Bypass -File .\run_phase4a.ps1`   (`-Quick` = 1 AraBERT seed; `-Batch 8` if out of GPU memory)
3. Send `phase4a_router.zip` to Claude.

## Phase 4b (two-stage router: text head + retrieval signals) - no API key, no cost
1. Phase 4a must be finished (it leaves `runs_router\bert-base-arabertv02_seed0..2`).
2. `powershell -ExecutionPolicy Bypass -File .\run_phase4b.ps1`
3. Send `phase4b_router.zip` to Claude.

## Phase 4c (escalate from C1 to C2 only when C1 looks unreliable)
1. `powershell -ExecutionPolicy Bypass -File .\run_phase4c.ps1 -Stage local`   (free, ~30-40 min) -> send `phase4c_local.zip`
2. Set `$env:OPENAI_API_KEY` and `$env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"`, then
   `powershell -ExecutionPolicy Bypass -File .\run_phase4c.ps1 -Stage repeat`  (about $1, hard cap $3) -> send `phase4c_repeat.zip`

## Phase 4d (better escalation detectors: reranker + fixed passage reader) - free
`powershell -ExecutionPolicy Bypass -File .\run_phase4d.ps1`  (~60 min; downloads bge-reranker-v2-m3 once) -> send `phase4d.zip`

## Phase 4e (final complexity router: report + saved models) - free
`powershell -ExecutionPolicy Bypass -File .\run_phase4e.ps1`  (~20 min) -> send `phase4e.zip`. Models stay in `models\final_router`.

## Phase 5 (Egyptian test set)
1. `$env:OPENAI_API_KEY = "sk-..."`; `$env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"`
2. `powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Stage build`   (~$0.30, cap $1)
3. Open `data\egy_test\review.csv` in Excel: for each row put `y` or `n` in **ok**; if wrong, type the right text in **fix**. Save as CSV.
4. `powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Stage import` -> send `phase5.zip`
Note: `data\egy_test\heldout_ids.json` lists 300 labelled questions that Phase 7 keeps OUT of router training.

## Phase 6 (RQ4: Egyptian -> MSA rewriting)
Set `$env:OPENAI_API_KEY` and `$env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"`, then
`powershell -ExecutionPolicy Bypass -File .\run_phase6.ps1` (~15 min, ~$0.10, cap $0.50) -> send `phase6.zip`.

## Phase 6b (better use of the rewrite)
`powershell -ExecutionPolicy Bypass -File .\run_phase6b.ps1` (~10-15 min, ~$0.10, cap $0.50) -> send `phase6b.zip`.

## Phase 7 (the full router: dialect head + rewriter + complexity router)
Set `$env:OPENAI_API_KEY` and `$env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"` (only for the demo), then
`powershell -ExecutionPolicy Bypass -File .\run_phase7.ps1` (~35-45 min) -> send `phase7.zip`.
Saves `models\dialect_head` and `models\complexity_router` (trained WITHOUT the 300 held-out questions).
Try it yourself: `.\.venv\Scripts\python.exe scripts\router_demo.py --question "مين اللي بنى السد العالي؟" --answer`
