# Phase 9: ROUTER V2 - confidence-aware cascade (C0 first -> C1 on reranked top-k -> C2), compared with Router v1
# on the SAME 1,000 Phase 8 test questions. Needs Phase 8 finished (runs_phase8).
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."
#   (no model variable needed: OpenAI -> gpt-4.1-mini; a free gateway such as freellmAPI -> the model saved in
#    free_model.txt by:  .\.venv\Scripts\python.exe scripts\pick_free_model.py   with BASE_URL http://localhost:3001/v1)
#   powershell -ExecutionPolicy Bypass -File .\run_phase9.ps1              # ~2-3 h, about $3-4 (hard cap $6)
#   powershell -ExecutionPolicy Bypass -File .\run_phase9.ps1 -ReportOnly  # rebuild the report (free)
# Safe to stop (Ctrl+C) and re-run: every stage continues where it stopped; LLM calls are cached.
# If the budget stops it, re-run with a higher cap (e.g. -Budget 7): only missing answers are paid for.
param([double]$Budget = 6, [int]$Workers = 6, [switch]$ReportOnly)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "runs_phase8\states.jsonl", "runs_phase8\answers.jsonl", "models\complexity_router\policy.json",
               "data\labels\main_full\labels.jsonl", "data\egy_test\egy_test.jsonl", "data\egy_test\heldout_ids.json") {
    if (-not (Test-Path $f)) { throw "$f is missing - finish Phase 8 first" }
}
if (-not $ReportOnly) {
    if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
    $free = $env:ARROUTE_LLM_BASE_URL -and ($env:ARROUTE_LLM_BASE_URL -notmatch "api\.openai\.com")
    if (-not $env:ARROUTE_LLM_MODEL) {
        if ($free) {                                   # free gateway (freellmAPI): ONE pinned model, picked by pick_free_model.py
            if (-not (Test-Path free_model.txt)) { throw "Run first:  .\.venv\Scripts\python.exe scripts\pick_free_model.py   (it saves free_model.txt)" }
            $env:ARROUTE_LLM_MODEL = (Get-Content free_model.txt -Raw).Trim()
            if (-not $PSBoundParameters.ContainsKey("Workers")) { $Workers = 2 }        # free tiers have small rate limits
            Write-Host "Free gateway: pinned model from free_model.txt, $Workers workers. Answers will NOT match the gpt-4.1-mini Phase 8 answers." -ForegroundColor Yellow
        } else { $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini" }                             # default = the model used in Phases 3-8
    }
    Write-Host "Model: $env:ARROUTE_LLM_MODEL" -ForegroundColor Cyan
    if (-not $env:ARROUTE_LLM_MODEL.EndsWith("gpt-4.1-mini")) {
        Write-Host "WARNING: model is '$env:ARROUTE_LLM_MODEL', not gpt-4.1-mini: results will not be comparable with Phases 3-8" -ForegroundColor Yellow }
}
New-Item -ItemType Directory -Force runs_v2 | Out-Null

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }

if (-not $ReportOnly) {
    Run "check the LLM endpoint (2 tiny calls)" { & $py scripts\check_llm.py }
    # 1) search + reranker for the training and ArDQA-calibration questions (test questions reuse Phase 8); choose k
    Run "prepare: search + reranker, choose k (GPU, free, ~30-40 min)" {
        cmd /c "$py -u scripts\router_v2.py --stage prepare 2>&1" | Tee-Object -FilePath runs_v2\prepare_console.txt }
    # 2) C0 and C1R answers WITH token log-probabilities (+ C2 for the ArDQA calibration questions), judged
    Run "answer: C0 + C1R with confidence (API, hard cap `$$Budget, ~1-1.5 h)" {
        cmd /c "$py -u scripts\router_v2.py --stage answer --budget_usd $Budget --workers $Workers 2>&1" |
            Tee-Object -FilePath runs_v2\answer_console.txt }
    # 3) reranker check of each C1R answer + answer checks
    Run "features (GPU, free, ~5-10 min)" {
        cmd /c "$py -u scripts\router_v2.py --stage features 2>&1" | Tee-Object -FilePath runs_v2\features_console.txt }
}
# 4) train the two detectors on training + calibration questions, choose thresholds, compare with v1 on the test set
Run "report (CPU, free, ~2 min)" {
    cmd /c "$py -u scripts\router_v2.py --stage report 2>&1" | Tee-Object -FilePath runs_v2\report_console.txt }

Compress-Archive -Path runs_v2\v2_report.txt, runs_v2\v2_report.json, runs_v2\k_report.txt, runs_v2\v2_config.json,
    runs_v2\answers.jsonl, runs_v2\features.jsonl, runs_v2\*_console.txt -DestinationPath phase9.zip -Force
Write-Host "`nDone. Send phase9.zip to Claude." -ForegroundColor Green
