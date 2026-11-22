# Phase 7: the FULL router (dialect head + rewriter + complexity router), trained, saved and checked.
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."          (only for the last, optional demo step)
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase7.ps1            # ~35-45 min, demo costs a few cents
#   powershell -ExecutionPolicy Bypass -File .\run_phase7.ps1 -NoDemo    # skip the LLM demo
# Finished steps are skipped on a re-run.
param([int]$Batch = 8, [switch]$NoDemo)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "data\splits\dialect_train.jsonl", "data\egy_test\egy_test.jsonl", "data\egy_test\heldout_ids.json",
               "runs_rewrite\rewrites.jsonl", "data\router_feats\rerank_feats.jsonl") {
    if (-not (Test-Path $f)) { throw "$f is missing - finish Phases 2a-6 first" }
}
New-Item -ItemType Directory -Force runs_router7 | Out-Null

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }

# 1) dialect head: train + SAVE (Phase 2a never saved it), check it on the Egyptian test set
if (-not (Test-Path "models\dialect_head\meta.json")) {
    Run "dialect head (~3 min)" { & $py scripts\train_dialect_final.py }
} else { Write-Host "`n== dialect head: already saved" -ForegroundColor Cyan }

# 2) complexity router retrained WITHOUT the 300 held-out questions (they are in the Egyptian test set)
if (-not (Test-Path "models\complexity_router\policy.json")) {
    Run "complexity router without the held-out questions (~12 min)" {
        & $py scripts\train_final_router.py --labels $labels --exclude data\egy_test\heldout_ids.json --out models\complexity_router --batch $Batch }
} else { Write-Host "`n== complexity router: already saved" -ForegroundColor Cyan }

# 3) RQ5 at the retrieval level: dialect-aware query plan vs simpler plans, on mixed Egyptian + MSA traffic
Run "query plans on mixed traffic (~10-15 min)" { cmd /c "$py scripts\eval_query_plan.py > runs_router7\query_plan_console.txt 2>&1" }
Get-Content runs_router7\query_plan_report.txt -Encoding UTF8 | Select-Object -First 45

# 4) end-to-end smoke test of the saved router (6 Egyptian + 6 MSA questions, answered by the LLM)
if (-not $NoDemo -and ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY) -and $env:ARROUTE_LLM_MODEL) {
    Run "full router demo (a few cents, hard cap `$0.20)" { cmd /c "$py scripts\router_demo.py --n 6 --answer > runs_router7\demo_console.txt 2>&1" }
    Get-Content runs_router7\demo_console.txt -Encoding UTF8
} else { Write-Host "`n== demo skipped (no API key set, or -NoDemo)" -ForegroundColor Yellow }

Copy-Item models\dialect_head\dialect_report.txt, models\dialect_head\meta.json runs_router7\ -Force
Copy-Item models\complexity_router\policy.json runs_router7\complexity_policy.json -Force
Compress-Archive -Path runs_router7 -DestinationPath phase7.zip -Force       # reports only, no model weights
Write-Host "`nDone. Send phase7.zip to Claude. (Models stay in models\dialect_head and models\complexity_router.)" -ForegroundColor Green
