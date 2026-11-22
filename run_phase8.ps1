# Phase 8: FINAL comparison on Egyptian input - standard RAG vs Adaptive-RAG vs YOUR router (+ ablations, MSA upper bound).
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1 -Smoke        # 10 questions, ~5 min, a few cents: check first
#   powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1               # 1,000 questions, ~2-3 h, about $4-5 (hard cap $5)
#   powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1 -Sol          # + stronger-rewriter check (gpt-6.1-sol, hard cap $2)
#   powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1 -ReportOnly   # rebuild the report from saved results (free)
# Safe to stop (Ctrl+C) and re-run: finished answers are saved and repeated LLM calls come from the cache for free.
# If the budget stops it early, re-run with a higher cap, e.g. -Budget 6: only the missing answers are paid for.
param([double]$Budget = 5, [int]$NArdqa = 500, [int]$Workers = 6, [switch]$Sol, [double]$SolBudget = 2,
      [switch]$Smoke, [switch]$ReportOnly)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "models\dialect_head\meta.json", "models\complexity_router\policy.json", "data\egy_test\egy_test.jsonl",
               "data\corpus\passages.jsonl") {
    if (-not (Test-Path $f)) { throw "$f is missing - finish Phase 7 first" }
}
if (-not $ReportOnly) {
    if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
    if (-not $env:ARROUTE_LLM_MODEL) { throw "Set `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini' (or your URL's name for it)" }
    if (-not $env:ARROUTE_LLM_MODEL.EndsWith("gpt-4.1-mini")) {
        Write-Host "WARNING: model is '$env:ARROUTE_LLM_MODEL', not gpt-4.1-mini: results will not be comparable with Phases 3-8" -ForegroundColor Yellow }
}
$out = if ($Smoke) { "runs_phase8_smoke" } else { "runs_phase8" }
New-Item -ItemType Directory -Force $out | Out-Null

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
if (-not $ReportOnly) { Run "check the LLM endpoint (2 tiny calls)" { & $py scripts\check_llm.py } }

if ($Smoke) {
    Run "SMOKE TEST: 10 questions end to end (hard cap `$0.30)" {
        cmd /c "$py -u scripts\eval_phase8.py --limit 10 --budget_usd 0.3 --workers $Workers --out $out 2>&1" |
            Tee-Object -FilePath "$out\phase8_console.txt" }
    Write-Host "`nSmoke test OK. Now run the full Phase 8:  powershell -ExecutionPolicy Bypass -File .\run_phase8.ps1" -ForegroundColor Green
    exit 0
}

$extra = if ($ReportOnly) { "--report_only" } else { "" }
Run "final comparison: route (GPU) -> answer C0/C1/C2 x 4 inputs (API, hard cap `$$Budget) -> stage C -> report" {
    cmd /c "$py -u scripts\eval_phase8.py --budget_usd $Budget --n_ardqa $NArdqa --workers $Workers --out $out $extra 2>&1" |
        Tee-Object -FilePath "$out\phase8_console.txt" }

if ($Sol -and -not $ReportOnly) {
    Run "stronger rewriter check: gpt-6.1-sol on 500 native Egyptian questions (hard cap `$$SolBudget)" {
        cmd /c "$py -u scripts\sol_rewrite_check.py --budget_usd $SolBudget --out $out 2>&1" |
            Tee-Object -FilePath "$out\sol_console.txt" }
}

Compress-Archive -Path $out -DestinationPath phase8.zip -Force       # reports + answers, no model weights
Write-Host "`nDone. Send phase8.zip to Claude." -ForegroundColor Green
