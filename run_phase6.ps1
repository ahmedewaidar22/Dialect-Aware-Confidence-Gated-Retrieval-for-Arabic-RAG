# Phase 6 (RQ4): Egyptian -> MSA rewriting. How much of the dialect gap in retrieval does it close?
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase6.ps1       # ~15 min, about $0.10 (hard cap $0.50)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini'" }
if (-not (Test-Path "data\egy_test\egy_test.jsonl")) { throw "data\egy_test\egy_test.jsonl missing: run run_phase5.ps1 first" }

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
Run "rewrite + retrieval in 5 versions (~15 min)" { cmd /c "$py scripts\rewrite_eval.py --budget_usd 0.5 > rewrite_console.txt 2>&1" }
Get-Content runs_rewrite\rewrite_report.txt -Encoding UTF8 | Select-Object -First 60
Compress-Archive -Path runs_rewrite, rewrite_console.txt -DestinationPath phase6.zip -Force
Write-Host "`nDone. Send phase6.zip to Claude." -ForegroundColor Green
Write-Host "Optional: open runs_rewrite\rewrite_review.csv and mark 50 rewrites ok = y/n (send it back too)." -ForegroundColor Yellow
