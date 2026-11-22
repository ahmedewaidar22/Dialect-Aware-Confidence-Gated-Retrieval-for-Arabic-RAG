# Phase 6b: better ways to use the rewrite (minimal-edit rewriting, searching with both questions).
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase6b.ps1      # ~10-15 min, about $0.10 (hard cap $0.50)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini'" }
if (-not (Test-Path "runs_rewrite\rewrites.jsonl")) { throw "runs_rewrite\rewrites.jsonl missing: run run_phase6.ps1 first" }

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
Run "rewrite variants + retrieval (~10-15 min)" { cmd /c "$py scripts\rewrite_variants.py --budget_usd 0.5 > variants_console.txt 2>&1" }
Get-Content runs_rewrite\variants_report.txt -Encoding UTF8 | Select-Object -First 50
Compress-Archive -Path runs_rewrite\variants_report.txt, runs_rewrite\variants_report.json, runs_rewrite\variants_ranks.jsonl, `
    variants_console.txt -DestinationPath phase6b.zip -Force
Write-Host "`nDone. Send phase6b.zip to Claude." -ForegroundColor Green
