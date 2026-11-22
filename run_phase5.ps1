# Phase 5: the Egyptian test set.
#   $env:ARROUTE_LLM_BASE_URL = "https://your-endpoint/v1"   (optional; default: OpenAI)
#   $env:ARROUTE_LLM_API_KEY  = "..."
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Stage build    # ~5 min, about $0.30 (hard cap $1)
#   ... open data\egy_test\review.csv in Excel, fill 'ok' (y/n) and, if needed, 'fix'; save as CSV ...
#   powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Stage import   # free
param([ValidateSet("build", "import")][string]$Stage = "build")
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "data\labels\main_full\labels.jsonl", "data\processed\ardqa.jsonl", "data\processed\pilot.jsonl",
               "data\processed\arabicaqa.jsonl", "data\processed\hotpot_ctx_ar.jsonl", "data\corpus\passages.jsonl") {
    if (-not (Test-Path $f)) { throw "$f is missing" }
}

switch ($Stage) {
  "build" {
    if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
    if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini'" }
    Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
    Run "build the Egyptian test set (hard cap `$1)" { & $py scripts\build_egy_testset.py --stage build --budget_usd 1 }
    Compress-Archive -Path data\egy_test\summary.txt, data\egy_test\egy_test.jsonl, data\egy_test\heldout_ids.json `
        -DestinationPath phase5_build.zip -Force
    Write-Host "`nNow open data\egy_test\review.csv in Excel and check it (about 1 hour)." -ForegroundColor Yellow
    Write-Host "Then run:  powershell -ExecutionPolicy Bypass -File .\run_phase5.ps1 -Stage import" -ForegroundColor Yellow
    Write-Host "You can send phase5_build.zip to Claude now if you want an early look." -ForegroundColor Green
  }
  "import" {
    Run "apply your review" { & $py scripts\build_egy_testset.py --stage import }
    Compress-Archive -Path data\egy_test\summary.txt, data\egy_test\egy_test.jsonl, data\egy_test\heldout_ids.json, `
        data\egy_test\review.csv -DestinationPath phase5.zip -Force
    Write-Host "`nDone. Send phase5.zip to Claude." -ForegroundColor Green
  }
}
