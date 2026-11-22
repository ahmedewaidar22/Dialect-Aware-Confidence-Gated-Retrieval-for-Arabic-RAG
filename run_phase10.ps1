# Phase 10: the FINAL system (dialect head -> rewrite -> C1 on the reranked top-k -> "I don't know" -> C2)
# + ONE consolidated results report for the thesis (tables + figures). The report is free (no API calls).
#   powershell -ExecutionPolicy Bypass -File .\run_phase10.ps1           # report only (~1 min, free)
#   powershell -ExecutionPolicy Bypass -File .\run_phase10.ps1 -Demo     # + run the final system on 12 questions
#     (the demo needs $env:ARROUTE_LLM_API_KEY and $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"; a few cents, hard cap $0.20)
# Try your own question afterwards:
#   .\.venv\Scripts\python.exe scripts\final_demo.py --question "مين اللي بنى السد العالي؟"
param([switch]$Demo)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "runs_phase8\states.jsonl", "runs_phase8\answers.jsonl", "runs_v2\answers.jsonl", "runs_v2\v2_report.json",
               "models\complexity_router\policy.json", "models\dialect_head\meta.json") {
    if (-not (Test-Path $f)) { throw "$f is missing - finish Phases 7-9 first" }
}
New-Item -ItemType Directory -Force runs_final | Out-Null

Run "tests (expect: 111 passed)" { & $py -m pytest tests -q }
Run "matplotlib for the figures" {
    cmd /c "$py -c `"import matplotlib`" >nul 2>&1"
    if ($LASTEXITCODE -ne 0) { & $py -m pip install matplotlib -q } }

# 1) the consolidated thesis report: tables 1-8 + cost/accuracy figures, from the saved results of Phases 7-9
Run "thesis report (free, ~1 min)" { cmd /c "$py -u scripts\final_report.py > runs_final\report_console.txt 2>&1" }
Get-Content runs_final\thesis_tables.md -Encoding UTF8 | Select-Object -First 60

# 2) optional: the final system end to end on 6 Egyptian + 6 MSA test questions
if ($Demo) {
    if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set `$env:ARROUTE_LLM_API_KEY first" }
    Run "final system demo (a few cents, hard cap `$0.20)" { cmd /c "$py -u scripts\final_demo.py --n 6 > runs_final\demo_console.txt 2>&1" }
    Get-Content runs_final\demo_console.txt -Encoding UTF8 | Select-Object -Last 70
}

Compress-Archive -Path runs_final -DestinationPath phase10.zip -Force
Write-Host "`nDone. Tables: runs_final\thesis_tables.md | figures: runs_final\cost_accuracy_*.png | send phase10.zip to Claude." -ForegroundColor Green
