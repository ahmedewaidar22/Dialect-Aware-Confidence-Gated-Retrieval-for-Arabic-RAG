# Phase 2a on your own GPU. Run from the ar-route folder, after setup_train.ps1:
#   powershell -ExecutionPolicy Bypass -File .\run_phase2a.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_phase2a.ps1 -Batch 16            # if you run out of VRAM
#   powershell -ExecutionPolicy Bypass -File .\run_phase2a.ps1 -WithArdqaDev        # adds the optional variant
# Stops at the first failure. Finished seeds are skipped on a re-run.
param([int]$Batch = 32, [switch]$WithArdqaDev)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}

if (-not (Test-Path "data\splits\dialect_train.jsonl")) {
    throw "data\splits is empty. Copy the .jsonl files from Google Drive (MyDrive/ar-route/data/splits) first."
}

Run "PyTorch version (MARBERT needs >= 2.6)" {
    & $py -c "import sys, torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); print('torch', torch.__version__); sys.exit(0 if v >= (2, 6) else 1)"
    if ($LASTEXITCODE -ne 0) { Write-Host "Fix: .\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126" -ForegroundColor Yellow }
}
Run "tests (expect: 58 passed)" { & $py -m pytest tests -q }
# 'cmd /c ... > file' writes the report as plain UTF-8 (PowerShell's tee/> would re-encode it)
Run "baseline numbers" { cmd /c "$py scripts\shortcut_check.py > shortcut_report.txt 2>&1" }

Run "SMOKE TEST (1 epoch, 1 seed)" {
    & $py scripts\train_head.py --task complexity --model aubmindlab/bert-base-arabertv02 --seeds 0 --epochs 1 --batch $Batch --out runs_smoke
}

foreach ($m in @("aubmindlab/bert-base-arabertv02", "UBC-NLP/MARBERTv2")) {
    Run "dialect head: $m" { & $py scripts\train_head.py --task dialect --model $m --seeds 0 1 2 --batch $Batch }
}
if ($WithArdqaDev) {
    Run "dialect head + ArDQA dev" { & $py scripts\train_head.py --task dialect --model UBC-NLP/MARBERTv2 --seeds 0 1 2 --batch $Batch --add_ardqa_dev }
}
foreach ($m in @("aubmindlab/bert-base-arabertv02", "UBC-NLP/MARBERTv2")) {
    Run "complexity head: $m" { & $py scripts\train_head.py --task complexity --model $m --seeds 0 1 2 3 4 --batch $Batch }
}

Run "summary: dialect" { cmd /c "$py scripts\summarize_runs.py --task dialect > summary_dialect.txt 2>&1" }
Run "summary: complexity" { cmd /c "$py scripts\summarize_runs.py --task complexity > summary_complexity.txt 2>&1" }
Get-Content summary_dialect.txt -Encoding UTF8
Get-Content summary_complexity.txt -Encoding UTF8

Compress-Archive -Path summary_dialect.txt, summary_complexity.txt, shortcut_report.txt, runs -DestinationPath phase2a_results.zip -Force
Write-Host "`nDone. Send phase2a_results.zip to Claude." -ForegroundColor Green
