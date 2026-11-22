# Phase 4a: train the complexity router on the OUTCOME labels and evaluate it as a router. No API calls, no cost.
#   powershell -ExecutionPolicy Bypass -File .\run_phase4a.ps1                # everything (~30-45 min on an RTX 3070)
#   powershell -ExecutionPolicy Bypass -File .\run_phase4a.ps1 -Batch 8       # if you run out of GPU memory
#   powershell -ExecutionPolicy Bypass -File .\run_phase4a.ps1 -Quick         # 1 AraBERT seed only (~10 min)
# Finished runs are skipped on a re-run, so you can stop it and start again.
param([int]$Batch = 16, [switch]$Quick)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"
$model = "aubmindlab/bert-base-arabertv02"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not (Test-Path $labels)) { throw "$labels not found. It is written by: run_phase3b.ps1 -Stage label" }

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
Run "GPU check" { & $py -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - this will be very slow')" }

# 1) cheap baselines (CPU, seconds): if AraBERT is not clearly better than these, it learned nothing new
Run "baseline: question length only"   { & $py scripts\train_router.py --backend length --labels $labels }
Run "baseline: character n-grams"      { & $py scripts\train_router.py --backend ngram  --labels $labels }
Run "baseline: n-grams, class-weighted" { & $py scripts\train_router.py --backend ngram  --labels $labels --balanced }

# 2) AraBERT 3-class head, 5-fold CV. Seeds show how stable the result is.
$seeds = if ($Quick) { @(0) } else { @(0, 1, 2) }
foreach ($s in $seeds) {
    Run "AraBERT router, seed $s" { & $py scripts\train_router.py --backend torch --model $model --seed $s --batch $Batch --labels $labels }
}
foreach ($s in $seeds) {
    Run "AraBERT router, class-weighted, seed $s" { & $py scripts\train_router.py --backend torch --model $model --seed $s --batch $Batch --labels $labels --balanced }
}
if (-not $Quick) {
    Run "AraBERT router, 'none' trained as C0 (variant)" { & $py scripts\train_router.py --backend torch --model $model --seed 0 --batch $Batch --labels $labels --none_as C0 }
}

# 3) report: all runs vs always-C0/C1/C2, the C1->C2 cascade and the oracle - using the stored outcomes
Run "router report" { cmd /c "$py scripts\eval_router.py --labels $labels > router_report_console.txt 2>&1" }
Get-Content runs_router\router_report.txt -Encoding UTF8 | Select-Object -First 40

Compress-Archive -Path runs_router, router_report_console.txt -DestinationPath phase4a_router.zip -Force   # small: no model weights
Write-Host "`nDone. Send phase4a_router.zip to Claude." -ForegroundColor Green
