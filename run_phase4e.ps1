# Phase 4e: the FINAL complexity router. Local and FREE (no API calls). ~20 min on an RTX 3070.
#   powershell -ExecutionPolicy Bypass -File .\run_phase4e.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_phase4e.ps1 -ReportOnly    # skip saving the models (2 min)
# Needs Phases 4a-4d (runs_router, runs_suff, data\router_feats). Saved models go to models\final_router (~1.6 GB).
param([int]$Batch = 8, [switch]$ReportOnly)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
foreach ($f in "runs_router\bert-base-arabertv02_seed0\oof.npz", "runs_suff\bert-base-arabertv02_c1_ok_v2_seed0\oof.npz",
               "runs_suff\bert-base-arabertv02_c1_ok_ans_v2_seed0\oof.npz", "data\router_feats\rerank_feats.jsonl",
               "data\router_feats\retrieval_feats.jsonl", "data\router_feats\escalation_feats.jsonl") {
    if (-not (Test-Path $f)) { throw "$f is missing - finish Phases 4a-4d first" }
}

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }

# 1) the RQ1 table: final router (3 stages x 4 operating points) vs every baseline, plus the fresh-run check
Run "final router report (~2 min)" { cmd /c "$py scripts\eval_final_router.py --labels $labels > final_console.txt 2>&1" }
Get-Content runs_final\final_report.txt -Encoding UTF8 | Select-Object -First 40

# 2) train the final models on ALL questions and save them with the chosen thresholds
if (-not $ReportOnly) {
    Run "train + save the final router (~15 min)" { & $py scripts\train_final_router.py --labels $labels --batch $Batch }
    Copy-Item models\final_router\policy.json runs_final\ -Force
}
Compress-Archive -Path runs_final, final_console.txt -DestinationPath phase4e.zip -Force    # report only, no model weights
Write-Host "`nDone. Send phase4e.zip to Claude. (Model weights stay in models\final_router - keep them.)" -ForegroundColor Green
