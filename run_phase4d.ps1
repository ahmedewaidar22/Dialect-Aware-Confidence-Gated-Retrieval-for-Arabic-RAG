# Phase 4d: better "will C1 fail / will C2 rescue?" detectors. Local and FREE (no API calls). ~60 min on an RTX 3070.
#   powershell -ExecutionPolicy Bypass -File .\run_phase4d.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_phase4d.ps1 -Batch 4     # if the GPU runs out of memory
# Needs Phase 4c 'local' finished (data\router_feats\escalation_feats.jsonl). Finished steps are skipped on a re-run.
param([int]$Batch = 8, [switch]$Rescue)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"
$model = "aubmindlab/bert-base-arabertv02"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not (Test-Path "data\router_feats\escalation_feats.jsonl")) { throw "Run run_phase4c.ps1 -Stage local first" }

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }

# 1) pretrained reranker (downloads ~2.3 GB once): relevance of C1's passages, of deeper candidates, and of C1's answer
if (-not (Test-Path "data\router_feats\rerank_feats.jsonl")) {
    Run "reranker signals: bge-reranker-v2-m3 on the top-20 passages (~10-15 min)" {
        & $py scripts\rerank_features.py --labels $labels --batch 16 }
} else { Write-Host "`n== reranker signals: already computed" -ForegroundColor Cyan }

# 2) passage reader, fixed: class-weighted loss, 6 epochs, best epoch picked by AUC
Run "passage reader v2: 'will C1 succeed?' BEFORE C1 answers (~16 min)" {
    & $py scripts\train_sufficiency.py --backend torch --model $model --seed 0 --batch $Batch --labels $labels }
Run "passage reader v2 + C1's answer: 'is this answer supported?' AFTER C1 (~16 min)" {
    & $py scripts\train_sufficiency.py --backend torch --model $model --seed 0 --batch $Batch --with_answer --labels $labels }
if ($Rescue) {
    Run "passage reader v2: 'will C2 rescue?' (optional)" {
        & $py scripts\train_sufficiency.py --backend torch --model $model --seed 0 --batch $Batch --target rescue --labels $labels }
}

# 3) report: every detector, before/after C1, honest thresholds, vs always-C2 and the random mix
Run "escalation report (a few minutes)" { cmd /c "$py scripts\eval_escalation.py --labels $labels > escalation_console.txt 2>&1" }
Get-Content runs_suff\escalation_report.txt -Encoding UTF8 | Select-Object -First 50
Copy-Item data\router_feats\rerank_feats_summary.txt runs_suff\ -Force
Compress-Archive -Path runs_suff, escalation_console.txt -DestinationPath phase4d.zip -Force
Write-Host "`nDone. Send phase4d.zip to Claude." -ForegroundColor Green
