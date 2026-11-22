# Phase 4c: escalate from C1 to C2 only when C1 looks unreliable.
#   powershell -ExecutionPolicy Bypass -File .\run_phase4c.ps1 -Stage local    # FREE: features + passage reader + report (~30-40 min)
#   powershell -ExecutionPolicy Bypass -File .\run_phase4c.ps1 -Stage repeat   # API: re-run 300 questions twice, hard cap $3 total
# Run 'local' first. 'repeat' needs $env:ARROUTE_LLM_API_KEY (+ $env:ARROUTE_LLM_BASE_URL) and $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini".
# Finished steps are skipped on a re-run.
param([ValidateSet("local", "repeat")][string]$Stage = "local", [int]$Batch = 8, [switch]$TwoSeeds)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"
$model = "aubmindlab/bert-base-arabertv02"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not (Test-Path $labels)) { throw "$labels not found (written by run_phase3b.ps1 -Stage label)" }
if (-not (Test-Path "data\corpus\passages.jsonl")) { throw "data\corpus is missing (written by run_phase3a.ps1)" }

switch ($Stage) {
  "local" {
    Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }
    Run "answer-check features (seconds)" { & $py scripts\escalation_features.py --labels $labels }
    Run "passage reader baseline: TF-IDF (seconds)" { & $py scripts\train_sufficiency.py --backend tfidf --labels $labels }
    $seeds = if ($TwoSeeds) { @(0, 1) } else { @(0) }
    foreach ($s in $seeds) {
        Run "passage reader: AraBERT predicts 'will C1 succeed', seed $s (~15 min)" {
            & $py scripts\train_sufficiency.py --backend torch --model $model --seed $s --batch $Batch --labels $labels }
    }
    Run "passage reader: AraBERT predicts 'will C2 rescue', seed 0 (~15 min)" {
        & $py scripts\train_sufficiency.py --backend torch --model $model --seed 0 --batch $Batch --target rescue --labels $labels }
    Run "escalation report (a few minutes)" { cmd /c "$py scripts\eval_escalation.py --labels $labels > escalation_console.txt 2>&1" }
    Get-Content runs_suff\escalation_report.txt -Encoding UTF8 | Select-Object -First 45
    Copy-Item data\router_feats\escalation_feats_summary.txt runs_suff\ -Force
    Compress-Archive -Path runs_suff, escalation_console.txt -DestinationPath phase4c_local.zip -Force
    Write-Host "`nDone. Send phase4c_local.zip to Claude." -ForegroundColor Green
  }
  "repeat" {
    if (-not ($env:ARROUTE_LLM_API_KEY -or $env:OPENAI_API_KEY)) { throw "Set the key first:   `$env:ARROUTE_LLM_API_KEY = '...'   (and `$env:ARROUTE_LLM_BASE_URL = 'https://.../v1' for your own URL)" }
    if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini'" }
    Run "repeat C1 + C2 on 300 questions, same settings as labelling (cap `$1.50)" {
        & $py scripts\repeat_check.py --n 300 --budget_usd 1.5 --labels $labels }
    Run "repeat with sampling noise, temperature 0.7 (cap `$1.50)" {
        & $py scripts\repeat_check.py --n 300 --budget_usd 1.5 --temperature 0.7 --labels $labels }
    Compress-Archive -Path runs_suff\repeat -DestinationPath phase4c_repeat.zip -Force
    Write-Host "`nDone. Send phase4c_repeat.zip to Claude." -ForegroundColor Green
  }
}
