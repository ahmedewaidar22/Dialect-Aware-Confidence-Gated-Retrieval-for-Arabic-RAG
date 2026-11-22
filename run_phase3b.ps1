# Phase 3b: outcome labels. Run the stages in order; send Claude the zip each stage produces.
#   $env:OPENAI_API_KEY    = "sk-..."
#   $env:ARROUTE_LLM_MODEL = "gpt-4.1-mini"
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage trial         # 30 questions, cap $1
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage c2-trial      # new C2 prompt on 100 HotpotQA q, cap $1
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage label         # ~1,900 questions, all strategies, cap $10
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage judge-export  # 100 answers for YOU to grade
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage judge-agree   # after you graded them
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage sensitivity   # GPT-6.1 Sol, 300 q, cap $5
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage trial-local   # FREE local generator (Ollama), 20 q
#   powershell -ExecutionPolicy Bypass -File .\run_phase3b.ps1 -Stage label-local   # local generator on all questions
# Every cap is HARD: a call is refused if its worst-case cost could cross the limit. Re-runs resume for free.
param([ValidateSet("trial", "c2-trial", "label", "judge-export", "judge-agree", "sensitivity", "trial-local", "label-local")]
      [string]$Stage = "trial", [int]$Workers = 4,
      [string]$LocalModel = "qwen3.5:9b", [string]$BaseUrl = "http://localhost:11434/v1")
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not $env:OPENAI_API_KEY)    { throw "Set the key first:   `$env:OPENAI_API_KEY = 'sk-...'" }
if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = 'gpt-4.1-mini'" }
if (-not (Test-Path "data\corpus\emb_bgem3.npy")) { throw "data\corpus is missing: run run_phase3a.ps1 first" }
$MINI = $env:ARROUTE_LLM_MODEL
# USD per 1M tokens (input,output). Sol checked Oct 2026; 4.1-mini = 2025 launch price: verify on OpenAI's pricing page.
$prices = @("--price", "gpt-4.1-mini=0.40,1.60", "--price", "gpt-6.1-sol=2,10")

function Check-Local {
    try { $m = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 5 }
    catch { throw "No local LLM server at $BaseUrl. Install Ollama from ollama.com, then run:  ollama pull $LocalModel" }
    if (-not ($m.data.id -contains $LocalModel)) { throw "Model '$LocalModel' is not installed. Run:  ollama pull $LocalModel" }
}

switch ($Stage) {
  "trial" {
    Run "tests (expect: 58 passed)" { & $py -m pytest tests -q }
    Run "TRIAL: 10 questions per source with $MINI (hard cap `$1)" {
        & $py scripts\label_outcomes.py --run trial_mini --n_per_source 10 --budget_usd 1 --workers $Workers @prices }
    Compress-Archive -Path data\labels\trial_mini\summary.txt, data\labels\trial_mini\labels.jsonl -DestinationPath phase3b_trial.zip -Force
    Write-Host "`nSend phase3b_trial.zip to Claude before running -Stage label." -ForegroundColor Green
  }
  "c2-trial" {
    Run "tests (expect: 58 passed)" { & $py -m pytest tests -q }
    Run "NEW C2 PROMPT: all strategies on the first 100 HotpotQA questions (hard cap `$1)" {
        & $py scripts\label_outcomes.py --run c2_trial --mode full --sources hotpot_ctx_ar --n_per_source 100 `
            --budget_usd 1 --workers $Workers @prices }
    Run "compare with the previous run on the same questions" {
        cmd /c "$py scripts\compare_labels.py main_mini c2_trial > compare_c2.txt 2>&1" }
    Get-Content compare_c2.txt -Encoding UTF8
    Compress-Archive -Path compare_c2.txt, data\labels\c2_trial\summary.txt, data\labels\c2_trial\labels.jsonl -DestinationPath phase3b_c2_trial.zip -Force
    Write-Host "`nSend phase3b_c2_trial.zip to Claude before running -Stage label." -ForegroundColor Green
  }
  "label" {
    Run "OUTCOME LABELS: HotpotQA + 1,000 ArabicaQA, ALL three strategies per question, $MINI (hard cap `$10)" {
        & $py scripts\label_outcomes.py --run main_full --mode full --budget_usd 10 --workers $Workers @prices }
    Compress-Archive -Path data\labels\main_full\summary.txt, data\labels\main_full\labels.jsonl -DestinationPath phase3b_labels_full.zip -Force
    Write-Host "`nSend phase3b_labels.zip to Claude." -ForegroundColor Green
  }
  "judge-export" {
    Run "export 100 judged answers" { & $py scripts\judge_sample.py export --run main_full --n 100 }
    Invoke-Item data\labels\main_full\judge_sample.csv
    Write-Host "Fill 'your_verdict (y/n)' for every row, save as CSV (UTF-8), then run -Stage judge-agree" -ForegroundColor Yellow
  }
  "judge-agree" {
    Write-Host "`n== judge vs your grades" -ForegroundColor Cyan
    cmd /c "$py scripts\judge_sample.py agree --run main_full > judge_agreement.txt 2>&1"
    $code = $LASTEXITCODE
    Get-Content judge_agreement.txt -Encoding UTF8                  # show the message even when it failed
    if ($code -ne 0) { throw "FAILED: judge vs your grades (see the message above)" }
    Compress-Archive -Path judge_agreement.txt, data\labels\main_full\judge_sample.csv -DestinationPath phase3b_judge.zip -Force
  }
  "trial-local" {
    Check-Local
    Run "TRIAL with LOCAL generator $LocalModel (free; judge = $MINI, cap `$1)" {
        & $py scripts\label_outcomes.py --run trial_local --n_per_source 10 --gen_model $LocalModel --gen_base_url $BaseUrl `
            --judge_model $MINI --retriever_device cpu --workers 2 --budget_usd 1 @prices }
    Run "compare with the mini trial" { cmd /c "$py scripts\compare_labels.py trial_mini trial_local > compare_local.txt 2>&1" }
    Get-Content compare_local.txt -Encoding UTF8
    Compress-Archive -Path compare_local.txt, data\labels\trial_local\summary.txt, data\labels\trial_local\labels.jsonl -DestinationPath phase3b_trial_local.zip -Force
    Write-Host "`nSend phase3b_trial_local.zip to Claude." -ForegroundColor Green
  }
  "label-local" {
    Check-Local
    Run "OUTCOME LABELS with LOCAL generator $LocalModel, all strategies (judge only is paid, cap `$3; takes hours)" {
        & $py scripts\label_outcomes.py --run main_local --mode full --gen_model $LocalModel --gen_base_url $BaseUrl `
            --judge_model $MINI --retriever_device cpu --workers 2 --budget_usd 3 @prices }
    Compress-Archive -Path data\labels\main_local\summary.txt, data\labels\main_local\labels.jsonl -DestinationPath phase3b_labels_local.zip -Force
  }
  "sensitivity" {
    Run "SENSITIVITY: GPT-6.1 Sol (low effort) on 100 questions per source (hard cap `$5)" {
        & $py scripts\label_outcomes.py --run sens_sol --gen_model gpt-6.1-sol --judge_model $MINI --effort low `
            --max_out_gen 2000 --n_per_source 100 --budget_usd 5 --workers $Workers @prices }
    Run "compare with the mini labels" { cmd /c "$py scripts\compare_labels.py main_full sens_sol > compare_report.txt 2>&1" }
    Get-Content compare_report.txt -Encoding UTF8
    Compress-Archive -Path compare_report.txt, data\labels\sens_sol\summary.txt -DestinationPath phase3b_sensitivity.zip -Force
  }
}
