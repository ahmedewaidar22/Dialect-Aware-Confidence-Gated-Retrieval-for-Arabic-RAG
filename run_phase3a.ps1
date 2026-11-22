# Phase 3a: translate HotpotQA items WITH their paragraphs, build the Arabic corpus + indexes,
# and re-measure retrieval (incl. the corrected dialect gap). Run from the ar-route folder:
#   $env:OPENAI_API_KEY    = "sk-..."
#   $env:ARROUTE_LLM_MODEL = "<the SAME model ID you used for the question translations>"
#   powershell -ExecutionPolicy Bypass -File .\run_phase3a.ps1                  # 1000 items
#   powershell -ExecutionPolicy Bypass -File .\run_phase3a.ps1 -N 300           # smaller / cheaper
#   powershell -ExecutionPolicy Bypass -File .\run_phase3a.ps1 -SkipTranslation # corpus + check only
param([int]$N = 1000, [int]$Workers = 8, [switch]$SkipTranslation)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}

if (-not (Test-Path "data\processed\arabicaqa_passages.jsonl")) {
    throw "Copy the whole data/processed folder from Google Drive (MyDrive/ar-route/data/processed) into data\processed first."
}
if (-not $SkipTranslation) {
    if (-not $env:OPENAI_API_KEY)    { throw "Set the key first:   `$env:OPENAI_API_KEY = 'sk-...'" }
    if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = '<model id>'" }
}

Run "install extra packages (BM25, BGE-M3, OpenAI)" { & $py -m pip install -q -r requirements-train.txt }
Run "tests (expect: 58 passed)" { & $py -m pytest tests -q }

if (-not $SkipTranslation) {
    Run "TRIAL: translate 10 items (question, answer, 10 paragraphs each)" {
        & $py scripts\translate_hotpot_context.py --n 10 --workers 4 }
    notepad data\raw\translated\hotpot_ctx_trial.txt
    Read-Host "Check the 10 examples in Notepad (names consistent? meaning kept?). Enter = translate $N items, Ctrl+C = stop"
    Run "translate $N items ($Workers in parallel; cached, so a re-run resumes)" {
        & $py scripts\translate_hotpot_context.py --n $N --workers $Workers }
}

Run "build corpus + BGE-M3 embeddings (first run downloads BGE-M3, about 2 GB)" { & $py scripts\build_corpus.py --dense }
Run "retrieval check (several minutes)" { cmd /c "$py scripts\retrieval_check.py > retrieval_report.txt 2>&1" }
Get-Content retrieval_report.txt -Encoding UTF8

$files = @("retrieval_report.txt", "data\corpus\corpus_stats.txt")
if (Test-Path "data\raw\translated\hotpot_ctx_review.csv") { $files += "data\raw\translated\hotpot_ctx_review.csv" }
Compress-Archive -Path $files -DestinationPath phase3a_results.zip -Force
Write-Host "`nDone. Send phase3a_results.zip to Claude." -ForegroundColor Green
