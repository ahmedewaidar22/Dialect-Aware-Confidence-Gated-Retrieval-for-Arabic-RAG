# Phase 3a fix: re-translate ONLY the HotpotQA items whose Arabic question leaked the answer (or that failed),
# rebuild the corpus, and re-run the retrieval check (now with paired significance tests).
# Use the SAME model ID as before, so the clean items come from the cache at no cost:
#   $env:OPENAI_API_KEY    = "sk-..."
#   $env:ARROUTE_LLM_MODEL = "<same model ID as the first run>"
#   powershell -ExecutionPolicy Bypass -File .\run_phase3a_fix.ps1
param([int]$N = 1000, [int]$Workers = 8)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not $env:OPENAI_API_KEY)    { throw "Set the key first:   `$env:OPENAI_API_KEY = 'sk-...'" }
if (-not $env:ARROUTE_LLM_MODEL) { throw "Set the model first: `$env:ARROUTE_LLM_MODEL = '<same model id as before>'" }

Run "tests (expect: 58 passed)" { & $py -m pytest tests -q }
Run "quality report BEFORE" { cmd /c "$py scripts\check_hotpot_ctx.py > hotpot_check_before.txt 2>&1" }
Get-Content hotpot_check_before.txt -Encoding UTF8
Run "re-translate only problem items (clean items come from the cache)" {
    & $py scripts\translate_hotpot_context.py --n $N --workers $Workers }
Run "quality report AFTER" { cmd /c "$py scripts\check_hotpot_ctx.py > hotpot_check_after.txt 2>&1" }
Get-Content hotpot_check_after.txt -Encoding UTF8
Run "rebuild corpus + embeddings" { & $py scripts\build_corpus.py --dense }
Run "retrieval check with significance tests" { cmd /c "$py scripts\retrieval_check.py > retrieval_report.txt 2>&1" }

Compress-Archive -Path hotpot_check_before.txt, hotpot_check_after.txt, retrieval_report.txt, data\corpus\corpus_stats.txt -DestinationPath phase3a_fix_results.zip -Force
Write-Host "`nDone. Send phase3a_fix_results.zip to Claude." -ForegroundColor Green
