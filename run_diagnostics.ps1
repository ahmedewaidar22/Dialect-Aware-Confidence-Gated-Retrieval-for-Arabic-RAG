# Cheap CPU diagnostics on the finished Phase 2a runs (no training, about a minute).
#   powershell -ExecutionPolicy Bypass -File .\run_diagnostics.ps1
# Writes diagnostics.zip: send it to Claude. ('cmd /c ... > file' keeps the Arabic text as plain UTF-8.)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$cx = "runs\complexity__aubmindlab_bert-base-arabertv02__seed0"
$dl = "runs\dialect__aubmindlab_bert-base-arabertv02__seed0"
foreach ($d in @($cx, $dl)) { if (-not (Test-Path $d)) { throw "missing $d (run run_phase2a.ps1 first)" } }

function Run($what, [scriptblock]$cmd) {
    Write-Host "== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}

Run "baselines + transfer baselines" { cmd /c "$py scripts\shortcut_check.py > shortcut_report.txt 2>&1" }
Run "complexity: native C1 questions flagged as multi-hop (false positives)" {
    cmd /c "$py scripts\inspect_errors.py --run $cx --set complexity_test --source arabicaqa --kind fp --n 30 > err_native_c1_flagged.txt 2>&1" }
Run "complexity: native C1 questions NOT flagged (for contrast)" {
    cmd /c "$py scripts\inspect_errors.py --run $cx --set complexity_test --source arabicaqa --kind tn --n 15 > err_native_c1_ok.txt 2>&1" }
Run "complexity: fatwa multi-hop questions that were missed" {
    cmd /c "$py scripts\inspect_errors.py --run $cx --set complexity_test --source mafqa --kind fn --n 10 > err_fatwa_c2_missed.txt 2>&1" }
Run "complexity: errors on the clean test (C1 flagged C2)" {
    cmd /c "$py scripts\inspect_errors.py --run $cx --set complexity_clean_test --kind fp --n 15 > err_clean_fp.txt 2>&1" }
Run "complexity: errors on the clean test (C2 missed)" {
    cmd /c "$py scripts\inspect_errors.py --run $cx --set complexity_clean_test --kind fn --n 15 > err_clean_fn.txt 2>&1" }
Run "dialect: Egyptian questions called MSA" {
    cmd /c "$py scripts\inspect_errors.py --run $dl --set dialect_test_ardqa --kind fn --n 25 > err_dialect_egy_as_msa.txt 2>&1" }
Run "dialect: MSA questions called Egyptian" {
    cmd /c "$py scripts\inspect_errors.py --run $dl --set dialect_test_ardqa --kind fp --n 25 > err_dialect_msa_as_egy.txt 2>&1" }

Compress-Archive -Path shortcut_report.txt, err_*.txt -DestinationPath diagnostics.zip -Force
Write-Host "`nDone. Send diagnostics.zip to Claude." -ForegroundColor Green
