# Collects the evaluation data into hf_ar-route-eval\ for upload to a Hugging Face dataset repo.
# Copies files only; nothing in the project is changed or deleted.
# Run from D:\Masters\ar-route-local\ar-route :   powershell -ExecutionPolicy Bypass -File .\prepare_hf_dataset.ps1

$out = "hf_ar-route-eval"
$map = [ordered]@{
  "test_egy_msa" = @("data\egy_test\egy_test.jsonl", "data\egy_test\heldout_ids.json", "data\egy_test\summary.txt",
                     "data\egy_test\review.csv", "data\egy_test\review_map.json")
  "confirmatory" = @("data\confirm_arabicaqa\confirm_test.jsonl", "data\confirm_arabicaqa\protocol.txt",
                     "data\confirm_arabicaqa\review.csv", "data\confirm_arabicaqa\confirm_test.jsonl.audit.csv")
  "results"      = @("runs_final\per_question.csv", "runs_confirm2\results.jsonl", "data\labels\main_full\labels.jsonl")
  "audit"        = @("runs_audit\provenance_flags.csv", "runs_audit\provenance_report.txt", "freeze_manifest.json")
}

foreach ($dir in $map.Keys) {
  New-Item -ItemType Directory -Force -Path (Join-Path $out $dir) | Out-Null
  foreach ($f in $map[$dir]) {
    if (Test-Path $f) {
      $name = Split-Path $f -Leaf
      if ($f -eq "runs_confirm2\results.jsonl") { $name = "confirmatory_results.jsonl" }
      if ($f -eq "data\labels\main_full\labels.jsonl") { $name = "training_outcome_labels.jsonl" }
      Copy-Item $f (Join-Path (Join-Path $out $dir) $name) -Force
      Write-Host "copied  $f"
    } else { Write-Host "MISSING $f" -ForegroundColor Yellow }
  }
}

Write-Host "`n--- where is the 667-question calibration set? ---"
Get-ChildItem data\splits -ErrorAction SilentlyContinue | Select-Object Name, Length | Format-Table -AutoSize

Write-Host "--- fields and row counts (for the dataset card) ---"
Get-ChildItem $out -Recurse -Include *.jsonl, *.csv | ForEach-Object {
  $lines = (Get-Content $_.FullName -Encoding UTF8 | Measure-Object -Line).Lines
  $first = Get-Content $_.FullName -Encoding UTF8 -TotalCount 1
  if ($_.Extension -eq ".jsonl") { $keys = (($first | ConvertFrom-Json).PSObject.Properties.Name) -join ", " }
  else { $keys = $first.TrimStart([char]0xFEFF) }
  Write-Host "$($_.Directory.Name)\$($_.Name)  rows=$lines"
  Write-Host "    fields: $keys"
}
Write-Host "`nDone. Folder: $((Resolve-Path $out).Path)"
