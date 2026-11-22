# Phase 4b: two-stage router. Stage A = the AraBERT text head from Phase 4a decides "skip retrieval?".
# Stage B = a small model on retrieval-confidence signals decides "is one search enough (C1) or iterate (C2)?".
# Local only: no API calls, no cost. Needs Phase 4a finished (runs_router\bert-base-arabertv02_seed0..2).
#   powershell -ExecutionPolicy Bypass -File .\run_phase4b.ps1          # ~10 min on an RTX 3070
# Finished steps are skipped on a re-run.
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$labels = "data\labels\main_full\labels.jsonl"
$feats = "data\router_feats\retrieval_feats.jsonl"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what" }
}
if (-not (Test-Path $labels)) { throw "$labels not found (written by run_phase3b.ps1 -Stage label)" }
if (-not (Test-Path "data\corpus\emb_bgem3.npy")) { throw "data\corpus is missing (written by run_phase3a.ps1)" }
foreach ($s in 0, 1, 2) {
    if (-not (Test-Path "runs_router\bert-base-arabertv02_seed$s\oof.npz")) { throw "Phase 4a run bert-base-arabertv02_seed$s missing: run run_phase4a.ps1 first" }
}

Run "tests (expect: 112 passed)" { & $py -m pytest tests -q }

# 1) one local search per question (same hybrid top-5 as C1) -> confidence features
if (-not (Test-Path $feats)) {
    $dev = if ($Cpu) { "cpu" } else { "cuda" }
    Run "retrieval features (BGE-M3 on $dev + BM25, ~5 min)" { & $py scripts\retrieval_features.py --labels $labels --device $dev }
} else { Write-Host "`n== retrieval features: already computed ($feats)" -ForegroundColor Cyan }

# 2) retrieval signals ALONE (3 classes, no question words): how much do they know?
Run "retrieval-only router (logistic regression)" { & $py scripts\train_router.py --backend retfeat --labels $labels }
Run "retrieval-only router (boosted trees)"       { & $py scripts\train_router.py --backend retfeat --clf hgb --labels $labels }

# 3) two-stage router: text head for C0, retrieval signals for C1 vs C2 (one per AraBERT seed)
foreach ($s in 0, 1, 2) {
    Run "two-stage router, AraBERT seed $s" { & $py scripts\train_router.py --backend twostage --text_run runs_router\bert-base-arabertv02_seed$s --labels $labels }
}
Run "two-stage router, boosted trees in stage B" { & $py scripts\train_router.py --backend twostage --clf hgb --text_run runs_router\bert-base-arabertv02_seed0 --labels $labels }

# 4) report over ALL runs (Phase 4a + 4b), using the stored outcomes
Run "router report (a few minutes: many bootstrap tests)" { cmd /c "$py scripts\eval_router.py --labels $labels > router_report_console.txt 2>&1" }

Copy-Item data\router_feats\retrieval_feats_summary.txt runs_router\ -Force
Compress-Archive -Path runs_router, router_report_console.txt -DestinationPath phase4b_router.zip -Force
Write-Host "`nDone. Send phase4b_router.zip to Claude." -ForegroundColor Green
