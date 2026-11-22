# One-time setup for LOCAL training on Windows (PowerShell). Run from the ar-route folder:
#   powershell -ExecutionPolicy Bypass -File .\setup_train.ps1
$ErrorActionPreference = "Stop"

Write-Host "== Prerequisites ==" -ForegroundColor Cyan
py -3.11 --version          # if you have another Python 3.10-3.12, change 3.11 below
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

Write-Host "== Virtual environment (.venv) ==" -ForegroundColor Cyan
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "== PyTorch with CUDA 12.6 (large download) ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126

Write-Host "== Training requirements ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt

setx PYTHONUTF8 1 | Out-Null
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1 | Out-Null

Write-Host "`nDone. Open a NEW PowerShell window, then:" -ForegroundColor Green
Write-Host "  cd <this folder>"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\check_train_env.py"
