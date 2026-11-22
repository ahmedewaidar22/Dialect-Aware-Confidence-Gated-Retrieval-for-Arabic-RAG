# AR-Route one-time setup for Windows (PowerShell).
# Run from the ar-route folder:   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "== Checking prerequisites ==" -ForegroundColor Cyan
py -3.11 --version
git --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

Write-Host "== Creating virtual environment (.venv) ==" -ForegroundColor Cyan
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "== Installing PyTorch with CUDA 12.1 ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121

Write-Host "== Installing project requirements ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Make Python use UTF-8 everywhere (Arabic text) and silence a harmless HF warning.
setx PYTHONUTF8 1 | Out-Null
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1 | Out-Null

Write-Host "`nDone. Open a NEW terminal, then run:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\check_gpu.py"
