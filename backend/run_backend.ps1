# PowerShell script to run Kuchipudi Backend
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "Starting Kuchipudi Platform FastAPI Backend Server..." -ForegroundColor Cyan
if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
} else {
    Write-Host "[ERROR] Virtual environment not found at .venv" -ForegroundColor Red
}
