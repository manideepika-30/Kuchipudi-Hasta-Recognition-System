@echo off
echo ===================================================
echo Starting Kuchipudi Platform FastAPI Backend Server
echo ===================================================
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    echo Using isolated backend virtual environment...
    .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    echo [ERROR] Virtual environment not found at .venv. Please create it first.
    pause
)
