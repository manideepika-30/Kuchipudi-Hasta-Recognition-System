@echo off
title Nritya AI - FastAPI Backend (Port 9000)
cd /d "%~dp0backend"
echo ========================================================
echo Starting Nritya AI FastAPI Backend on http://127.0.0.1:9000
echo ========================================================
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
pause
