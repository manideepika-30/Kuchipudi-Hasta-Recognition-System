@echo off
title Nritya AI - React Frontend (Port 3000)
cd /d "%~dp0frontend"
echo ========================================================
echo Starting Nritya AI React Frontend on http://localhost:3000
echo ========================================================
call npm run dev
pause
