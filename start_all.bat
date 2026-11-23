@echo off
title Nritya AI - Launcher
echo ========================================================
echo Launching Nritya AI Platform (Backend + Frontend)
echo ========================================================
start "Nritya AI Backend" cmd /k "%~dp0start_backend.bat"
timeout /t 3 /nobreak >nul
start "Nritya AI Frontend" cmd /k "%~dp0start_frontend.bat"
echo.
echo Both servers have been launched in separate console windows!
echo Open your web browser at: http://localhost:3000
echo.
