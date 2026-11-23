@echo off
title Kuchipudi Hasta Detection - Frontend
cd /d "%~dp0frontend"
echo ========================================================
echo   Kuchipudi Hasta Detection - Traditional Learning Platform
echo   Starting Vite React Dev Server...
echo ========================================================
npm run dev
if errorlevel 1 pause
