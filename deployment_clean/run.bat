@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0.venv\Lib\site-packages;%PYTHONPATH%"
"%~dp0..\Python311\python.exe" "%~dp0test.py" %*
if errorlevel 1 pause

