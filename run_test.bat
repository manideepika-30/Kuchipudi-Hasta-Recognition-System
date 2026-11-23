@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0deployment_clean\.venv\Lib\site-packages;%PYTHONPATH%"
"%~dp0Python311\python.exe" "%~dp0deployment_clean\test.py" %*
if errorlevel 1 pause

