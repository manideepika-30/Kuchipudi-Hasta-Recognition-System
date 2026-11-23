$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$env:PYTHONPATH = "$PSScriptRoot\deployment_clean\.venv\Lib\site-packages"
& "$PSScriptRoot\Python311\python.exe" "$PSScriptRoot\deployment_clean\test.py" @args

