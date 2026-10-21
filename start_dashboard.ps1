$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPythonPath)) {
    $taskPythonPath = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $taskPythonPath)) {
    throw 'README에 따라 Python 가상환경을 먼저 설치하세요.'
}
& $taskPythonPath -m streamlit run app.py --server.address 127.0.0.1
