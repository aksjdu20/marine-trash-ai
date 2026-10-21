param(
    [int]$Camera = 0,
    [ValidateSet('A', 'B')][string]$Variant = 'A',
    [double]$Confidence = 0.25
)
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python virtual environment missing. See README.md.'
}
& $pythonPath -m src.webcam --camera $Camera --variant $Variant --conf $Confidence
exit $LASTEXITCODE
