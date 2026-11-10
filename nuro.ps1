$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

Push-Location $repo
try {
    $env:PYTHONPATH = $repo
    & $python -m host.cli @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
