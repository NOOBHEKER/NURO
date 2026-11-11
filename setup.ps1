# Plug-and-play bootstrap for Windows.
#
#   .\setup.ps1            # create .venv, install the project, run the doctor
#   .\setup.ps1 -Recreate  # rebuild .venv from scratch
#
# If PowerShell blocks the script:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$NoDev,
    [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py -3", "python", "python3")) {
        $parts = $candidate.Split(" ")
        $exe = $parts[0]
        $pre = @()
        if ($parts.Length -gt 1) { $pre = $parts[1..($parts.Length - 1)] }
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $version = & $exe @pre -c "import sys; print(sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version) {
                return @{ Exe = $exe; Pre = $pre }
            }
        } catch { }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "Python 3.10+ was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run this script."
    exit 2
}

Push-Location $repo
try {
    $env:PYTHONPATH = $repo
    $cliArgs = @("-m", "host.cli", "setup")
    if ($Recreate) { $cliArgs += @("--recreate", "--confirm", "RECREATE") }
    if ($NoDev) { $cliArgs += "--no-dev" }
    if ($SkipDoctor) { $cliArgs += "--skip-doctor" }

    & $python.Exe @($python.Pre) @cliArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
