# Put the `nuro` console script on this user's PATH.
#
#   .\install.ps1           # show what would change (no changes)
#   .\install.ps1 -Apply    # write the user PATH
#
# Run .\setup.ps1 first: it creates .venv and installs the console script.
[CmdletBinding()]
param([switch]$Apply)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$venvScripts = Join-Path $repo ".venv\Scripts"
$console = Join-Path $venvScripts "nuro.exe"

if (-not (Test-Path $console)) {
    Write-Host "nuro.exe was not found at $console" -ForegroundColor Yellow
    Write-Host "Run .\setup.ps1 first, then re-run this script."
    exit 2
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$entries = @()
if ($userPath) { $entries = $userPath.Split(";", [StringSplitOptions]::RemoveEmptyEntries) }

$already = $entries | Where-Object { $_.TrimEnd("\") -ieq $venvScripts.TrimEnd("\") }
if ($already) {
    Write-Host "Already on the user PATH: $venvScripts" -ForegroundColor Green
    exit 0
}

if (-not $Apply) {
    Write-Host "Would add this directory to your user PATH:"
    Write-Host "  $venvScripts"
    Write-Host ""
    Write-Host "Re-run with -Apply to write it:"
    Write-Host "  .\install.ps1 -Apply"
    exit 0
}

$newPath = (@($entries) + $venvScripts) -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
Write-Host "Added to the user PATH: $venvScripts" -ForegroundColor Green
Write-Host "Open a new terminal, then run:  nuro demo"
