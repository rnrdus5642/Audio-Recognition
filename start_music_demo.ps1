param([int]$Port = 7861)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$demoPython = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $demoPython)) {
    $demoPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $demoPython)) {
    throw 'Python environment missing: .venv-gpu or .venv'
}
& $demoPython -X utf8 -m python.build.build_music_presentation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $demoPython -X utf8 -m python.tools.web_test --music-demo --port $Port
exit $LASTEXITCODE
