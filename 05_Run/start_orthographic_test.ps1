param(
    [ValidateRange(1, 65535)][int]$Port = 8782,
    [string]$Bundle = '',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$testRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$testPython = Join-Path $testRoot '01_Project\.venv-gpu\Scripts\python.exe'
$testScript = Join-Path $testRoot '01_Project\python\tools\child_orthographic_web\server.py'
if (-not $Bundle) {
    $Bundle = Join-Path $testRoot '04_Models_and_Experiments\orthographic-jamo-20261027\releases\epoch3-final-web-20261031'
}
foreach ($required in @($testPython, $testScript, (Join-Path $Bundle 'release.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing: $required" }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is in use. Open http://127.0.0.1:$Port/ or choose another -Port."
}
Write-Host "Orthographic model test: http://127.0.0.1:$Port/"
Write-Host "Device: $Device. Keep this window open. Ctrl+C stops this test server."
& $testPython -B -X utf8 -u $testScript --bundle $Bundle --port $Port --device $Device --threads 2
if ($LASTEXITCODE -ne 0) { throw "Test server exited with code $LASTEXITCODE" }
