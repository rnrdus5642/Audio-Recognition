param([int]$Port = 8778)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$reviewPython = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
$reviewDirectory = Join-Path $PSScriptRoot '.venv\child-jamo-web-20260916\phonetic-labels-20260929'
$reviewBundle = Get-FileHash -LiteralPath (Join-Path $reviewDirectory 'review.json') -Algorithm SHA256
$reviewUrl = "http://127.0.0.1:$Port"
$reviewHealth = $null
try { $reviewHealth = Invoke-RestMethod -Uri "$reviewUrl/health" -TimeoutSec 2 } catch {}
if ($reviewHealth) {
    if ($reviewHealth.app -eq 'training-pronunciation-audit' -and $reviewHealth.review_sha256 -eq $reviewBundle.Hash.ToLowerInvariant()) {
        Write-Host "Pronunciation review is already running: $reviewUrl/"
        exit 0
    }
    throw "Port $Port is used by another question set or application."
}
if (-not (Test-Path -LiteralPath $reviewPython)) { throw "Python environment missing: $reviewPython" }
& $reviewPython -X utf8 python/tools/child_jamo_label_audit/server.py --directory $reviewDirectory --page python/tools/child_jamo_label_audit/phonetic.html --port $Port
exit $LASTEXITCODE
