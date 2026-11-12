param([int]$Port = 8780)
$ErrorActionPreference = 'Stop'
$projectDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) '01_Project'
$bookPython = Join-Path $projectDirectory '.venv-gpu\Scripts\python.exe'
$bookDirectory = Join-Path $projectDirectory '.venv\child-jamo-web-20260916\book-label-review-20261003'
$bookBundle = Get-FileHash -LiteralPath (Join-Path $bookDirectory 'review.json') -Algorithm SHA256
$bookUrl = "http://127.0.0.1:$Port"
$bookHealth = $null
try { $bookHealth = Invoke-RestMethod -Uri "$bookUrl/health" -TimeoutSec 2 } catch {}
if ($bookHealth) {
    if ($bookHealth.app -eq 'training-pronunciation-audit' -and $bookHealth.review_sha256 -eq $bookBundle.Hash.ToLowerInvariant()) {
        Write-Output "Book label review is already running: $bookUrl/"
        return
    }
    throw "Port $Port is used by another question set or application."
}
if (-not (Test-Path -LiteralPath $bookPython)) { throw "Python environment missing: $bookPython" }
$bookServer = Join-Path $projectDirectory 'python\tools\child_jamo_label_audit\server.py'
& $bookPython -B -X utf8 $bookServer --directory $bookDirectory --page (Join-Path $bookDirectory 'listen.html') --port $Port
exit $LASTEXITCODE
