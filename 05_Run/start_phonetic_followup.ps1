param([int]$Port = 8779)
$ErrorActionPreference = 'Stop'
$projectDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) '01_Project'
$followupPython = Join-Path $projectDirectory '.venv-gpu\Scripts\python.exe'
$followupDirectory = Join-Path $projectDirectory '.venv\child-jamo-web-20260916\phonetic-followup-20261003'
$followupBundle = Get-FileHash -LiteralPath (Join-Path $followupDirectory 'review.json') -Algorithm SHA256
$followupUrl = "http://127.0.0.1:$Port"
$followupHealth = $null
try { $followupHealth = Invoke-RestMethod -Uri "$followupUrl/health" -TimeoutSec 2 } catch {}
if ($followupHealth) {
    if ($followupHealth.app -eq 'training-pronunciation-audit' -and $followupHealth.review_sha256 -eq $followupBundle.Hash.ToLowerInvariant()) {
        Write-Output "Additional listening review is already running: $followupUrl/"
        return
    }
    throw "Port $Port is used by another question set or application."
}
if (-not (Test-Path -LiteralPath $followupPython)) { throw "Python environment missing: $followupPython" }
$followupServer = Join-Path $projectDirectory 'python\tools\child_jamo_label_audit\server.py'
& $followupPython -B -X utf8 $followupServer --directory $followupDirectory --page (Join-Path $followupDirectory 'listen.html') --port $Port
exit $LASTEXITCODE
