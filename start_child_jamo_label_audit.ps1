param([int]$Port = 8777)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$auditPython = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
$auditUrl = "http://127.0.0.1:$Port"
$auditHealth = $null
try { $auditHealth = Invoke-RestMethod -Uri "$auditUrl/health" -TimeoutSec 2 } catch {}
if ($auditHealth) {
    if ($auditHealth.app -eq 'training-pronunciation-audit') {
        Write-Host "Listening review is already running: $auditUrl/"
        exit 0
    }
    throw "Port $Port is used by another application. No process was stopped."
}
if (-not (Test-Path -LiteralPath $auditPython)) { throw "Python environment missing: $auditPython" }
& $auditPython -X utf8 python/tools/child_jamo_label_audit/server.py --port $Port
exit $LASTEXITCODE
