param([int]$Port = 8781, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$reviewWorkspace = Split-Path -Parent $PSScriptRoot
$reviewPython = Join-Path $reviewWorkspace '01_Project\.venv-gpu\Scripts\python.exe'
$reviewServer = Join-Path $reviewWorkspace '01_Project\python\tools\child_audio_review\server.py'
$reviewState = Join-Path $reviewWorkspace '02_Datasets\검토\3세_청취'
$reviewUrl = "http://127.0.0.1:$Port/"
if (-not (Test-Path -LiteralPath $reviewPython)) { throw "Python environment missing: $reviewPython" }
$reviewHealth = $null
try { $reviewHealth = Invoke-RestMethod -Uri ($reviewUrl + 'health') -TimeoutSec 2 } catch {}
if ($reviewHealth -and $reviewHealth.app -ne 'age3-listening-room-v1') { throw "Port $Port belongs to another application." }
if (-not $reviewHealth) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
    New-Item -ItemType Directory -Path $reviewState -Force | Out-Null
    $reviewArguments = @('-B', '-X', 'utf8', '-u', ('"' + $reviewServer + '"'), '--port', $Port)
    $reviewProcess = Start-Process -FilePath $reviewPython -ArgumentList $reviewArguments -WorkingDirectory $reviewWorkspace -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $reviewState 'server.stdout.log') -RedirectStandardError (Join-Path $reviewState 'server.stderr.log')
    for ($reviewTry = 0; $reviewTry -lt 180; $reviewTry++) {
        Start-Sleep -Seconds 1
        if ($reviewProcess.HasExited) { throw "Server exited. Read logs in: $reviewState" }
        try { $reviewHealth = Invoke-RestMethod -Uri ($reviewUrl + 'health') -TimeoutSec 2 } catch {}
        if ($reviewHealth -and $reviewHealth.app -eq 'age3-listening-room-v1') { break }
    }
    if (-not $reviewHealth) { throw "Server did not become ready. Read logs in: $reviewState" }
}
Write-Output "Ready: $reviewUrl (PID $($reviewHealth.pid))"
if (-not $NoBrowser) { Start-Process $reviewUrl }
