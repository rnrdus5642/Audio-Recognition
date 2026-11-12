[CmdletBinding()]
param(
    [string]$ThreadId,
    [switch]$Status,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
if ([System.IO.Path]::GetPathRoot($projectRoot) -ne 'C:\') { throw 'Use the C-drive project.' }
$runRoot = Join-Path $projectRoot '04_Models_and_Experiments\orthographic-jamo-20261027\main'
$reportRoot = Join-Path $runRoot 'reporting'
$configPath = Join-Path $reportRoot 'config.json'
$watcherPath = Join-Path $projectRoot '01_Project\python\tools\child_training_watch\watch.py'
$pythonPath = Join-Path $projectRoot '01_Project\.venv-gpu\Scripts\python.exe'
$active = $null
$ownerPath = Join-Path $reportRoot 'owner.json'
if (Test-Path -LiteralPath $ownerPath) {
    $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.pid)" -ErrorAction SilentlyContinue
    if ($candidate -and $candidate.CommandLine -and $candidate.CommandLine.Contains($watcherPath)) { $active = $candidate }
}
if ($Status) {
    if ($active) { Write-Output "Result watcher: PID $($active.ProcessId)" }
    else { Write-Output 'No active result watcher.' }
    $statusPath = Join-Path $reportRoot 'status.json'
    if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 }
    return
}
if ($Stop) {
    if ($active) { [System.IO.File]::WriteAllText((Join-Path $reportRoot 'stop.request'), (Get-Date -Format o)) }
    Write-Output 'Only the report watcher will stop. Training continues.'
    return
}
if ($active) { Write-Output "Result watcher already running: PID $($active.ProcessId)"; return }
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $configPath)) {
    if (-not $ThreadId) { $ThreadId = $env:CODEX_THREAD_ID }
    if ($ThreadId -notmatch '^[a-fA-F0-9-]{36}$') { throw 'A valid target Codex thread UUID is required.' }
    if (-not $env:CODEX_HOME) { throw 'Run initial setup from the authorized Codex session.' }
    $codexPath = Join-Path $env:LOCALAPPDATA 'Programs\OpenAI\Codex\bin\codex.exe'
    if (-not (Test-Path -LiteralPath $codexPath)) { throw 'Codex CLI was not found.' }
    [pscustomobject]@{
        project_root = $projectRoot
        run_dir = $runRoot
        thread_id = $ThreadId
        codex_home = $env:CODEX_HOME
        codex_exe = $codexPath
        poll_seconds = 30
        created = Get-Date -Format o
    } | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding UTF8
} elseif ($ThreadId) {
    $saved = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($saved.thread_id -ne $ThreadId) { throw 'Do not redirect an existing report subscription to a different conversation.' }
}
$stopPath = Join-Path $reportRoot 'stop.request'
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6)
$arguments = @('-B','-X','utf8','-u',('"' + $watcherPath + '"'),'--config',('"' + $configPath + '"'))
$watcherProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $reportRoot "$stamp.out.log") -RedirectStandardError (Join-Path $reportRoot "$stamp.err.log") -PassThru
Write-Output "Started result watcher launcher PID $($watcherProcess.Id)."
Write-Output "Reporting state: $reportRoot"
