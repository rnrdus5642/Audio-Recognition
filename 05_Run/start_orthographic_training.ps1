[CmdletBinding()]
param(
    [switch]$Resume,
    [switch]$Status,
    [switch]$Pause
)
$ErrorActionPreference = 'Stop'
if (([int]$Resume.IsPresent + [int]$Status.IsPresent + [int]$Pause.IsPresent) -gt 1) {
    throw 'Use only one of -Resume, -Status or -Pause.'
}
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
if ([System.IO.Path]::GetPathRoot($projectRoot) -ne 'C:\') {
    throw 'This launcher is restricted to the C-drive project.'
}
$trainingRoot = Join-Path $projectRoot '04_Models_and_Experiments\orthographic-jamo-20261027'
$runRoot = Join-Path $trainingRoot 'main'
$progressPath = Join-Path $runRoot 'progress.json'
$ownerPath = Join-Path $runRoot 'owner.json'
$trainerPath = Join-Path $projectRoot '01_Project\python\tools\child_orthographic_finetune\train.py'
$pythonPath = Join-Path $projectRoot '01_Project\.venv-gpu\Scripts\python.exe'
$pausePath = Join-Path $runRoot 'pause.request'
$active = $null
if (Test-Path -LiteralPath $ownerPath) {
    $owner = Get-Content -LiteralPath $ownerPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $($owner.pid)" -ErrorAction SilentlyContinue
    if ($candidate -and $candidate.CommandLine -and $candidate.CommandLine.Contains($trainerPath)) {
        $active = $candidate
    }
}
if ($Status) {
    if ($active) { Write-Output "Training process: $($active.ProcessId)" }
    else { Write-Output 'No active main training process.' }
    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -LiteralPath $progressPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $progress | Select-Object updated,status,stage,pid,total_updates,head_updates,joint_updates,train_hours,completed_equivalents,elapsed_seconds,last_update | Format-List
    }
    return
}
if ($Pause) {
    if (-not $active) { throw 'No active main training process to pause.' }
    [System.IO.File]::WriteAllText($pausePath, (Get-Date -Format o), [System.Text.UTF8Encoding]::new($false))
    Write-Output 'Pause requested. The current update is committed, or the current evaluation cursor is saved.'
    return
}
if ($active) { throw "Training is already running (PID $($active.ProcessId)). Use -Status." }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'The project GPU Python environment is missing.' }
$proofPath = Join-Path $trainingRoot 'preflight.json'
if (-not (Test-Path -LiteralPath $proofPath)) { throw 'Complete preflight.py before launching training.' }
$proof = Get-Content -LiteralPath $proofPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($proof.status -ne 'passed') { throw 'The GPU preflight has not passed.' }
$checkpointRoot = Join-Path $runRoot 'checkpoints'
$receipts = @(Get-ChildItem -LiteralPath $checkpointRoot -Filter 'ckpt-*.pt.json' -File -ErrorAction SilentlyContinue)
if ($receipts.Count -and -not $Resume) { throw 'This run has checkpoints. Use -Resume to continue without restarting.' }
if ($Resume -and -not $receipts.Count) { throw 'There is no committed checkpoint to resume.' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
if ($Resume -and (Test-Path -LiteralPath $pausePath)) {
    Remove-Item -LiteralPath $pausePath
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6)
$stdoutPath = Join-Path $runRoot "launcher-$stamp.out.log"
$stderrPath = Join-Path $runRoot "launcher-$stamp.err.log"
$arguments = @('-B','-X','utf8','-u',('"' + $trainerPath + '"'),'--name','main')
if ($Resume) { $arguments += '--resume' }
$trainingProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
[pscustomobject]@{
    launcher_pid = $trainingProcess.Id
    started = Get-Date -Format o
    resumed = $Resume.IsPresent
    stdout = $stdoutPath
    stderr = $stderrPath
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runRoot 'launcher.json') -Encoding UTF8
Write-Output "Started hidden training launcher PID $($trainingProcess.Id)."
Write-Output "Progress: $progressPath"
Write-Output "Logs: $stdoutPath"
Write-Output 'Use this script with -Status to verify progress; -Resume restores the latest verified checkpoint after interruption.'
$reportConfig = Join-Path $runRoot 'reporting\config.json'
if (Test-Path -LiteralPath $reportConfig) {
    try { & (Join-Path $PSScriptRoot 'start_training_reports.ps1') }
    catch { Write-Warning "Training started, but the result watcher needs attention: $_" }
}
