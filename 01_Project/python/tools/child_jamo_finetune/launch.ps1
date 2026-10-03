<#
    Start (or resume) the long fine-tuning run detached from this shell.

    The run takes days. Start-Process leaves it owned by the session's
    service host rather than the console that launched it, so closing the
    terminal - or the CLI that typed the command - does not kill it.

    Two writers to one run directory would interleave checkpoints and
    leave neither usable, so this refuses to start while another live
    process holds the run, and train.py holds the same lock itself. The
    launcher does not report success until the process is alive and the
    log has actually advanced.

        powershell -File python\tools\child_jamo_finetune\launch.ps1
        powershell -File python\tools\child_jamo_finetune\launch.ps1 -Resume

    Progress:  .venv\hubert-child-20260914\<name>\progress.json
    Log:       .venv\hubert-child-20260914\run\<name>.log
#>
param(
    [switch]$Resume,
    [string]$Name = 'main',
    [int]$OffloadLayers = 26,
    [int]$BatchBudget = 640000,
    [int]$Accumulate = 8,
    [string[]]$Extra = @()
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$python = Join-Path $repo '.venv-gpu\Scripts\python.exe'
$script = Join-Path $repo 'python\tools\child_jamo_finetune\train.py'
$root = Join-Path $repo '.venv\hubert-child-20260914'

if ($Name -notmatch '^[A-Za-z0-9._-]+$') {
    throw "run name '$Name' must be a plain identifier"
}
$work = Join-Path $root $Name
# Everything this run writes has to stay inside the work root.
$resolvedRoot = (Resolve-Path $root).Path
New-Item -ItemType Directory -Force -Path $work | Out-Null
if (-not (Resolve-Path $work).Path.StartsWith($resolvedRoot)) {
    throw "run directory $work escapes $resolvedRoot"
}

# Informative only. train.py takes a real OS byte-range lock on run.lock
# and holds it for its lifetime, which is what actually prevents a second
# writer; this just gives a clearer message before spending a model load.
$owner = Join-Path $work 'run.owner.json'
if (Test-Path $owner) {
    $held = Get-Content $owner -Raw | ConvertFrom-Json
    if (Get-Process -Id $held.pid -ErrorAction SilentlyContinue) {
        throw "run '$Name' appears to be owned by pid $($held.pid) (started $($held.started))"
    }
    Write-Output "previous owner pid $($held.pid) is gone; train.py will re-take the lock"
}

$arguments = @($script, '--name', $Name, '--offload-layers', $OffloadLayers,
               '--batch-budget', $BatchBudget, '--accumulate', $Accumulate)
if ($Resume) { $arguments += '--resume' }
$arguments += $Extra

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# The fixture suite points this at a ~1M-parameter stand-in encoder. A
# leaked value would silently train the wrong model for days, so it is
# cleared here and the real weight hash is asserted below.
Remove-Item Env:\CHILD_JAMO_ENCODER -ErrorAction SilentlyContinue

$expected = '14977266907588c28da9403bc4c04478a29b49968dfe206fb9cb8e71e4d2e31d'
$weights = Join-Path $repo '.venv\encoder-screen-0914\models\hubertko1b\model.safetensors'
$actual = (Get-FileHash -Path $weights -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) {
    throw "encoder weights are not the verified original: expected $expected, found $actual"
}
Write-Output "encoder verified: $actual"

$log = Join-Path $root "run\$Name.log"
$before = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $work "stdout-$stamp.log") `
    -RedirectStandardError  (Join-Path $work "stderr-$stamp.log")

# Loading a 3.85 GB encoder and the manifests takes a while before the
# first log line, so give it a real chance before judging.
$deadline = (Get-Date).AddSeconds(180)
$advanced = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ($process.HasExited) { break }
    $now = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }
    if ($now -gt $before) { $advanced = $true; break }
}

if ($process.HasExited) {
    $err = Get-Content (Join-Path $work "stderr-$stamp.log") -Tail 20 -ErrorAction SilentlyContinue
    throw "training exited immediately (code $($process.ExitCode)):`n$($err -join "`n")"
}
if (-not $advanced) {
    Write-Warning "pid $($process.Id) is alive but $log has not advanced yet"
} else {
    Write-Output "pid $($process.Id) is running and the log is advancing"
}

@{ pid = $process.Id; name = $Name; started = $stamp;
   arguments = ($arguments -join ' ') } |
    ConvertTo-Json | Set-Content (Join-Path $work 'launch.json') -Encoding utf8
Write-Output "progress $work\progress.json"
Write-Output "log      $log"
