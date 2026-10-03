<#
    아이 음성 인식 확인 뷰어를 백그라운드로 띄웁니다.

    학습된 모델(best.pt)을 불러와 로컬에서만 동작하는 웹 페이지를 엽니다.
    창 없이 실행되므로 이 터미널을 닫아도 서버는 계속 살아 있습니다.

        powershell -File start_child_jamo_web.ps1
        powershell -File start_child_jamo_web.ps1 -OpenBrowser
        powershell -File start_child_jamo_web.ps1 -Stop

    주소 : http://127.0.0.1:8768
    로그 : .venv\child-jamo-web-20260916\server.log

    소유권: 서버가 스스로 server.pid.json 에 자기 PID 와 1회용 토큰을 적고,
    /api/status 로 같은 토큰을 돌려줍니다. -Stop 은 그 토큰이 일치할 때만
    그 PID 를 종료하므로, 재부팅 후 PID 가 재사용되었거나 다른 프로그램이
    같은 포트를 쓰고 있어도 남의 프로세스를 죽이지 않습니다.
#>
param(
    [int]$Port = 8768,
    [switch]$OpenBrowser,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$python = Join-Path $repo '.venv-gpu-repaired311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $repo '.venv-gpu\Scripts\python.exe'
}
$script = Join-Path $repo 'python\tools\child_jamo_web\server.py'
$work = Join-Path $repo '.venv\child-jamo-web-20260916'
$pidFile = Join-Path $work 'server.pid.json'
New-Item -ItemType Directory -Force -Path $work | Out-Null

function Read-Record {
    if (-not (Test-Path $pidFile)) { return $null }
    try { return Get-Content $pidFile -Raw -Encoding utf8 | ConvertFrom-Json }
    catch { return $null }
}

# 기록된 인스턴스가 "지금 그 인스턴스"인지 확인합니다. 살아 있는 PID 만으로는
# 부족합니다. 토큰까지 같아야 우리 것입니다.
function Confirm-Owned($record) {
    if (-not $record) { return $null }
    if (-not (Get-Process -Id $record.pid -ErrorAction SilentlyContinue)) { return $null }
    try {
        $probe = Invoke-RestMethod -Uri ($record.url + 'api/status') -TimeoutSec 5
    } catch { return $null }
    if ($probe.instance.token -ne $record.token) { return $null }
    if ($probe.instance.pid -ne $record.pid) { return $null }
    return $probe
}

if ($Stop) {
    $record = Read-Record
    $probe = Confirm-Owned $record
    if (-not $probe) {
        Write-Output '이 스크립트가 띄운 뷰어가 실행 중이 아닙니다. (다른 프로세스는 건드리지 않습니다)'
        return
    }
    Stop-Process -Id $record.pid -Force
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 400
        if (-not (Get-Process -Id $record.pid -ErrorAction SilentlyContinue)) { break }
    }
    $still = Get-NetTCPConnection -LocalPort $record.port -State Listen -ErrorAction SilentlyContinue
    if (Get-Process -Id $record.pid -ErrorAction SilentlyContinue) {
        throw "pid $($record.pid) 이(가) 아직 살아 있습니다."
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    if ($still) {
        Write-Warning "포트 $($record.port) 에 다른 프로그램이 남아 있습니다. 건드리지 않았습니다."
    } else {
        Write-Output "중지했습니다 (pid $($record.pid))."
    }
    return
}

$existing = Confirm-Owned (Read-Record)
if ($existing) {
    $record = Read-Record
    Write-Output "이미 실행 중입니다: pid $($record.pid), $($record.url)"
    if ($OpenBrowser) { Start-Process $record.url }
    return
}

$inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($inUse) { throw "포트 $Port 을(를) 이미 다른 프로그램이 쓰고 있습니다." }

$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

Remove-Item $pidFile -ErrorAction SilentlyContinue
$stdout = Join-Path $work 'server.log'
$stderr = Join-Path $work 'server.err.log'
$process = Start-Process -FilePath $python `
    -ArgumentList @($script, '--port', $Port, '--warm') `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

# 모델이 3.85GB 라 첫 기동에 시간이 걸립니다. 실제로 /api/status 가
# 준비 완료로 응답할 때까지 기다립니다.
$url = "http://127.0.0.1:$Port/"
$deadline = (Get-Date).AddSeconds(300)
$ready = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if ($process.HasExited) { break }
    $ready = Confirm-Owned (Read-Record)
    if ($ready -and $ready.model.ready) { break }
    $ready = $null
}

if ($process.HasExited) {
    $tail = Get-Content $stderr -Tail 25 -ErrorAction SilentlyContinue
    throw "서버가 바로 종료되었습니다 (code $($process.ExitCode)):`n$($tail -join "`n")"
}

$record = Read-Record
if ($ready) {
    Write-Output "준비 완료 - pid $($record.pid), $($record.url)"
} else {
    Write-Warning "서버는 떴지만 모델이 아직 준비되지 않았습니다. 로그: $stdout"
}
if ($OpenBrowser) { Start-Process $url }
Write-Output "로그: $stdout"
