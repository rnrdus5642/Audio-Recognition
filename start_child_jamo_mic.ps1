param([int]$Port = 8776)
$ErrorActionPreference = 'Stop'
$micPython = Join-Path $PSScriptRoot '.venv-gpu\Scripts\python.exe'
$micScript = Join-Path $PSScriptRoot 'python\tools\child_jamo_web\mic_server.py'
$micUrl = "http://127.0.0.1:$Port/"
$micStatus = $null
try { $micStatus = Invoke-RestMethod -Uri ($micUrl + 'api/status') -TimeoutSec 3 } catch { }
if ($micStatus -and $micStatus.instance.app -eq 'child-jamo-microphone') {
    Write-Output "이미 실행 중입니다: $micUrl"
    return
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "포트 $Port 을 다른 프로그램이 사용 중입니다. -Port 로 다른 번호를 지정해 주세요."
}
Write-Output "마이크 테스트: $micUrl"
Write-Output '이 창을 열어 두세요. Ctrl+C 를 누르면 서버가 종료됩니다.'
& $micPython -u -X utf8 $micScript --port $Port
