param([ValidateRange(1, 65535)][int]$Port = 7860)

$ErrorActionPreference = 'Stop'
$streamProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\01_Project'))
$streamPython = Join-Path $streamProject '.venv-gpu\Scripts\python.exe'
$streamModel = Join-Path $streamProject 'python\tools\child_finetune\runs\child\best'
$streamMatrix = Join-Path $streamProject 'shared\confusion_matrices\ko_child_v2.json'
$streamUrl = "http://127.0.0.1:$Port/"

foreach ($requiredFile in @($streamPython, (Join-Path $streamModel 'model.safetensors'), $streamMatrix)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required local file is missing: $requiredFile"
    }
}

# Do not silently reuse another model already serving on this port.
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Open $streamUrl or stop that server before restarting. Use -Port to choose another port."
}

Write-Host "Child syllable recognition: $streamUrl"
Write-Host 'Enter the target word, press Prepare, then start the microphone.'
Write-Host 'Keep this window open. Press Ctrl+C to stop the server.'

Push-Location -LiteralPath $streamProject
try {
    & $streamPython -u -X utf8 -m python.tools.web_test --port $Port --model $streamModel --matrix $streamMatrix
    if ($LASTEXITCODE -ne 0) {
        throw "Recognition server exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
