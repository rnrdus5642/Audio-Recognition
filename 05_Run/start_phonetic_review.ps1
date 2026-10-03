$ErrorActionPreference = 'Stop'
$projectDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) '01_Project'
& (Join-Path $projectDirectory 'start_child_jamo_phonetic_review.ps1') @args
