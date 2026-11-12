$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'start_legacy_stream.ps1') @args
