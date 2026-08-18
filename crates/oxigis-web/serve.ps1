#!/usr/bin/env pwsh
<#
.SYNOPSIS
    OxiGIS web shell — build the wasm bundle and serve it locally.

.DESCRIPTION
    Windows PowerShell counterpart of serve.sh.

.PARAMETER Port
    Local port to serve on. Defaults to 8080.

.EXAMPLE
    .\crates\oxigis-web\serve.ps1            # dev profile, port 8080
    .\crates\oxigis-web\serve.ps1 -Port 9000 # dev profile, port 9000
    $env:OXIGIS_WASM_PROFILE = "wasm-release"; .\crates\oxigis-web\serve.ps1

.NOTES
    Serving on localhost is deliberate: WebGPU is only exposed in a secure
    context, and `http://localhost` counts as one while a LAN IP over plain
    HTTP does not (that would silently drop the app to the WebGL2 fallback).
#>
param(
    [Parameter(Position = 0)]
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$crateDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $crateDir "..\..")).Path

# `dev` keeps the build fast; `wasm-release` is the size-optimised workspace
# profile (opt-level=s, fat LTO) used for anything shipped.
$profile = $env:OXIGIS_WASM_PROFILE
if ([string]::IsNullOrEmpty($profile)) {
    $profile = "dev"
}

if (-not (Get-Command wasm-pack -ErrorAction SilentlyContinue)) {
    Write-Error "serve.ps1: ``wasm-pack`` not found in PATH.`n  install it with: cargo install wasm-pack"
    exit 1
}

# Prefer `python`, which is what a standard Windows install provides;
# fall back to `python3` for environments that only expose that name.
$pythonCmd = $null
foreach ($candidate in @("python", "python3")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $pythonCmd = $candidate
        break
    }
}
if ($null -eq $pythonCmd) {
    Write-Error "serve.ps1: ``python`` not found in PATH.`n  install python, or serve $crateDir with any static file server"
    exit 1
}

Write-Host "==> wasm-pack build (profile: $profile)"
switch ($profile) {
    "dev"     { $profileArgs = @("--dev") }
    "release" { $profileArgs = @("--release") }
    "profiling" { $profileArgs = @("--profiling") }
    default   { $profileArgs = @("--profile", $profile) }
}

# Run from the repo root so .cargo/config.toml (getrandom wasm_js backend) and
# the workspace lockfile apply. Output lands in crates/oxigis-web/pkg/, which is
# exactly what index.html imports.
Push-Location $repoRoot
try {
    & wasm-pack build "crates/oxigis-web" --target web @profileArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "==> serving $crateDir at http://localhost:$Port/"
Write-Host "    (Ctrl-C to stop; reload the page after a rebuild)"
& $pythonCmd -m http.server $Port --bind 127.0.0.1 --directory $crateDir
