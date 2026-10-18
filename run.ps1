param(
    [ValidateSet('Check', 'Interface', 'Service', 'Tests')]
    [string]$Mode = 'Check'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localFile = Join-Path $PSScriptRoot 'local-paths.json'
if (-not (Test-Path -LiteralPath $localFile)) {
    throw 'Create local-paths.json with python and qwen2b paths; see docs/UNIFIED_SETUP.md.'
}
$localConfig = Get-Content -LiteralPath $localFile -Raw -Encoding UTF8 | ConvertFrom-Json
$runtimePython = [string]$localConfig.python
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Configured Python does not exist.' }
$env:PYTHONPATH = (Join-Path $PSScriptRoot '.runtime-packages') + [IO.Path]::PathSeparator + $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:FORTE_MODEL_DIR = [string]$localConfig.qwen2b
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.cache/yolo'
switch ($Mode) {
    'Check' { & $runtimePython -m scripts.check_unified }
    'Interface' { & $runtimePython -m app.api }
    'Service' { & $runtimePython -m scripts.run_local }
    'Tests' {
        $env:FORTE_PEER_SNAPSHOT = $PSScriptRoot
        & $runtimePython -m pytest -q --junitxml=artifacts/integration-tests.xml
    }
}
exit $LASTEXITCODE
