param([string]$WorkerHost = '127.0.0.1')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python environment' }
}
& ./.venv/Scripts/python.exe -m pip install -r requirements-api.txt
if ($LASTEXITCODE -ne 0) { throw 'Cannot install API dependencies' }
& ./.venv/Scripts/python.exe -m scripts.setup_coordinator
if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize local configuration' }
& ./.venv/Scripts/python.exe -m app.api --worker-host $WorkerHost
exit $LASTEXITCODE
