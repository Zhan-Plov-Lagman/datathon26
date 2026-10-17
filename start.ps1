param(
    [string]$WorkerHost = '127.0.0.1',
    [switch]$WithWorker,
    [string]$Python = '.venv/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Python)) {
    if ($Python -ne '.venv/Scripts/python.exe') { throw 'Selected Python environment does not exist' }
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python environment' }
}
& $Python -m pip install -r requirements-api.txt
if ($LASTEXITCODE -ne 0) { throw 'Cannot install API dependencies' }
& $Python -m scripts.setup_coordinator
if ($LASTEXITCODE -ne 0) { throw 'Cannot initialize local configuration' }
if ($WithWorker) {
    if ($WorkerHost -ne '127.0.0.1') { throw 'Single-laptop mode keeps the worker API on loopback' }
    & $Python -m scripts.run_local
} else {
    & $Python -m app.api --worker-host $WorkerHost
}
exit $LASTEXITCODE
