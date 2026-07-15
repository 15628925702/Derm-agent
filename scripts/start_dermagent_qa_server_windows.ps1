param(
    [string]$ProjectRoot = $(Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = "G:\Anaconda\envs\dermagent-xh\python.exe",
    [string]$ListenHost = "127.0.0.1",
    [int]$Port = 8015,
    [string]$ApiKey = "EMPTY"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$scriptPath = Join-Path $ProjectRoot "scripts\serve_dermagent_qa.py"
if (-not (Test-Path $scriptPath)) {
    throw "QA server script not found: $scriptPath"
}

$logsDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$outLog = Join-Path $logsDir "dermagent_qa_server.out.log"
$errLog = Join-Path $logsDir "dermagent_qa_server.err.log"

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*$scriptPath*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$proc = Start-Process -FilePath $PythonExe -ArgumentList @($scriptPath, '--host', $ListenHost, '--port', [string]$Port, '--api-key', $ApiKey) -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$proc.Id

