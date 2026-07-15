param(
    [string]$TargetDir = "G:\0-newResearch\models\Hulu-Med-4B",
    [string]$RepoId = "ZJU-AI4H/Hulu-Med-4B",
    [string]$PythonExe = "G:\Anaconda\envs\dermagent-xh\python.exe",
    [string]$Endpoint = "https://hf-mirror.com",
    [int]$MaxWorkers = 4
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
$env:HF_ENDPOINT = $Endpoint

Write-Host "Repo ID    : $RepoId"
Write-Host "Target dir : $TargetDir"
Write-Host "Endpoint   : $Endpoint"
Write-Host "Workers    : $MaxWorkers"
Write-Host ""

$code = @'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=r"__REPO_ID__",
    local_dir=r"__TARGET_DIR__",
    max_workers=__MAX_WORKERS__,
)
print(path)
'@

$code = $code.Replace("__REPO_ID__", $RepoId).Replace("__TARGET_DIR__", $TargetDir).Replace("__MAX_WORKERS__", [string]$MaxWorkers)
$tempPy = Join-Path ([System.IO.Path]::GetTempPath()) "download_hulumed_model_windows.py"
Set-Content -LiteralPath $tempPy -Value $code -Encoding UTF8
try {
    & $PythonExe $tempPy
}
finally {
    Remove-Item -LiteralPath $tempPy -Force -ErrorAction SilentlyContinue
}
