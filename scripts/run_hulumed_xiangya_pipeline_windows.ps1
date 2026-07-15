param(
    [string]$RepoRoot = "G:\0-newResearch\temp\Derm-agent",
    [string]$PythonExe = "G:\Anaconda\envs\dermagent-xh\python.exe",
    [string]$RawDriveDir = "G:\0-newResearch\temp\xiangya_drive_raw"
)

$ErrorActionPreference = "Stop"

$logDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "hulumed_xiangya_pipeline_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

Write-Log "Pipeline started"

Write-Log "Step 1: Ensure CUDA PyTorch"
powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\install_pytorch_cuda_windows.ps1")

Write-Log "Step 2: Download Hulu-Med model"
powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\download_hulumed_model_fast_windows.ps1")

Write-Log "Step 3: Download Xiangya Google Drive folder"
powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "scripts\download_xiangya_drive_windows.ps1") -TargetDir $RawDriveDir

Write-Log "Step 4: Normalize Xiangya dataset layout"
& $PythonExe (Join-Path $RepoRoot "scripts\prepare_xiangya_sft_dataset.py") `
    --source-root $RawDriveDir `
    --target-root (Join-Path $RepoRoot "data\sft数据")

Write-Log "Step 5: Start Hulu-Med server in 4bit mode"
$env:LOAD_IN_4BIT = "1"
$env:CPU_MAX_MEMORY_GB = "32"
$env:GPU_MAX_MEMORY_GB = "5.0"
Start-Process powershell -ArgumentList @(
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $RepoRoot "scripts\start_hulumed_server_windows.ps1"),
    "-ProjectRoot",
    $RepoRoot
) -WindowStyle Hidden
Start-Sleep -Seconds 20

Write-Log "Step 6: Smoke run Xiangya + Hulu-Med"
& $PythonExe (Join-Path $RepoRoot "scripts\smoke_run_xiangya_hulumed.py") `
    --repo-root $RepoRoot `
    --data-root (Join-Path $RepoRoot "data\sft数据")

Write-Log "Pipeline finished"
