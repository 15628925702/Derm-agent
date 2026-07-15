param(
    [string]$PythonExe = "G:\Anaconda\envs\dermagent-xh\python.exe",
    [string]$WheelDir = "G:\0-newResearch\temp\wheels",
    [string]$MirrorBase = "https://mirrors.aliyun.com/pytorch-wheels/cu126"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

$wheels = @(
    "torch-2.7.0+cu126-cp310-cp310-win_amd64.whl",
    "torchvision-0.22.0+cu126-cp310-cp310-win_amd64.whl",
    "torchaudio-2.7.0+cu126-cp310-cp310-win_amd64.whl"
)

foreach ($wheel in $wheels) {
    $target = Join-Path $WheelDir $wheel
    $url = "$MirrorBase/" + $wheel.Replace("+", "%2B")
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $wheel"
        & "C:\Windows\System32\curl.exe" -L -C - --retry 5 --retry-delay 5 $url -o $target
    }
}

Write-Host ""
Write-Host "Installing local CUDA wheels..."
& $PythonExe -m pip install --force-reinstall `
    (Join-Path $WheelDir "torch-2.7.0+cu126-cp310-cp310-win_amd64.whl") `
    (Join-Path $WheelDir "torchvision-0.22.0+cu126-cp310-cp310-win_amd64.whl") `
    (Join-Path $WheelDir "torchaudio-2.7.0+cu126-cp310-cp310-win_amd64.whl")

Write-Host ""
Write-Host "Verifying CUDA torch..."
& $PythonExe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('cuda_version', torch.version.cuda)"
