param(
    [string]$ProjectRoot = $(Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"

$condaEnvName = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "dermagent-xh" }
$hostAddress = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$port = if ($env:PORT) { $env:PORT } else { "8013" }
$apiKey = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "EMPTY" }
$device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }
$dtype = if ($env:DTYPE) { $env:DTYPE } else { "float16" }
$servedModelName = if ($env:SERVED_MODEL_NAME) { $env:SERVED_MODEL_NAME } else { "Hulu-Med-4B" }
$deviceMap = if ($env:DEVICE_MAP) { $env:DEVICE_MAP } else { "auto" }
$gpuMaxMemoryGb = if ($env:GPU_MAX_MEMORY_GB) { $env:GPU_MAX_MEMORY_GB } else { "5.0" }
$cpuMaxMemoryGb = if ($env:CPU_MAX_MEMORY_GB) { $env:CPU_MAX_MEMORY_GB } else { "32" }
$offloadFolder = if ($env:OFFLOAD_FOLDER) { $env:OFFLOAD_FOLDER } else { (Join-Path $ProjectRoot ".offload\hulumed") }
$maxNewTokens = if ($env:MAX_NEW_TOKENS_DEFAULT) { $env:MAX_NEW_TOKENS_DEFAULT } else { "256" }
$loadIn4Bit = if ($env:LOAD_IN_4BIT) { $env:LOAD_IN_4BIT } else { "0" }
$loadIn8Bit = if ($env:LOAD_IN_8BIT) { $env:LOAD_IN_8BIT } else { "0" }

$workspaceRoot = Split-Path -Parent $ProjectRoot
$preferredModelPaths = @(
    $(if ($env:DERMAGENT_MODELS_ROOT) { Join-Path $env:DERMAGENT_MODELS_ROOT "Hulu-Med-4B" } else { $null }),
    $(if ($env:DERMAGENT_MODELS_ROOT) { Join-Path $env:DERMAGENT_MODELS_ROOT "Hulu-Med-7B" } else { $null }),
    (Join-Path $workspaceRoot "models\Hulu-Med-4B"),
    (Join-Path $workspaceRoot "models\Hulu-Med-7B"),
    "G:\0-newResearch\models\Hulu-Med-4B",
    "G:\0-newResearch\models\Hulu-Med-7B"
) | Where-Object { $_ }

$modelPath = $env:MODEL_PATH
if (-not $modelPath) {
    $modelPath = $preferredModelPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $modelPath) {
    throw "Hulu-Med model directory not found. Checked: $($preferredModelPaths -join '; ')"
}

$pythonBin = Join-Path "G:\Anaconda\envs\$condaEnvName" "python.exe"
if (-not (Test-Path $pythonBin)) {
    throw "Python not found for conda env '$condaEnvName': $pythonBin"
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null
New-Item -ItemType Directory -Force -Path $offloadFolder | Out-Null

$launchArgs = @(
    (Join-Path $ProjectRoot "scripts\serve_transformers_openai.py"),
    "--model-path", $modelPath,
    "--served-model-name", $servedModelName,
    "--backend", "hulumed",
    "--host", $hostAddress,
    "--port", $port,
    "--api-key", $apiKey,
    "--device", $device,
    "--dtype", $dtype,
    "--max-new-tokens-default", $maxNewTokens,
    "--device-map", $deviceMap,
    "--gpu-max-memory-gb", $gpuMaxMemoryGb,
    "--cpu-max-memory-gb", $cpuMaxMemoryGb,
    "--offload-folder", $offloadFolder
)

if ($loadIn4Bit -eq "1") {
    $launchArgs += "--load-in-4bit"
}
if ($loadIn8Bit -eq "1") {
    $launchArgs += "--load-in-8bit"
}

Write-Host "Project root   : $ProjectRoot"
Write-Host "Model path     : $modelPath"
Write-Host "Python         : $pythonBin"
Write-Host "Host/port      : $hostAddress`:$port"
Write-Host "Device         : $device"
Write-Host "Dtype          : $dtype"
Write-Host "Device map     : $deviceMap"
Write-Host "GPU max mem GB : $gpuMaxMemoryGb"
Write-Host "CPU max mem GB : $cpuMaxMemoryGb"
Write-Host "Offload folder : $offloadFolder"
Write-Host "Load in 4bit   : $loadIn4Bit"
Write-Host "Load in 8bit   : $loadIn8Bit"
Write-Host ""

& $pythonBin @launchArgs
