$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultProjectRoot = Split-Path -Parent $ScriptDir
if ($args.Count -gt 0 -and $args[0]) {
    $ProjectRoot = $args[0]
} else {
    $ProjectRoot = $DefaultProjectRoot
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "outputs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "state") | Out-Null

$CondaEnvName = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "derm-qwen" }
$ModelRoots = @("/models", "/root/models")
$PreferredModelPaths = @(
    "/models/Qwen2.5-VL-7B-Instruct",
    "/root/models/Qwen2.5-VL-7B-Instruct"
)

if ($env:MODEL_PATH) {
    $ModelPath = $env:MODEL_PATH
} else {
    $ModelPath = $PreferredModelPaths[0]
}

$HostValue = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$PortValue = if ($env:PORT) { $env:PORT } else { "8000" }
$ApiKey = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "EMPTY" }
$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { Join-Path $ProjectRoot "logs" }
$LogFile = if ($env:LOG_FILE) { $env:LOG_FILE } else { Join-Path $LogDir "qwen_server.log" }
$PidFile = if ($env:PID_FILE) { $env:PID_FILE } else { Join-Path $ProjectRoot "state/qwen_server.pid" }
$ForceRestart = if ($env:FORCE_RESTART) { $env:FORCE_RESTART } else { "0" }
$GpuMemoryUtilization = if ($env:GPU_MEMORY_UTILIZATION) { $env:GPU_MEMORY_UTILIZATION } else { "0.92" }
$MaxModelLen = if ($env:MAX_MODEL_LEN) { $env:MAX_MODEL_LEN } else { "24576" }
$MaxNumSeqs = if ($env:MAX_NUM_SEQS) { $env:MAX_NUM_SEQS } elseif ([int]$MaxModelLen -gt 12288) { "1" } else { "2" }
$CpuOffloadGb = if ($env:CPU_OFFLOAD_GB) { $env:CPU_OFFLOAD_GB } else { "0" }
$EnforceEager = if ($env:ENFORCE_EAGER) { $env:ENFORCE_EAGER } else { "0" }
$CompilationConfig = if ($env:COMPILATION_CONFIG) { $env:COMPILATION_CONFIG } else { '{"mode":0,"cudagraph_mode":0}' }
$WaitSeconds = if ($env:WAIT_SECONDS) { [int]$env:WAIT_SECONDS } else { 300 }
$CheckIntervalSeconds = if ($env:CHECK_INTERVAL_SECONDS) { [int]$env:CHECK_INTERVAL_SECONDS } else { 5 }

if (-not (Test-Path $LogDir -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

function Resolve-ModelPath {
    param (
        [string[]]$PreferredPaths,
        [string]$CurrentPath,
        [string[]]$SearchRoots
    )

    if (Test-Path $CurrentPath -PathType Container) {
        return $CurrentPath
    }

    foreach ($PreferredPath in $PreferredPaths) {
        if (Test-Path $PreferredPath -PathType Container) {
            return $PreferredPath
        }
    }

    foreach ($SearchRoot in $SearchRoots) {
        if (Test-Path $SearchRoot -PathType Container) {
            $Candidate = Get-ChildItem -Path $SearchRoot -Directory -Filter "Qwen*" |
                Sort-Object FullName |
                Select-Object -First 1
            if ($Candidate) {
                return $Candidate.FullName
            }
        }
    }

    return $CurrentPath
}

function Resolve-VllmBin {
    if (Get-Command vllm -ErrorAction SilentlyContinue) {
        return "vllm"
    }

    $EnvBin = "/root/miniconda3/envs/$CondaEnvName/bin/vllm"
    if (Test-Path $EnvBin -PathType Leaf) {
        return $EnvBin
    }

    return $null
}

function Test-ServiceReady {
    param (
        [string]$Url,
        [string]$BearerToken
    )

    try {
        $Headers = @{ Authorization = "Bearer $BearerToken" }
        $Response = Invoke-WebRequest -Uri $Url -Headers $Headers -UseBasicParsing -TimeoutSec 10
        return $Response.Content
    } catch {
        return $null
    }
}

$ResolvedModelPath = Resolve-ModelPath -PreferredPaths $PreferredModelPaths -CurrentPath $ModelPath -SearchRoots $ModelRoots
if (-not (Test-Path $ResolvedModelPath -PathType Container)) {
    Write-Host "[error] model directory was not found."
    Write-Host "[error] preferred paths: $($PreferredModelPaths -join ', ')"
    Write-Host "[error] current MODEL_PATH: $ModelPath"
    exit 1
}

$VllmBin = Resolve-VllmBin
if (-not $VllmBin) {
    Write-Host "[error] vllm executable not found."
    Write-Host "[error] activate conda env '$CondaEnvName' or install vllm in it."
    exit 1
}

if ($env:SERVED_MODEL_NAME) {
    $ServedModelName = $env:SERVED_MODEL_NAME
} else {
    $ServedModelName = Split-Path $ResolvedModelPath -Leaf
}

$ServiceUrl = "http://$HostValue`:$PortValue/v1/models"
$ExistingPayload = Test-ServiceReady -Url $ServiceUrl -BearerToken $ApiKey
if ($ExistingPayload -and $ForceRestart -ne "1") {
    Write-Host "[ok] Qwen service is already ready."
    Write-Host $ExistingPayload
    exit 0
}

$ExistingProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*vllm serve*" -and $_.CommandLine -like "*--port $PortValue*"
}
if ($ExistingProcesses) {
    Write-Host "[warn] found stale vLLM process(es): $($ExistingProcesses.ProcessId -join ', ')"
    $ExistingProcesses | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}

Write-Host "Project root      : $ProjectRoot"
Write-Host "Model path        : $ResolvedModelPath"
Write-Host "Served model name : $ServedModelName"
Write-Host "Base URL          : http://$HostValue`:$PortValue/v1"
Write-Host "API key           : $ApiKey"
Write-Host "Log file          : $LogFile"
Write-Host "PID file          : $PidFile"
Write-Host "Force restart     : $ForceRestart"
Write-Host "GPU mem util      : $GpuMemoryUtilization"
Write-Host "Max model len     : $MaxModelLen"
Write-Host "Max num seqs      : $MaxNumSeqs"
Write-Host "MM limit          : image=1 video=0"
Write-Host "CPU offload (GB)  : $CpuOffloadGb"
Write-Host "Enforce eager     : $EnforceEager"
Write-Host "Compilation config: $CompilationConfig"

$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "expandable_segments:True" }

$ArgumentList = @(
    "serve"
    $ResolvedModelPath
    "--host", $HostValue
    "--port", $PortValue
    "--api-key", $ApiKey
    "--served-model-name", $ServedModelName
    "--gpu-memory-utilization", $GpuMemoryUtilization
    "--max-model-len", $MaxModelLen
    "--max-num-seqs", $MaxNumSeqs
    "--compilation-config", $CompilationConfig
    "--limit-mm-per-prompt.image", "1"
    "--limit-mm-per-prompt.video", "0"
    "--skip-mm-profiling"
)

if ($CpuOffloadGb -ne "0") {
    $ArgumentList += @("--cpu-offload-gb", $CpuOffloadGb)
}

if ($EnforceEager -eq "1") {
    $ArgumentList += "--enforce-eager"
}

$Process = Start-Process -FilePath $VllmBin -ArgumentList $ArgumentList -RedirectStandardOutput $LogFile -RedirectStandardError $LogFile -PassThru
Set-Content -Path $PidFile -Value $Process.Id

Write-Host "[info] waiting for Qwen service on $ServiceUrl"
Write-Host "[info] startup pid: $($Process.Id)"

$Attempts = [Math]::Max(1, [int]($WaitSeconds / $CheckIntervalSeconds))
for ($i = 0; $i -lt $Attempts; $i++) {
    $Payload = Test-ServiceReady -Url $ServiceUrl -BearerToken $ApiKey
    if ($Payload) {
        Write-Host "[ok] Qwen service is ready."
        Write-Host $Payload
        exit 0
    }
    Start-Sleep -Seconds $CheckIntervalSeconds
}

Write-Host "[error] Qwen service did not become ready within $WaitSeconds seconds."
Write-Host "[error] check log: $LogFile"
if (Test-Path $LogFile -PathType Leaf) {
    Get-Content $LogFile -Tail 80
}
exit 1
