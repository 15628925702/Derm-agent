param(
    [string]$Url = "https://drive.google.com/drive/folders/1saXHjVrjRhbvNuuLMMqU2j2mbkcgc0gq?usp=sharing",
    [string]$TargetDir = "G:\0-newResearch\temp\xiangya_drive_raw",
    [string]$PythonExe = "G:\Anaconda\envs\dermagent-xh\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

Write-Host "Drive URL   : $Url"
Write-Host "Target dir  : $TargetDir"
Write-Host ""

& $PythonExe -m gdown --folder --continue --output $TargetDir $Url
