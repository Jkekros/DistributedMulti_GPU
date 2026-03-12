# ============================================================
# Launch Entire Cluster from Master (Windows PowerShell)
# Starts workers remotely via SSH, then starts master
# All nodes run Windows natively (CUDA on master, DirectML on workers)
# Usage: .\launch_cluster.ps1 [script.py]
# ============================================================

param(
    [string]$Script = "distributed_train.py",
    [string]$SshUser = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv
& "$env:USERPROFILE\gpu-cluster-env\Scripts\Activate.ps1"

# Load config
$configRaw = python -c @"
import yaml, json
with open(r'$ScriptDir\cluster_config.yaml') as f:
    cfg = yaml.safe_load(f)
workers = [w['ip'] for w in cfg['cluster']['workers']]
print(json.dumps({
    'master_ip': cfg['cluster']['master']['ip'],
    'master_port': cfg['cluster']['master']['port'],
    'world_size': cfg['training']['world_size'],
    'workers': workers
}))
"@

$config = $configRaw | ConvertFrom-Json
$workerIps = $config.workers

# Determine SSH user
if (-not $SshUser) {
    $SshUser = $env:USERNAME
}

# Project path on workers (same structure as master)
$WorkerProjectDir = "DistributedMulti_GPU"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Launching GPU Cluster" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master:      $($config.master_ip) (RTX 5090, CUDA)"
Write-Host "  Workers:     $($workerIps -join ', ') (RX 7900 XTX, DirectML)"
Write-Host "  World Size:  $($config.world_size)"
Write-Host "  Script:      $Script"
Write-Host "  SSH User:    $SshUser"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------
# Start each worker via SSH (Windows to Windows)
# -----------------------------------------------------------
$workerJobs = @()
$rank = 1

foreach ($ip in $workerIps) {
    Write-Host "[Cluster] Starting rank $rank on $ip (via SSH)..." -ForegroundColor Yellow

    # SSH into the worker's Windows and run launch_worker.ps1
    $job = Start-Job -ScriptBlock {
        param($user, $ip, $rank, $script, $projectDir)
        & ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 `
            "${user}@${ip}" `
            "cd $projectDir; powershell -ExecutionPolicy Bypass -File launch_worker.ps1 -Rank $rank -Script $script"
    } -ArgumentList $SshUser, $ip, $rank, $Script, $WorkerProjectDir

    $workerJobs += $job
    $rank++
}

Write-Host ""
Write-Host "[Cluster] All $($workerJobs.Count) workers launched. Starting master (rank 0)..." -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------
# Start master locally (blocks until training completes)
# -----------------------------------------------------------
& "$ScriptDir\launch_master.ps1" -Script $Script

# -----------------------------------------------------------
# Wait for workers to finish
# -----------------------------------------------------------
Write-Host ""
Write-Host "[Cluster] Master finished. Waiting for workers..." -ForegroundColor Yellow

foreach ($job in $workerJobs) {
    $result = Receive-Job -Job $job -Wait -ErrorAction SilentlyContinue
    if ($result) { Write-Host $result }
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "[Cluster] All nodes finished. Training complete." -ForegroundColor Green
