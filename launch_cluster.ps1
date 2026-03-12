# ============================================================
# Launch Entire Cluster from Master (Windows PowerShell)
# Starts WSL2 workers remotely via SSH, then starts master
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

# Determine project path inside WSL2 on workers
# Workers need the project files copied to their WSL2 filesystem
$WslProjectDir = "~/DistributedMulti_GPU"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Launching GPU Cluster" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master:      $($config.master_ip) (RTX 5090, Windows)"
Write-Host "  Workers:     $($workerIps -join ', ') (RX 7900 XTX, WSL2)"
Write-Host "  World Size:  $($config.world_size)"
Write-Host "  Script:      $Script"
Write-Host "  SSH User:    $SshUser"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------
# Start each worker via SSH -> WSL2
# -----------------------------------------------------------
$workerJobs = @()
$rank = 1

foreach ($ip in $workerIps) {
    Write-Host "[Cluster] Starting rank $rank on $ip (via SSH -> WSL2)..." -ForegroundColor Yellow
    
    # SSH into the Windows machine, then invoke WSL2 to run the worker
    # The worker machines run SSH on Windows, then we call wsl to enter Ubuntu
    $sshCmd = "cd $WslProjectDir && bash launch_worker.sh $rank $Script"
    
    # Start as background job
    $job = Start-Job -ScriptBlock {
        param($user, $ip, $cmd)
        # SSH into the worker's Windows, then run inside WSL2
        & ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${user}@${ip}" "wsl -d Ubuntu-22.04 -- bash -ic '$cmd'"
    } -ArgumentList $SshUser, $ip, $sshCmd
    
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
