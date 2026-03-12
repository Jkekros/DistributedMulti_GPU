# ============================================================
# Launch ComfyUI Cluster from Master (Windows PowerShell)
# Starts ComfyUI on all worker nodes via SSH->WSL2,
# starts ComfyUI on master, starts load balancer + shard coordinator
# Usage: .\launch_comfy_cluster.ps1
# ============================================================

param(
    [string]$SshUser = "",
    [switch]$NoBalancer,
    [switch]$NoSharding,
    [switch]$WorkersOnly,
    [switch]$MasterOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- Load config ----
$configRaw = python -c @"
import yaml, json, sys
with open(r'$ScriptDir\cluster_config.yaml') as f:
    cfg = yaml.safe_load(f)
workers = [w['ip'] for w in cfg['cluster']['workers']]
comfy = cfg.get('comfyui', {})
sharding = comfy.get('sharding', {})
print(json.dumps({
    'master_ip': cfg['cluster']['master']['ip'],
    'workers': workers,
    'balancer_port': comfy.get('balancer_port', 8100),
    'master_comfy_port': comfy.get('master_comfy_port', 8188),
    'worker_comfy_ports': comfy.get('worker_comfy_ports', [8188, 8188, 8188]),
    'coordinator_port': sharding.get('coordinator_port', 29600),
    'sharding_enabled': sharding.get('enabled', False),
    'models_path': comfy.get('models_path', 'C:\\\\ComfyUI\\\\models'),
    'models_path_wsl': comfy.get('models_path_wsl', '/mnt/c/ComfyUI/models'),
}))
"@

$config = $configRaw | ConvertFrom-Json
$workerIps = $config.workers

if (-not $SshUser) {
    $SshUser = $env:USERNAME
}

$ComfyDirMaster = "C:\ComfyUI"
$ComfyDirWSL    = "~/ComfyUI"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI GPU Cluster Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master:       $($config.master_ip) (RTX 5090)"
Write-Host "  Workers:      $($workerIps -join ', ') (RX 7900 XTX)"
Write-Host "  Balancer:     http://$($config.master_ip):$($config.balancer_port)"
Write-Host "  Dashboard:    http://$($config.master_ip):$($config.balancer_port)/cluster"
Write-Host "  Sharding:     $(if ($config.sharding_enabled -and -not $NoSharding) {'Enabled'} else {'Disabled'})"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$allJobs = @()

# ============================================================
# 1. Start ComfyUI on worker nodes (WSL2)
# ============================================================
if (-not $MasterOnly) {
    $rank = 1
    foreach ($ip in $workerIps) {
        $workerPort = $config.worker_comfy_ports[$rank - 1]
        Write-Host "[ComfyUI] Starting worker rank $rank at $ip`:$workerPort ..." -ForegroundColor Yellow

        $wslCmd = @"
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CLUSTER_RANK=$rank
export CLUSTER_COORDINATOR_URL=http://$($config.master_ip):$($config.coordinator_port)
cd $ComfyDirWSL
source venv/bin/activate 2>/dev/null || true
python main.py --listen 0.0.0.0 --port $workerPort --preview-method auto
"@

        $job = Start-Job -ScriptBlock {
            param($user, $ip, $cmd)
            & ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 `
                "${user}@${ip}" "wsl -d Ubuntu-22.04 -- bash -ic '$cmd'"
        } -ArgumentList $SshUser, $ip, ($wslCmd -replace "`n", " && ")

        $allJobs += @{ Job=$job; Name="Worker-$rank ($ip)" }
        $rank++
    }

    Write-Host "[ComfyUI] $($workerIps.Count) worker(s) launching..." -ForegroundColor Green
    Write-Host ""

    # Give workers a moment to start binding ports
    Start-Sleep -Seconds 5
}

# ============================================================
# 2. Start Shard Coordinator on master
# ============================================================
if ($config.sharding_enabled -and -not $NoSharding) {
    Write-Host "[Shard] Starting coordinator on port $($config.coordinator_port)..." -ForegroundColor Magenta

    $shardJob = Start-Job -ScriptBlock {
        param($dir, $port)
        Set-Location $dir
        & python model_shard_coordinator.py --config cluster_config.yaml --port $port
    } -ArgumentList $ScriptDir, $config.coordinator_port

    $allJobs += @{ Job=$shardJob; Name="ShardCoordinator" }
    Start-Sleep -Seconds 2
}

# ============================================================
# 3. Start ComfyUI on master node (Windows native)
# ============================================================
if (-not $WorkersOnly) {
    Write-Host "[ComfyUI] Starting master ComfyUI on port $($config.master_comfy_port)..." -ForegroundColor Yellow

    $env:CLUSTER_RANK = "0"
    $env:CLUSTER_COORDINATOR_URL = "http://localhost:$($config.coordinator_port)"

    $masterJob = Start-Job -ScriptBlock {
        param($comfyDir, $port)
        Set-Location $comfyDir
        & python main.py --listen 0.0.0.0 --port $port --preview-method auto
    } -ArgumentList $ComfyDirMaster, $config.master_comfy_port

    $allJobs += @{ Job=$masterJob; Name="MasterComfyUI" }

    # Wait for master to start
    Start-Sleep -Seconds 8
}

# ============================================================
# 4. Start Load Balancer on master
# ============================================================
if (-not $NoBalancer) {
    Write-Host "[Balancer] Starting load balancer on port $($config.balancer_port)..." -ForegroundColor Cyan

    $balancerJob = Start-Job -ScriptBlock {
        param($dir, $port, $configPath)
        Set-Location $dir
        & python comfy_balancer.py --config $configPath --port $port
    } -ArgumentList $ScriptDir, $config.balancer_port, "$ScriptDir\cluster_config.yaml"

    $allJobs += @{ Job=$balancerJob; Name="LoadBalancer" }
    Start-Sleep -Seconds 3
}

# ============================================================
# Ready!
# ============================================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  ComfyUI Cluster is RUNNING" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Open in browser:" -ForegroundColor White
Write-Host "    http://$($config.master_ip):$($config.balancer_port)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard:" -ForegroundColor White
Write-Host "    http://$($config.master_ip):$($config.balancer_port)/cluster" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Individual nodes:" -ForegroundColor White
Write-Host "    Master:   http://$($config.master_ip):$($config.master_comfy_port)" -ForegroundColor Gray
$rank = 0
foreach ($ip in $workerIps) {
    $p = $config.worker_comfy_ports[$rank]
    Write-Host "    Worker-$($rank+1): http://${ip}:${p}" -ForegroundColor Gray
    $rank++
}
Write-Host ""
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Yellow
Write-Host ""

# ============================================================
# Monitor loop — keep running, show status, clean up on exit
# ============================================================
try {
    while ($true) {
        # Check if any job has failed
        foreach ($entry in $allJobs) {
            $j = $entry.Job
            if ($j.State -eq "Failed") {
                Write-Host "[!] $($entry.Name) FAILED:" -ForegroundColor Red
                Receive-Job -Job $j -ErrorAction SilentlyContinue | Write-Host
            }
        }
        Start-Sleep -Seconds 10
    }
}
finally {
    # Clean up all jobs on exit
    Write-Host ""
    Write-Host "[Cluster] Shutting down all services..." -ForegroundColor Yellow

    foreach ($entry in $allJobs) {
        Write-Host "  Stopping $($entry.Name)..." -ForegroundColor Gray
        Stop-Job -Job $entry.Job -ErrorAction SilentlyContinue
        Remove-Job -Job $entry.Job -Force -ErrorAction SilentlyContinue
    }

    Write-Host "[Cluster] All services stopped." -ForegroundColor Green
}
