# ============================================================
# Launch Master Node (Rank 0) — RTX 5090, Windows Native
# Run this in PowerShell on the master machine
# Usage: .\launch_master.ps1 [script.py]
# ============================================================

param(
    [string]$Script = "distributed_train.py"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv
& "$env:USERPROFILE\gpu-cluster-env\Scripts\Activate.ps1"

# Load config
$config = python -c @"
import yaml, json
with open(r'$ScriptDir\cluster_config.yaml') as f:
    cfg = yaml.safe_load(f)
print(json.dumps({
    'master_ip': cfg['cluster']['master']['ip'],
    'master_port': cfg['cluster']['master']['port'],
    'world_size': cfg['training']['world_size'],
    'batch_size': cfg['training']['batch_size_per_gpu']
}))
"@ | ConvertFrom-Json

$masterIp   = $config.master_ip
$masterPort = $config.master_port
$worldSize  = $config.world_size
$batchSize  = $config.batch_size

# Environment variables
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MASTER_ADDR = $masterIp
$env:MASTER_PORT = $masterPort
$env:WORLD_SIZE  = $worldSize
$env:RANK        = "0"
$env:LOCAL_RANK  = "0"
$env:GLOO_SOCKET_IFNAME = "Ethernet"  # Windows interface name — adjust if needed

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Master Node (Rank 0) - RTX 5090" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master IP:   $masterIp"
Write-Host "  Master Port: $masterPort"
Write-Host "  World Size:  $worldSize"

try {
    $gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    Write-Host "  GPU:         $gpuName"
} catch {
    Write-Host "  GPU:         RTX 5090 (nvidia-smi unavailable)"
}

Write-Host "  Backend:     gloo (mixed CUDA + ROCm)"
Write-Host "============================================" -ForegroundColor Cyan

# Launch training
$scriptPath = Join-Path $ScriptDir $Script

python $scriptPath `
    --master_addr $masterIp `
    --master_port $masterPort `
    --world_size $worldSize `
    --rank 0 `
    --local_rank 0 `
    --backend gloo `
    --batch_size $batchSize `
    --config "$ScriptDir\cluster_config.yaml"
