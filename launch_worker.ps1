# ============================================================
# Launch Worker Node (Rank 1-3) — RX 7900 XTX, Windows + DirectML
# Run in PowerShell on each AMD worker machine
# Usage: .\launch_worker.ps1 -Rank <1-3> [-Script distributed_train.py]
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1, 3)]
    [int]$Rank,

    [string]$Script = "distributed_train.py"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv
& "$env:USERPROFILE\gpu-cluster-env\Scripts\Activate.ps1"

# -----------------------------------------------------------
# Load config
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Environment: DirectML + distributed
# -----------------------------------------------------------
$env:MASTER_ADDR = $masterIp
$env:MASTER_PORT = $masterPort
$env:WORLD_SIZE  = $worldSize
$env:RANK        = $Rank
$env:LOCAL_RANK  = "0"
$env:GLOO_SOCKET_IFNAME = "Ethernet"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Worker (Rank $Rank) — DirectML" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master IP:   $masterIp"
Write-Host "  Master Port: $masterPort"
Write-Host "  World Size:  $worldSize"
Write-Host "  Node Rank:   $Rank"
Write-Host "  Backend:     gloo (mixed CUDA + DirectML)"

try {
    $gpuInfo = python -c "import torch_directml; print('DirectML: ' + str(torch_directml.device()))" 2>$null
    Write-Host "  GPU:         $gpuInfo"
} catch {
    Write-Host "  GPU:         RX 7900 XTX (DirectML)"
}

Write-Host "============================================" -ForegroundColor Cyan

# -----------------------------------------------------------
# Launch training
# -----------------------------------------------------------
$scriptPath = Join-Path $ScriptDir $Script

python $scriptPath `
    --master_addr $masterIp `
    --master_port $masterPort `
    --world_size $worldSize `
    --rank $Rank `
    --local_rank 0 `
    --backend gloo `
    --batch_size $batchSize `
    --config "$ScriptDir\cluster_config.yaml"
