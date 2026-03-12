# ============================================================
# ComfyUI Setup — Master Node (RTX 5090, Windows Native)
# Installs ComfyUI + load balancer on the master machine
# Run in PowerShell (Admin not required unless firewall needed)
# ============================================================

param(
    [string]$ComfyDir = "C:\ComfyUI",
    [switch]$SkipComfyInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Log($msg) { Write-Host "[COMFY-MASTER] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# -----------------------------------------------------------
# 1. Activate cluster venv
# -----------------------------------------------------------
$venvDir = "$env:USERPROFILE\gpu-cluster-env"
if (-not (Test-Path "$venvDir\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Run setup_master.ps1 first." -ForegroundColor Red
    exit 1
}
& "$venvDir\Scripts\Activate.ps1"

# -----------------------------------------------------------
# 2. Install ComfyUI
# -----------------------------------------------------------
if (-not $SkipComfyInstall) {
    Write-Log "Installing ComfyUI at $ComfyDir..."
    
    if (Test-Path "$ComfyDir\main.py") {
        Write-Warn "ComfyUI already exists at $ComfyDir"
    } else {
        git clone https://github.com/comfyanonymous/ComfyUI.git $ComfyDir
    }

    # Install ComfyUI dependencies with CUDA PyTorch
    Push-Location $ComfyDir
    pip install -r requirements.txt
    Pop-Location

    Write-Log "ComfyUI installed."
} else {
    Write-Warn "Skipping ComfyUI install (--SkipComfyInstall)"
}

# -----------------------------------------------------------
# 3. Install load balancer dependencies
# -----------------------------------------------------------
Write-Log "Installing load balancer dependencies..."
pip install aiohttp aiohttp-cors websockets pyyaml psutil

# -----------------------------------------------------------
# 4. Create model directories (shared)
# -----------------------------------------------------------
$modelsDir = "$ComfyDir\models"
Write-Log "Ensuring model directories exist at $modelsDir..."

$subDirs = @("checkpoints", "clip", "controlnet", "embeddings", "loras", "unet", "vae", "upscale_models")
foreach ($d in $subDirs) {
    $p = Join-Path $modelsDir $d
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
}

# -----------------------------------------------------------
# 5. Firewall rules for ComfyUI + balancer
# -----------------------------------------------------------
Write-Log "Adding firewall rules..."
try {
    New-NetFirewallRule -DisplayName "ComfyUI (8188)" `
        -Direction Inbound -Protocol TCP -LocalPort 8188 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "ComfyUI Balancer (8100)" `
        -Direction Inbound -Protocol TCP -LocalPort 8100 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "Model Shard Coordinator (29600)" `
        -Direction Inbound -Protocol TCP -LocalPort 29600 -Action Allow -ErrorAction SilentlyContinue
} catch {
    Write-Warn "Run as Admin to add firewall rules, or add manually."
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI Master Setup Complete" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI:     $ComfyDir"
Write-Host "  Models:      $modelsDir"
Write-Host ""
Write-Log "Next: Place your models in $modelsDir\checkpoints\"
Write-Log "Then run: .\launch_comfy_cluster.ps1"
