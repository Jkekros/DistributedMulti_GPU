# ============================================================
# ComfyUI Setup — Worker Node (RX 7900 XTX, Windows + DirectML)
# Run in PowerShell on each AMD worker machine
# ============================================================

param(
    [string]$ComfyDir = "C:\ComfyUI"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Log($msg) { Write-Host "[COMFY-WORKER] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# -----------------------------------------------------------
# 1. Activate cluster venv
# -----------------------------------------------------------
$venvDir = "$env:USERPROFILE\gpu-cluster-env"
if (-not (Test-Path "$venvDir\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found. Run setup_worker.ps1 first." -ForegroundColor Red
    exit 1
}
& "$venvDir\Scripts\Activate.ps1"

# -----------------------------------------------------------
# 2. Install ComfyUI
# -----------------------------------------------------------
Write-Log "Installing ComfyUI at $ComfyDir..."

if (Test-Path "$ComfyDir\main.py") {
    Write-Warn "ComfyUI already exists at $ComfyDir"
} else {
    git clone https://github.com/comfyanonymous/ComfyUI.git $ComfyDir
}

Push-Location $ComfyDir
pip install -r requirements.txt
Pop-Location

# -----------------------------------------------------------
# 3. Ensure torch-directml is installed for ComfyUI
# -----------------------------------------------------------
Write-Log "Ensuring DirectML is installed..."
pip install torch-directml

# -----------------------------------------------------------
# 4. Create model directories
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
# 5. Install additional deps
# -----------------------------------------------------------
Write-Log "Installing load balancer / cluster deps..."
pip install aiohttp websockets pyyaml psutil

# -----------------------------------------------------------
# 6. Firewall rules
# -----------------------------------------------------------
Write-Log "Adding firewall rules..."
try {
    New-NetFirewallRule -DisplayName "ComfyUI (8188)" `
        -Direction Inbound -Protocol TCP -LocalPort 8188 -Action Allow -ErrorAction SilentlyContinue
} catch {
    Write-Warn "Run as Admin to add firewall rules, or add manually."
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI Worker Setup Complete (DirectML)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI:  $ComfyDir"
Write-Host "  Models:   $modelsDir"
Write-Host ""
Write-Log "Copy models from master to $modelsDir\checkpoints\"
Write-Log "Or set up a network share (see README)"
