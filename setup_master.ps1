# ============================================================
# Master Node Setup — NVIDIA RTX 5090 (Windows Native + CUDA)
# Run this in PowerShell as Administrator on the master machine
# ============================================================

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[MASTER-SETUP] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

Write-Log "Setting up RTX 5090 Master Node (Windows + CUDA)"

# -----------------------------------------------------------
# 1. Check NVIDIA Driver
# -----------------------------------------------------------
Write-Log "Checking NVIDIA driver..."
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host "  GPU: $nvidiaSmi"
} catch {
    Write-Warn "nvidia-smi not found. Install the latest NVIDIA Game Ready or Studio driver from:"
    Write-Host "  https://www.nvidia.com/Download/index.aspx"
    Write-Host "  Select: RTX 5090, Windows 11, Game Ready Driver"
    Write-Host ""
    Write-Host "After installing, reboot and re-run this script."
    exit 1
}

# -----------------------------------------------------------
# 2. Check/Install Python
# -----------------------------------------------------------
Write-Log "Checking Python..."
try {
    $pyVersion = & python --version 2>&1
    Write-Host "  $pyVersion"
} catch {
    Write-Warn "Python not found. Installing via winget..."
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
    Write-Host "  Please restart your terminal after Python install, then re-run this script."
    exit 1
}

# -----------------------------------------------------------
# 3. Create Python virtual environment
# -----------------------------------------------------------
$venvDir = "$env:USERPROFILE\gpu-cluster-env"
Write-Log "Creating virtual environment at $venvDir..."

if (Test-Path $venvDir) {
    Write-Warn "Virtual environment already exists at $venvDir"
} else {
    python -m venv $venvDir
}

# Activate
& "$venvDir\Scripts\Activate.ps1"

python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------
# 4. Install PyTorch with CUDA
# -----------------------------------------------------------
Write-Log "Installing PyTorch with CUDA support..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# -----------------------------------------------------------
# 5. Install cluster dependencies
# -----------------------------------------------------------
Write-Log "Installing cluster dependencies..."

pip install pyyaml psutil paramiko flask requests accelerate transformers datasets safetensors tqdm tensorboard

# -----------------------------------------------------------
# 6. Configure Windows Firewall
# -----------------------------------------------------------
Write-Log "Configuring Windows Firewall rules..."

# PyTorch distributed ports
try {
    New-NetFirewallRule -DisplayName "PyTorch Distributed (TCP 29500)" `
        -Direction Inbound -Protocol TCP -LocalPort 29500 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "PyTorch Distributed Range" `
        -Direction Inbound -Protocol TCP -LocalPort 29400-29600 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "GPU Cluster Monitoring" `
        -Direction Inbound -Protocol TCP -LocalPort 8265 -Action Allow -ErrorAction SilentlyContinue
    Write-Host "  Firewall rules added"
} catch {
    Write-Warn "Could not add firewall rules. Run PowerShell as Administrator."
}

# -----------------------------------------------------------
# 7. Enable SSH Server (for remote management)
# -----------------------------------------------------------
Write-Log "Enabling OpenSSH Server..."
try {
    $sshCapability = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
    if ($sshCapability.State -ne 'Installed') {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    }
    Start-Service sshd -ErrorAction SilentlyContinue
    Set-Service -Name sshd -StartupType Automatic -ErrorAction SilentlyContinue
    Write-Host "  SSH server enabled"
} catch {
    Write-Warn "Could not enable SSH server. Run as Administrator or enable manually."
}

# -----------------------------------------------------------
# 8. Verify installation
# -----------------------------------------------------------
Write-Log "Verifying installation..."

& "$venvDir\Scripts\python.exe" -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'GPU count:       {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name:        {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA version:    {torch.version.cuda}')
"@

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Master Node Setup Complete (RTX 5090)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Log "NEXT STEPS:"
Write-Host "  1. On each AMD worker machine, run setup_worker_wsl.ps1"
Write-Host "  2. Edit cluster_config.yaml with correct IPs if needed"
Write-Host "  3. Run: .\launch_cluster.ps1 to start training"
