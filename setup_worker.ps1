# ============================================================
# Worker Node Setup — AMD RX 7900 XTX (Windows Native + DirectML)
# Run this in PowerShell as Administrator on EACH AMD worker
# No WSL2 required — DirectML works natively on Windows
# ============================================================

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[WORKER-SETUP] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

Write-Log "Setting up AMD RX 7900 XTX Worker Node (Windows + DirectML)"

# -----------------------------------------------------------
# 1. Check AMD GPU driver
# -----------------------------------------------------------
Write-Log "Checking AMD GPU..."
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" } | Select-Object -First 1
    if ($gpu) {
        Write-Host "  GPU:    $($gpu.Name)"
        Write-Host "  Driver: $($gpu.DriverVersion)"
    } else {
        Write-Warn "No AMD GPU detected. Make sure the latest AMD Adrenalin driver is installed."
        Write-Host "  Download from: https://www.amd.com/en/support"
    }
} catch {
    Write-Warn "Could not query GPU info."
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
    Write-Host "  Please restart PowerShell after install, then re-run this script."
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
# 4. Install PyTorch + DirectML
# -----------------------------------------------------------
Write-Log "Installing PyTorch + torch-directml..."

pip install torch torchvision torchaudio
pip install torch-directml

# -----------------------------------------------------------
# 5. Install cluster dependencies
# -----------------------------------------------------------
Write-Log "Installing cluster dependencies..."

pip install pyyaml psutil paramiko flask requests accelerate transformers datasets safetensors tqdm tensorboard

# -----------------------------------------------------------
# 6. Configure Windows Firewall
# -----------------------------------------------------------
Write-Log "Configuring Windows Firewall..."

try {
    New-NetFirewallRule -DisplayName "GPU Cluster Distributed" `
        -Direction Inbound -Protocol TCP -LocalPort 29400-29600 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "GPU Cluster Monitor" `
        -Direction Inbound -Protocol TCP -LocalPort 8265 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "ComfyUI (8188)" `
        -Direction Inbound -Protocol TCP -LocalPort 8188 -Action Allow -ErrorAction SilentlyContinue
    Write-Host "  Firewall rules added"
} catch {
    Write-Warn "Could not add firewall rules. Run as Administrator."
}

# -----------------------------------------------------------
# 7. Enable SSH Server
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
    Write-Warn "Could not enable SSH. Run as Administrator or enable manually."
}

# -----------------------------------------------------------
# 8. Verify installation
# -----------------------------------------------------------
Write-Log "Verifying installation..."

& "$venvDir\Scripts\python.exe" -c @"
import torch
print(f'PyTorch version: {torch.__version__}')

try:
    import torch_directml
    dml_device = torch_directml.device()
    print(f'DirectML available: True')
    print(f'DirectML device:    {dml_device}')
    t = torch.randn(2, 2).to(dml_device)
    r = t @ t.T
    print(f'GPU compute test:   OK ({r.shape})')
except ImportError:
    print('DirectML: NOT INSTALLED - run: pip install torch-directml')
except Exception as e:
    print(f'DirectML error: {e}')
"@

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Worker Node Setup Complete (DirectML)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Log "NEXT STEPS:"
Write-Host "  1. Set up SSH keys from master (see README)"
Write-Host "  2. Run health_check.py from master to verify connectivity"
Write-Host "  3. Run launch_cluster.ps1 from master to start"
