# ============================================================
# Worker Node Setup — Step 1: Windows-side WSL2 Setup
# Run this in PowerShell as Administrator on EACH AMD worker
# This installs WSL2 + Ubuntu, then calls the Linux-side setup
# ============================================================

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[WORKER-SETUP] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

Write-Log "Setting up AMD RX 7900 XTX Worker Node (WSL2 + ROCm)"

# -----------------------------------------------------------
# 1. Enable WSL2
# -----------------------------------------------------------
Write-Log "Enabling WSL2..."

# Enable required Windows features
try {
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
} catch {
    Write-Warn "Could not enable WSL features. Ensure you're running as Administrator."
}

# Set WSL2 as default
wsl --set-default-version 2

# -----------------------------------------------------------
# 2. Install Ubuntu 22.04 in WSL2
# -----------------------------------------------------------
Write-Log "Installing Ubuntu 22.04 LTS in WSL2..."

$wslDistros = wsl --list --quiet 2>$null
if ($wslDistros -match "Ubuntu-22.04") {
    Write-Warn "Ubuntu-22.04 already installed in WSL2"
} else {
    wsl --install -d Ubuntu-22.04
    Write-Host ""
    Write-Warn "Ubuntu is installing. You'll be prompted to create a username/password."
    Write-Host "  After setup completes, RE-RUN this script to continue with ROCm install."
    exit 0
}

# -----------------------------------------------------------
# 3. Configure Windows Firewall for cluster communication
# -----------------------------------------------------------
Write-Log "Configuring Windows Firewall..."

try {
    New-NetFirewallRule -DisplayName "GPU Cluster Distributed" `
        -Direction Inbound -Protocol TCP -LocalPort 29400-29600 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "GPU Cluster RCCL" `
        -Direction Inbound -Protocol TCP -LocalPort 29700-29900 -Action Allow -ErrorAction SilentlyContinue
    New-NetFirewallRule -DisplayName "GPU Cluster Monitor" `
        -Direction Inbound -Protocol TCP -LocalPort 8265 -Action Allow -ErrorAction SilentlyContinue
    # Allow WSL2 traffic
    New-NetFirewallRule -DisplayName "WSL2 Inbound" `
        -Direction Inbound -InterfaceAlias "vEthernet (WSL)" -Action Allow -ErrorAction SilentlyContinue
} catch {
    Write-Warn "Some firewall rules may not have been added. Run as Administrator."
}

# -----------------------------------------------------------
# 4. Configure WSL2 for GPU passthrough
# -----------------------------------------------------------
Write-Log "Configuring WSL2 GPU passthrough..."

$wslConfig = @"
[wsl2]
memory=32GB
processors=8
gpuSupport=true
nestedVirtualization=true

[network]
generateResolvConf=true
"@

$wslConfigPath = "$env:USERPROFILE\.wslconfig"
if (-not (Test-Path $wslConfigPath)) {
    Set-Content -Path $wslConfigPath -Value $wslConfig
    Write-Host "  Created $wslConfigPath"
} else {
    Write-Warn ".wslconfig already exists. Verify GPU passthrough is enabled."
}

# -----------------------------------------------------------
# 5. WSL2 port forwarding (so master can reach WSL2 services)
# -----------------------------------------------------------
Write-Log "Setting up port forwarding from Windows to WSL2..."

# Get WSL2 IP
$wslIp = wsl -d Ubuntu-22.04 -- hostname -I 2>$null
$wslIp = $wslIp.Trim().Split(" ")[0]

if ($wslIp) {
    Write-Host "  WSL2 IP: $wslIp"
    
    # Forward PyTorch distributed port into WSL2
    netsh interface portproxy add v4tov4 listenport=29500 listenaddress=0.0.0.0 connectport=29500 connectaddress=$wslIp 2>$null
    netsh interface portproxy add v4tov4 listenport=29501 listenaddress=0.0.0.0 connectport=29501 connectaddress=$wslIp 2>$null
    
    Write-Host "  Port forwarding configured (29500-29501 -> WSL2)"
} else {
    Write-Warn "Could not get WSL2 IP. Start Ubuntu first: wsl -d Ubuntu-22.04"
}

# -----------------------------------------------------------
# 6. Copy and run the Linux-side setup inside WSL2
# -----------------------------------------------------------
Write-Log "Running ROCm setup inside WSL2..."

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$linuxSetupScript = "$scriptDir\setup_worker_wsl.sh"

if (Test-Path $linuxSetupScript) {
    # Convert Windows path to WSL path and run
    $wslPath = wsl -d Ubuntu-22.04 -- wslpath "$linuxSetupScript"
    wsl -d Ubuntu-22.04 -- bash -c "chmod +x $wslPath && sudo bash $wslPath"
} else {
    Write-Warn "setup_worker_wsl.sh not found. Copy it to the same directory as this script."
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Worker Node Windows-Side Setup Complete" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Log "IMPORTANT: A reboot may be required for WSL2 GPU passthrough."
Write-Log "After reboot, verify GPU in WSL2:"
Write-Host '  wsl -d Ubuntu-22.04 -- rocminfo | Select-String "Marketing Name"'
