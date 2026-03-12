#!/bin/bash
# ============================================================
# Worker Node Setup — Step 2: Inside WSL2 (Ubuntu 22.04)
# This script runs INSIDE WSL2 on each AMD worker machine
# Called automatically by setup_worker.ps1, or run manually:
#   wsl -d Ubuntu-22.04 -- sudo bash setup_worker_wsl.sh
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[WSL2-SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

log "Setting up ROCm inside WSL2 for RX 7900 XTX"

# -----------------------------------------------------------
# 1. System prerequisites
# -----------------------------------------------------------
log "Installing system packages..."
apt-get update && apt-get upgrade -y
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    net-tools \
    htop \
    libnuma-dev \
    libopenmpi-dev \
    openmpi-bin \
    openssh-server

# -----------------------------------------------------------
# 2. Install ROCm for WSL2
# -----------------------------------------------------------
ROCM_VERSION="6.3"

log "Installing ROCm ${ROCM_VERSION} (WSL2 mode)..."

# AMD GPU driver is provided by Windows — we only need ROCm userspace
mkdir -p /etc/apt/keyrings
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Use the Ubuntu 22.04 (jammy) repo
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" | \
    tee /etc/apt/sources.list.d/rocm.list

# Pin ROCm packages
cat <<EOF | tee /etc/apt/preferences.d/rocm-pin-600
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

apt-get update

# Install ROCm userspace libraries (no kernel driver needed in WSL2)
apt-get install -y \
    rocm-dev \
    rocm-libs \
    rccl \
    hip-runtime-amd \
    rocm-smi-lib \
    rocminfo

# -----------------------------------------------------------
# 3. Environment setup
# -----------------------------------------------------------
log "Configuring ROCm environment..."

cat <<'EOF' > /etc/profile.d/rocm-wsl.sh
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/opencl/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
EOF

source /etc/profile.d/rocm-wsl.sh

# Add all users to required groups
REAL_USER="${SUDO_USER:-$USER}"
usermod -aG video "$REAL_USER" 2>/dev/null || true
usermod -aG render "$REAL_USER" 2>/dev/null || true

# -----------------------------------------------------------
# 4. Create Python virtual environment
# -----------------------------------------------------------
VENV_DIR="/home/${REAL_USER}/gpu-cluster-env"
log "Creating Python venv at ${VENV_DIR}..."

sudo -u "$REAL_USER" python3 -m venv "$VENV_DIR"
sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# -----------------------------------------------------------
# 5. Install PyTorch with ROCm support
# -----------------------------------------------------------
log "Installing PyTorch with ROCm..."

sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.2

# -----------------------------------------------------------
# 6. Install cluster dependencies
# -----------------------------------------------------------
log "Installing cluster dependencies..."

sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install \
    pyyaml \
    psutil \
    paramiko \
    flask \
    requests \
    accelerate \
    transformers \
    datasets \
    safetensors \
    tqdm \
    tensorboard

# -----------------------------------------------------------
# 7. Configure SSH inside WSL2
# -----------------------------------------------------------
log "Configuring SSH server inside WSL2..."

# Generate host keys if missing
ssh-keygen -A 2>/dev/null || true

# Configure sshd for WSL2
sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config

# Start SSH (systemd may not be available in WSL2)
service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null || true

# -----------------------------------------------------------
# 8. Network tuning
# -----------------------------------------------------------
log "Applying network performance tuning..."

sysctl -w net.core.rmem_max=16777216 2>/dev/null || true
sysctl -w net.core.wmem_max=16777216 2>/dev/null || true
sysctl -w net.core.rmem_default=1048576 2>/dev/null || true
sysctl -w net.core.wmem_default=1048576 2>/dev/null || true

# -----------------------------------------------------------
# 9. Create auto-start script for WSL2
# -----------------------------------------------------------
log "Creating WSL2 boot script..."

cat <<'BOOT_EOF' > /home/${REAL_USER}/start_cluster_services.sh
#!/bin/bash
# Start services needed for GPU cluster on WSL2 boot
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64

# Start SSH
sudo service ssh start 2>/dev/null || sudo /usr/sbin/sshd 2>/dev/null || true

echo "WSL2 cluster services started."
echo "GPU: $(rocminfo 2>/dev/null | grep 'Marketing Name' | head -1 || echo 'checking...')"
echo "IP:  $(hostname -I | awk '{print $1}')"
BOOT_EOF

chmod +x "/home/${REAL_USER}/start_cluster_services.sh"
chown "$REAL_USER:$REAL_USER" "/home/${REAL_USER}/start_cluster_services.sh"

# -----------------------------------------------------------
# 10. Verify
# -----------------------------------------------------------
log "Verifying installation..."

echo ""
echo "============================================"
echo "  WSL2 Worker Setup Complete (RX 7900 XTX)"
echo "============================================"

# Check ROCm
if command -v rocminfo &> /dev/null; then
    echo -e "${GREEN}[OK]${NC} ROCm installed"
    rocminfo 2>/dev/null | grep "Marketing Name" || echo "  (GPU may not be visible until Windows reboot)"
else
    echo -e "${RED}[FAIL]${NC} ROCm not found"
fi

# Check PyTorch
sudo -u "$REAL_USER" "$VENV_DIR/bin/python3" -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available:  {torch.cuda.is_available()}')
print(f'GPU count:       {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name:        {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
" 2>/dev/null || echo -e "${RED}[FAIL]${NC} PyTorch ROCm verification failed (may need reboot)"

echo ""
log "NEXT STEPS:"
echo "  1. Reboot Windows to apply GPU passthrough"
echo "  2. After reboot, open PowerShell and run:"
echo "     wsl -d Ubuntu-22.04 -- bash ~/start_cluster_services.sh"
echo "  3. Verify GPU: wsl -d Ubuntu-22.04 -- rocminfo"
echo "  4. Accept SSH key from master node"
