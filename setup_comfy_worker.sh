#!/bin/bash
# ============================================================
# ComfyUI Setup — Worker Node (RX 7900 XTX, inside WSL2)
# Installs ComfyUI with ROCm PyTorch inside WSL2
# Run: wsl -d Ubuntu-22.04 -- bash setup_comfy_worker.sh
# ============================================================

set -euo pipefail

COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
VENV_DIR="$HOME/gpu-cluster-env"

log()  { echo -e "\033[0;32m[COMFY-WORKER]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }

# Activate venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run setup_worker_wsl.sh first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ROCm environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

# -----------------------------------------------------------
# 1. Install ComfyUI
# -----------------------------------------------------------
log "Installing ComfyUI at $COMFY_DIR..."

if [ -f "$COMFY_DIR/main.py" ]; then
    warn "ComfyUI already exists at $COMFY_DIR"
else
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
fi

cd "$COMFY_DIR"
pip install -r requirements.txt

# -----------------------------------------------------------
# 2. Link or sync models from shared storage
# -----------------------------------------------------------
log "Setting up model access..."

# Option 1: Symlink to Windows-mounted models (if using shared folder)
# The master can share its C:\ComfyUI\models via SMB
WINDOWS_MODELS="/mnt/c/ComfyUI/models"

if [ -d "$WINDOWS_MODELS" ]; then
    log "Found Windows models at $WINDOWS_MODELS, creating symlinks..."
    for subdir in checkpoints clip controlnet embeddings loras unet vae upscale_models; do
        if [ -d "$WINDOWS_MODELS/$subdir" ]; then
            rm -rf "$COMFY_DIR/models/$subdir" 2>/dev/null || true
            ln -sf "$WINDOWS_MODELS/$subdir" "$COMFY_DIR/models/$subdir"
            log "  Linked models/$subdir -> $WINDOWS_MODELS/$subdir"
        fi
    done
else
    warn "Windows models not found at $WINDOWS_MODELS"
    warn "You'll need to copy/sync models manually, or set up a network share."
    warn "See README for model sharing options."
fi

# -----------------------------------------------------------
# 3. Install additional dependencies
# -----------------------------------------------------------
pip install aiohttp websockets pyyaml psutil

log ""
log "============================================"
log "  ComfyUI Worker Setup Complete"
log "============================================"
log "  ComfyUI dir:  $COMFY_DIR"
log "  GPU backend:  ROCm (HSA_OVERRIDE_GFX_VERSION=11.0.0)"
log ""
log "  To start: bash launch_comfy_worker.sh"
