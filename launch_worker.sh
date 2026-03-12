#!/bin/bash
# ============================================================
# Launch Worker Node (Rank 1-3) — RX 7900 XTX, WSL2 + ROCm
# Run inside WSL2 on each AMD worker machine
# Usage: bash launch_worker.sh <rank> [script.py]
#   rank: 1-3 (each worker gets a unique rank)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/gpu-cluster-env/bin/activate"

# -----------------------------------------------------------
# Validate rank argument
# -----------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <rank> [script.py]"
    echo "  rank: 1-3 (unique ID for this worker)"
    exit 1
fi

NODE_RANK=$1

if [ "$NODE_RANK" -lt 1 ] || [ "$NODE_RANK" -gt 3 ]; then
    echo "Error: rank must be between 1 and 3"
    exit 1
fi

# -----------------------------------------------------------
# Load configuration
# -----------------------------------------------------------
CONFIG_FILE="${SCRIPT_DIR}/cluster_config.yaml"

MASTER_IP=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
print(cfg['cluster']['master']['ip'])
")

MASTER_PORT=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
print(cfg['cluster']['master']['port'])
")

WORLD_SIZE=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
print(cfg['training']['world_size'])
")

BATCH_SIZE=$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
print(cfg['training']['batch_size_per_gpu'])
")

# -----------------------------------------------------------
# Environment: ROCm + distributed (WSL2)
# -----------------------------------------------------------
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT="$MASTER_PORT"
export WORLD_SIZE="$WORLD_SIZE"
export RANK="$NODE_RANK"
export LOCAL_RANK=0

# Gloo settings — WSL2 virtual ethernet
export GLOO_SOCKET_IFNAME=eth0
export RCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# ROCm performance tuning
export GPU_MAX_HW_QUEUES=8
export HIP_FORCE_DEV_KERNARG=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "  Starting Worker (Rank ${NODE_RANK}) — WSL2"
echo "============================================"
echo "  Master IP:   $MASTER_IP"
echo "  Master Port: $MASTER_PORT"
echo "  World Size:  $WORLD_SIZE"
echo "  Node Rank:   $NODE_RANK"
echo "  GPU:         $(rocm-smi --showproductname 2>/dev/null | grep 'Card series' | head -1 || echo 'RX 7900 XTX')"
echo "  Backend:     gloo (mixed CUDA + ROCm)"
echo "  OS:          WSL2 ($(lsb_release -ds 2>/dev/null || echo 'Ubuntu'))"
echo "============================================"

# -----------------------------------------------------------
# Launch training script
# -----------------------------------------------------------
SCRIPT="${2:-${SCRIPT_DIR}/distributed_train.py}"

python3 "$SCRIPT" \
    --master_addr "$MASTER_IP" \
    --master_port "$MASTER_PORT" \
    --world_size "$WORLD_SIZE" \
    --rank "$NODE_RANK" \
    --local_rank 0 \
    --backend gloo \
    --batch_size "$BATCH_SIZE" \
    --config "$CONFIG_FILE"
