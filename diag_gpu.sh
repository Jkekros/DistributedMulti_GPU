#!/bin/bash
# Quick GPU diagnostic for WSL2 AMD workers
# Run: wsl -d Ubuntu-22.04 -- bash diag_gpu.sh

echo "=== WSL2 GPU Diagnostic ==="
echo ""
echo "1. Kernel version:"
uname -r
echo ""
echo "2. /dev/kfd (ROCm kernel device):"
ls -la /dev/kfd 2>&1
echo ""
echo "3. /dev/dri/ (GPU render nodes):"
ls -la /dev/dri/ 2>&1
echo ""
echo "4. dmesg GPU messages:"
dmesg 2>/dev/null | grep -i -E "amdgpu|drm|gpu|kfd" | head -10
echo "(if empty, dmesg may need root)"
echo ""
echo "5. rocminfo:"
rocminfo 2>&1 | head -30
echo ""
echo "6. ROCm SMI:"
rocm-smi 2>&1 | head -15
echo ""
echo "7. HIP devices:"
/opt/rocm/bin/hipInfo 2>&1 | head -15 || echo "hipInfo not found"
echo ""
echo "8. Windows GPU driver (DirectX):"
ls -la /dev/dxg 2>&1
echo ""
echo "=== Done ==="
