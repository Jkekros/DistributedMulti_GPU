# Mixed GPU Cluster: RTX 5090 + 3x RX 7900 XTX (Windows)

Distributed PyTorch cluster unifying **1 NVIDIA RTX 5090** (master, Windows + CUDA) with **3 AMD RX 7900 XTX** (workers, Windows + WSL2 + ROCm) as a single compute fabric.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MASTER (Rank 0)                     │
│         NVIDIA RTX 5090 — 32 GB VRAM            │
│     Windows Native │ PyTorch + CUDA              │
│           IP: 192.168.0.15:29500                │
└────────────┬──────────┬──────────┬──────────────┘
             │          │          │
     ┌───────┘      ┌───┘      ┌───┘
     │              │          │
┌────┴────────┐ ┌───┴────────┐ ┌┴─────────────┐
│  Worker 1   │ │  Worker 2  │ │   Worker 3   │
│  Rank 1     │ │  Rank 2    │ │   Rank 3     │
│  7900 XTX   │ │  7900 XTX  │ │   7900 XTX   │
│  WSL2+ROCm  │ │  WSL2+ROCm │ │   WSL2+ROCm  │
│  24 GB VRAM │ │  24 GB     │ │   24 GB      │
│  .0.171     │ │  .0.172    │ │   .0.174     │
└─────────────┘ └────────────┘ └──────────────┘
```

**Total VRAM: 32 + (3 × 24) = 104 GB**

## How It Works

- **Backend: `gloo`** — the only PyTorch distributed backend that works across both CUDA and ROCm
- **Master:** Windows native with CUDA PyTorch (RTX 5090)
- **Workers:** Windows + WSL2 (Ubuntu 22.04) with ROCm PyTorch (RX 7900 XTX)
- PyTorch's `torch.cuda` API works for both NVIDIA (CUDA) and AMD (ROCm via HIP)
- Cross-vendor gradient sync goes through **CPU-staged tensor transfers**:
  GPU → CPU → gloo all-reduce → CPU → GPU
- The `MixedClusterDDP` wrapper in `distributed_train.py` handles this automatically

## Files

| File | Where it runs | Purpose |
|------|---------------|---------|
| `cluster_config.yaml` | — | IPs, ports, GPU types, training params |
| `setup_master.ps1` | Master (PowerShell) | Install CUDA PyTorch on Windows |
| `setup_worker.ps1` | Workers (PowerShell) | Install WSL2, configure GPU passthrough |
| `setup_worker_wsl.sh` | Workers (inside WSL2) | Install ROCm + PyTorch in WSL2 |
| `launch_master.ps1` | Master (PowerShell) | Start rank 0 training |
| `launch_worker.sh` | Workers (inside WSL2) | Start rank 1-3 training |
| `launch_cluster.ps1` | Master (PowerShell) | One-click: SSH → WSL2 workers + start master |
| `distributed_train.py` | All nodes | Training with mixed-vendor DDP |
| `health_check.py` | Master | Verify nodes, GPUs, connectivity |
| **ComfyUI** | | |
| `setup_comfy_master.ps1` | Master (PowerShell) | Install ComfyUI + balancer deps on master |
| `setup_comfy_worker.sh` | Workers (inside WSL2) | Install ComfyUI in WSL2, link models |
| `launch_comfy_cluster.ps1` | Master (PowerShell) | One-click ComfyUI cluster launch |
| `comfy_balancer.py` | Master | Load balancer across all ComfyUI nodes |
| `model_shard_coordinator.py` | Master | Coordinate model sharding across GPUs |
| `comfy_cluster_node.py` | All nodes (custom node) | Shard-aware checkpoint loader for ComfyUI |

## Quick Start

### 1. Edit Configuration

Open `cluster_config.yaml` and verify your IPs are correct:
```yaml
master:
  ip: "192.168.0.15"      # Your RTX 5090 machine
workers:
  - ip: "192.168.0.171"   # Worker 1
  - ip: "192.168.0.172"   # Worker 2
  - ip: "192.168.0.174"   # Worker 3
```

### 2. Setup Master (RTX 5090 Machine)

Open PowerShell **as Administrator**:
```powershell
.\setup_master.ps1
```

This installs:
- Python venv with PyTorch CUDA
- Windows Firewall rules
- OpenSSH Server

### 3. Setup Each Worker (RX 7900 XTX Machines)

On each AMD worker, open PowerShell **as Administrator**:
```powershell
.\setup_worker.ps1
```

This installs:
- WSL2 with Ubuntu 22.04
- GPU passthrough configuration
- Port forwarding (Windows → WSL2)
- Then automatically runs `setup_worker_wsl.sh` inside WSL2 to install ROCm + PyTorch

**After first run, reboot and run again** (first run enables WSL2, second run installs Ubuntu + ROCm).

### 4. Verify GPU in WSL2 (on each worker)

After reboot:
```powershell
wsl -d Ubuntu-22.04 -- rocminfo
```
You should see your RX 7900 XTX listed.

### 5. Configure SSH

From master, set up SSH keys to each worker:
```powershell
# Generate key if needed
ssh-keygen -t rsa -b 4096

# Copy to each worker (SSH must be enabled on workers)
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh user@192.168.0.171 "cat >> .ssh/authorized_keys"
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh user@192.168.0.172 "cat >> .ssh/authorized_keys"
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh user@192.168.0.174 "cat >> .ssh/authorized_keys"
```

### 6. Health Check

From the master:
```powershell
& "$env:USERPROFILE\gpu-cluster-env\Scripts\Activate.ps1"
python health_check.py --ssh-user your_username
```

### 7. Launch Training

**Option A — One command from master (recommended):**
```powershell
.\launch_cluster.ps1
```
This SSHes into all workers, launches WSL2 training processes, and starts the master locally.

**Option B — Manual launch:**

On master (PowerShell):
```powershell
.\launch_master.ps1
```

On each worker (open WSL2 terminal):
```bash
cd /mnt/c/path/to/DistributedMulti_GPU
bash launch_worker.sh 1   # worker 1
bash launch_worker.sh 2   # worker 2
bash launch_worker.sh 3   # worker 3
```

## WSL2 Networking Notes

### Port Forwarding
WSL2 uses NAT networking — it has its own internal IP. `setup_worker.ps1` configures
Windows port forwarding so the master can reach SOCm services inside WSL2:

```
Master → Worker Windows IP (192.168.0.171:29500)
    → WSL2 port forward → Ubuntu (172.x.x.x:29500)
```

### WSL2 IP Changes on Reboot
WSL2 gets a new internal IP each time Windows reboots. The port forwarding in
`setup_worker.ps1` needs to be refreshed after each reboot. You can run:
```powershell
# On each worker after reboot:
$wslIp = wsl -d Ubuntu-22.04 -- hostname -I
$wslIp = $wslIp.Trim().Split(" ")[0]
netsh interface portproxy add v4tov4 listenport=29500 listenaddress=0.0.0.0 connectport=29500 connectaddress=$wslIp
```

### Start WSL2 Services After Reboot
On each worker after reboot:
```powershell
wsl -d Ubuntu-22.04 -- bash ~/start_cluster_services.sh
```

## Performance Notes

### Bottleneck: Network Bandwidth
All gradient communication goes through CPU and network (gloo):
```
GPU compute → gradients → copy to CPU → gloo all-reduce → copy back to GPU
```

- **1 GbE**: ~125 MB/s — most time spent on gradient sync
- **10 GbE**: ~1.25 GB/s — practical for medium models
- **25/40 GbE**: Ideal

### Tips for Best Performance
1. **Larger batch sizes** — amortize communication overhead
2. **Gradient accumulation** — accumulate N micro-batches before syncing
3. **Mixed precision (FP16/BF16)** — halves gradient traffic
4. **Use wired Ethernet** — Wi-Fi adds too much latency/jitter

### Replacing the Demo Model
Edit `distributed_train.py` and swap `DemoModel` with your model:
```python
model = YourModel().to(device)
model = MixedClusterDDP(model, device, world_size)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `rocminfo` shows nothing in WSL2 | Reboot Windows; install latest AMD Adrenalin driver |
| `Connection refused` on workers | Ensure port forwarding is set up; check WSL2 is running |
| WSL2 IP changed after reboot | Re-run port forwarding (see WSL2 Networking Notes) |
| `HSA_STATUS_ERROR` in WSL2 | Ensure `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set |
| Workers hang at init | Master must start first; check firewall on all machines |
| Slow training | Check network: `iperf3` between nodes; use wired Ethernet |
| `No module named yaml` | Activate venv: `& ~\gpu-cluster-env\Scripts\Activate.ps1` |
| Can't SSH to workers | Enable OpenSSH Server: `Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0` |
| PowerShell execution policy | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## ComfyUI Cluster Mode

Run ComfyUI across all 4 GPUs — the load balancer distributes generation work automatically and the shard coordinator can split large models across multiple nodes.

### ComfyUI Architecture

```
  Browser → http://192.168.0.15:8100
               │
       ┌───────┴───────┐
       │ Load Balancer  │  (comfy_balancer.py)
       │  Port 8100     │  Strategy: least_busy / round_robin / vram_aware
       └──┬───┬───┬───┬─┘
          │   │   │   │
  ┌───────┘   │   │   └───────┐
  │           │   │           │
  ▼           ▼   ▼           ▼
Master:8188  W1  W2          W3
RTX 5090   7900XTX  7900XTX  7900XTX
 32GB       24GB     24GB     24GB
```

### ComfyUI Setup

**1. Setup master (PowerShell as Admin):**
```powershell
.\setup_comfy_master.ps1
```

**2. Setup workers (inside WSL2 on each worker):**
```bash
cd ~/DistributedMulti_GPU
bash setup_comfy_worker.sh
```

**3. Install the custom cluster node on all nodes:**
```powershell
# On master (Windows):
Copy-Item comfy_cluster_node.py C:\ComfyUI\custom_nodes\comfy_cluster_node.py

# On workers (inside WSL2):
cp comfy_cluster_node.py ~/ComfyUI/custom_nodes/comfy_cluster_node.py
```

**4. Share models across all nodes.**
All machines need access to the same checkpoint files. Options:
- **SMB/NFS share**: Mount a network drive on all nodes
- **Manual copy**: Copy models to `C:\ComfyUI\models\` on each machine (WSL2 accesses via `/mnt/c/ComfyUI/models/`)

### Launch ComfyUI Cluster

From the master, one command starts everything:
```powershell
.\launch_comfy_cluster.ps1
```

This starts:
1. ComfyUI on each worker (via SSH → WSL2)
2. Shard coordinator on master (port 29600)
3. ComfyUI on master (port 8188)
4. Load balancer (port 8100)

**Open your browser to `http://192.168.0.15:8100`** — this is the unified entry point. The load balancer routes your prompts to the least-busy node.

### Cluster Dashboard

Visit **`http://192.168.0.15:8100/cluster`** to see:
- All nodes and their status (online/offline)
- Queue depth per node
- VRAM usage
- Active/completed job counts

### Load Balancer Strategies

Set `strategy` in `cluster_config.yaml`:
| Strategy | Behavior |
|----------|----------|
| `least_busy` | Routes to the node with the smallest queue (default) |
| `round_robin` | Alternates between nodes evenly |
| `vram_aware` | Prefers nodes with more free VRAM |

### Model Sharding

For models too large for a single GPU (e.g., >20 GB), the shard coordinator splits the model across multiple GPUs:

1. In ComfyUI, use the **"Load Checkpoint (Cluster Shard)"** node instead of the default loader
2. If the model exceeds `shard_threshold_gb` (default: 20), it auto-shards
3. Each node loads only its assigned layers

Configure in `cluster_config.yaml`:
```yaml
comfyui:
  sharding:
    enabled: true
    coordinator_port: 29600
    shard_threshold_gb: 20
```

### Launch Options

```powershell
# Start everything (default)
.\launch_comfy_cluster.ps1

# Without load balancer (access each node directly)
.\launch_comfy_cluster.ps1 -NoBalancer

# Without model sharding
.\launch_comfy_cluster.ps1 -NoSharding

# Workers only (don't start master ComfyUI)
.\launch_comfy_cluster.ps1 -WorkersOnly

# Master only (no workers)
.\launch_comfy_cluster.ps1 -MasterOnly
```
