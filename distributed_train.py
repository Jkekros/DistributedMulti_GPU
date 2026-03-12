"""
============================================================
Distributed Training Script — Heterogeneous GPU Cluster
Master: NVIDIA RTX 5090 (CUDA, Windows native)
Workers: AMD RX 7900 XTX (ROCm, WSL2 Ubuntu)
Backend: gloo (the only backend supporting mixed vendors)

This script:
  1. Initializes PyTorch distributed with gloo backend
  2. Detects whether running on CUDA or ROCm automatically
  3. Shards model + data across all 4 GPUs (1 master + 3 workers)
  4. Handles CPU-mediated tensor transfers for cross-vendor comms
  5. Includes a demo training loop you can replace with your model
============================================================
"""

import os
import sys
import time
import argparse
import socket
import yaml
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


# ============================================================
# GPU Backend Detection
# ============================================================
def detect_gpu_backend():
    """
    Detect whether this node has CUDA (NVIDIA) or ROCm (AMD).
    PyTorch ROCm maps to torch.cuda API, so we distinguish
    by checking for NVIDIA vs AMD device names.
    """
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected — falling back to CPU")
        return "cpu", torch.device("cpu")

    device_name = torch.cuda.get_device_name(0).lower()
    if "nvidia" in device_name or "geforce" in device_name or "rtx" in device_name:
        backend = "cuda"
    else:
        # AMD GPUs show up as e.g. "Radeon RX 7900 XTX" via ROCm
        backend = "rocm"

    device = torch.device("cuda:0")
    return backend, device


# ============================================================
# Gloo-Safe Collective Operations
# ============================================================
# Gloo backend does NOT support GPU tensors directly for all ops
# in mixed-vendor setups. We move to CPU for collectives, then
# back to GPU for compute. This is the key to making mixed
# CUDA + ROCm clusters work.

def safe_all_reduce(tensor, op=dist.ReduceOp.SUM):
    """All-reduce that works across CUDA + ROCm via CPU staging."""
    cpu_tensor = tensor.detach().cpu()
    dist.all_reduce(cpu_tensor, op=op)
    tensor.data.copy_(cpu_tensor.to(tensor.device))
    return tensor


def safe_broadcast(tensor, src=0):
    """Broadcast that works across CUDA + ROCm via CPU staging."""
    cpu_tensor = tensor.detach().cpu()
    dist.broadcast(cpu_tensor, src=src)
    tensor.data.copy_(cpu_tensor.to(tensor.device))
    return tensor


def safe_all_gather(tensor_list, tensor):
    """All-gather that works across CUDA + ROCm via CPU staging."""
    cpu_tensor = tensor.detach().cpu()
    cpu_list = [torch.zeros_like(cpu_tensor) for _ in tensor_list]
    dist.all_gather(cpu_list, cpu_tensor)
    for i, t in enumerate(cpu_list):
        tensor_list[i].data.copy_(t.to(tensor_list[i].device))
    return tensor_list


# ============================================================
# Custom DDP Wrapper for Mixed Clusters
# ============================================================
class MixedClusterDDP(nn.Module):
    """
    A simplified DistributedDataParallel that uses CPU-staged
    gradient synchronization. Required because standard DDP
    uses NCCL/RCCL which can't cross vendor boundaries.

    For each backward pass, gradients are:
      1. Moved to CPU
      2. All-reduced across all nodes via gloo
      3. Moved back to the local GPU
    """

    def __init__(self, module, device, world_size):
        super().__init__()
        self.module = module
        self.device = device
        self.world_size = world_size

        # Broadcast initial parameters from rank 0 to all workers
        self._sync_initial_params()

        # Register gradient hooks
        self._register_hooks()

    def _sync_initial_params(self):
        """Ensure all nodes start with identical model weights."""
        for param in self.module.parameters():
            safe_broadcast(param.data, src=0)

    def _register_hooks(self):
        """Register backward hooks for gradient synchronization."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(grad):
            # All-reduce gradients via CPU (gloo)
            safe_all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)
            return grad
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# ============================================================
# Demo Model (replace with your actual model)
# ============================================================
class DemoModel(nn.Module):
    """
    Simple model for testing the cluster. Replace this with
    your actual model (LLM, diffusion model, etc.)
    """

    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Demo Dataset
# ============================================================
def create_demo_dataset(num_samples=10000, input_dim=1024, output_dim=512):
    """Create a random dataset for testing. Replace with real data."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    return TensorDataset(X, y)


# ============================================================
# Training Loop
# ============================================================
def train(args):
    # ------ Init distributed ------
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(minutes=10),
    )

    # ------ Detect GPU ------
    gpu_backend, device = detect_gpu_backend()
    hostname = socket.gethostname()

    print(f"[Rank {args.rank}] {hostname} | GPU backend: {gpu_backend} | "
          f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ------ Create model ------
    model = DemoModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Rank {args.rank}] Model parameters: {param_count:,}")

    # ------ Wrap in mixed-cluster DDP ------
    model = MixedClusterDDP(model, device, args.world_size)

    # ------ Optimizer ------
    optimizer = optim.AdamW(model.module.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()

    # ------ Dataset + Distributed Sampler ------
    dataset = create_demo_dataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # ------ Training loop ------
    num_epochs = args.epochs
    print(f"[Rank {args.rank}] Starting training: {num_epochs} epochs, "
          f"batch_size={args.batch_size}, {len(dataloader)} batches/epoch")

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()  # Gradients are auto-synced by MixedClusterDDP hooks
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 50 == 0 and args.rank == 0:
                print(f"  [Epoch {epoch+1}/{num_epochs}] "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(batch_count, 1)

        # Aggregate loss across all ranks
        loss_tensor = torch.tensor([avg_loss], device=device)
        safe_all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / args.world_size

        if args.rank == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Global Loss: {global_avg_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Throughput: {len(dataset)/epoch_time:.0f} samples/s")

    # ------ Save model (rank 0 only) ------
    if args.rank == 0:
        save_path = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
        }, save_path)
        print(f"[Rank 0] Model saved to {save_path}")

    # ------ Cleanup ------
    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] Training complete. Exiting.")


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Distributed Training — Mixed GPU Cluster")
    parser.add_argument("--master_addr", type=str, required=True, help="Master node IP")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of GPUs")
    parser.add_argument("--rank", type=int, required=True, help="This node's rank (0=master)")
    parser.add_argument("--local_rank", type=int, default=0, help="Local GPU rank (always 0)")
    parser.add_argument("--backend", type=str, default="gloo", help="Distributed backend (gloo)")
    parser.add_argument("--config", type=str, default="cluster_config.yaml", help="Config path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    # Load config for any additional settings
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if args.batch_size == 32:  # Use config default if not overridden
            args.batch_size = config.get("training", {}).get("batch_size_per_gpu", 32)

    train(args)


if __name__ == "__main__":
    main()
