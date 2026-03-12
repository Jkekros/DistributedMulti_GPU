"""
============================================================
Model Sharding Coordinator for ComfyUI
Enables loading a single large model (e.g. SDXL, Flux, large LLMs)
across multiple GPUs on different machines.

How it works:
  1. When ComfyUI on any node tries to load a model that's too large
     for its local VRAM, it contacts this coordinator
  2. The coordinator splits the model's state_dict into chunks
  3. Each chunk is assigned to a different node's GPU
  4. During inference, tensors are computed locally on each shard,
     with intermediate results passed between nodes via gloo

Limitations:
  - Higher latency than single-GPU (network round trips per layer)
  - Works best for models that are layer-sequential (transformers)
  - Requires all nodes to be online and healthy
============================================================
"""

import os
import sys
import json
import time
import math
import asyncio
import logging
import argparse
from typing import Optional

import yaml
import torch
import torch.distributed as dist
from datetime import timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SHARD] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class ModelShardCoordinator:
    """
    Coordinates model sharding across cluster nodes.
    Runs on the master node and communicates with workers
    via PyTorch distributed (gloo backend).
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.cluster = self.config["cluster"]
        self.nodes = self._get_nodes()
        self.shard_config = self.config.get("comfyui", {}).get("sharding", {})
        self.threshold_gb = self.shard_config.get("shard_threshold_gb", 20)
        self.is_initialized = False

    def _get_nodes(self):
        """Get all nodes with their VRAM info."""
        nodes = []
        master = self.cluster["master"]
        nodes.append({
            "rank": 0,
            "ip": master["ip"],
            "vram_gb": master.get("vram_gb", 32),
            "gpu_backend": master.get("gpu_backend", "cuda"),
            "name": master["hostname"],
        })
        for i, worker in enumerate(self.cluster.get("workers", [])):
            nodes.append({
                "rank": i + 1,
                "ip": worker["ip"],
                "vram_gb": worker.get("vram_gb", 24),
                "gpu_backend": worker.get("gpu_backend", "rocm"),
                "name": worker["hostname"],
            })
        return nodes

    def compute_shard_plan(self, model_size_gb: float, layer_count: int) -> dict:
        """
        Given a model size and layer count, compute how to split
        the model across available nodes.

        Returns a shard plan like:
        {
            "shards": [
                {"rank": 0, "layers": [0, 1, 2, 3], "estimated_gb": 8.5},
                {"rank": 1, "layers": [4, 5, 6, 7], "estimated_gb": 7.2},
                ...
            ]
        }
        """
        if model_size_gb <= self.threshold_gb:
            # Model fits on a single GPU — no sharding needed
            return {
                "sharded": False,
                "reason": f"Model ({model_size_gb:.1f}GB) fits within threshold ({self.threshold_gb}GB)",
                "shards": [{"rank": 0, "layers": list(range(layer_count)), "estimated_gb": model_size_gb}],
            }

        # Calculate available VRAM per node (leave 2GB headroom for ComfyUI + OS)
        headroom_gb = 2.0
        available = []
        for node in self.nodes:
            avail = node["vram_gb"] - headroom_gb
            available.append({"rank": node["rank"], "name": node["name"], "avail_gb": avail})

        total_avail = sum(n["avail_gb"] for n in available)

        if model_size_gb > total_avail:
            return {
                "sharded": False,
                "reason": f"Model ({model_size_gb:.1f}GB) exceeds total cluster VRAM ({total_avail:.1f}GB)",
                "error": True,
            }

        # Distribute layers proportionally to available VRAM
        gb_per_layer = model_size_gb / layer_count
        shards = []
        layer_idx = 0

        for node_info in available:
            if layer_idx >= layer_count:
                break

            # How many layers can this node hold?
            max_layers = int(node_info["avail_gb"] / gb_per_layer)
            max_layers = max(1, min(max_layers, layer_count - layer_idx))

            layers = list(range(layer_idx, layer_idx + max_layers))

            shards.append({
                "rank": node_info["rank"],
                "name": node_info["name"],
                "layers": layers,
                "estimated_gb": round(len(layers) * gb_per_layer, 1),
            })

            layer_idx += max_layers

        # Handle remaining layers (assign to node with most headroom)
        while layer_idx < layer_count:
            best = min(shards, key=lambda s: s["estimated_gb"])
            best["layers"].append(layer_idx)
            best["estimated_gb"] = round(len(best["layers"]) * gb_per_layer, 1)
            layer_idx += 1

        log.info(f"Shard plan for {model_size_gb:.1f}GB model ({layer_count} layers):")
        for shard in shards:
            log.info(f"  Rank {shard['rank']} ({shard['name']}): "
                     f"layers {shard['layers'][0]}-{shard['layers'][-1]} "
                     f"({shard['estimated_gb']}GB)")

        return {"sharded": True, "shards": shards}

    def split_state_dict(self, state_dict: dict, shard_plan: dict) -> list[dict]:
        """
        Split a model state_dict according to a shard plan.
        Each shard gets the layers assigned to it.
        """
        if not shard_plan.get("sharded", False):
            return [state_dict]

        # Group state_dict keys by layer number
        # Most transformer models use patterns like:
        #   model.layers.0.self_attn.q_proj.weight
        #   model.layers.15.mlp.gate_proj.weight
        layer_keys = {}
        shared_keys = {}  # Keys not assigned to a specific layer (embeddings, norm, etc.)

        for key, tensor in state_dict.items():
            # Try to extract layer number from key
            layer_num = self._extract_layer_num(key)
            if layer_num is not None:
                if layer_num not in layer_keys:
                    layer_keys[layer_num] = {}
                layer_keys[layer_num][key] = tensor
            else:
                shared_keys[key] = tensor

        # Build per-shard state dicts
        shard_dicts = []
        for shard in shard_plan["shards"]:
            shard_dict = {}
            # Add shared keys to rank 0 only (embeddings, final norm, etc.)
            if shard["rank"] == 0:
                shard_dict.update(shared_keys)
            for layer_num in shard["layers"]:
                if layer_num in layer_keys:
                    shard_dict.update(layer_keys[layer_num])
            shard_dicts.append(shard_dict)

        return shard_dicts

    def _extract_layer_num(self, key: str) -> Optional[int]:
        """Extract layer number from a state_dict key."""
        import re
        # Common patterns: layers.0., blocks.0., layer.0., h.0.
        match = re.search(r'(?:layers?|blocks?|h)\.(\d+)\.', key)
        if match:
            return int(match.group(1))
        return None


class ShardedModelWrapper(torch.nn.Module):
    """
    Wraps a sharded model for distributed inference.
    Each node holds a subset of layers and passes activations
    to the next node in the pipeline.
    """

    def __init__(self, local_layers: torch.nn.ModuleList, rank: int,
                 world_size: int, device: torch.device):
        super().__init__()
        self.local_layers = local_layers
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def forward(self, x):
        """
        Pipeline-parallel forward pass.
        Receive input from previous rank, process local layers,
        send output to next rank.
        """
        # Receive from previous rank (if not rank 0)
        if self.rank > 0:
            x = self._receive_tensor(self.rank - 1)
            x = x.to(self.device)

        # Process local layers
        for layer in self.local_layers:
            x = layer(x)

        # Send to next rank (if not last rank)
        if self.rank < self.world_size - 1:
            self._send_tensor(x, self.rank + 1)

        return x

    def _send_tensor(self, tensor: torch.Tensor, dst_rank: int):
        """Send tensor to another rank via CPU (gloo)."""
        cpu_tensor = tensor.detach().cpu().contiguous()
        # Send shape first
        shape_tensor = torch.tensor(list(cpu_tensor.shape), dtype=torch.long)
        dist.send(shape_tensor, dst=dst_rank)
        dist.send(cpu_tensor, dst=dst_rank)

    def _receive_tensor(self, src_rank: int) -> torch.Tensor:
        """Receive tensor from another rank via CPU (gloo)."""
        # Receive shape first
        shape_tensor = torch.zeros(8, dtype=torch.long)  # Max 8 dims
        dist.recv(shape_tensor, src=src_rank)
        shape = [s.item() for s in shape_tensor if s.item() > 0]
        # Receive data
        tensor = torch.zeros(*shape)
        dist.recv(tensor, src=src_rank)
        return tensor


# ============================================================
# HTTP API for the coordinator (called by ComfyUI custom node)
# ============================================================
async def run_coordinator_server(config_path: str, port: int = 29600):
    """Run the shard coordinator as an HTTP server."""
    from aiohttp import web

    coordinator = ModelShardCoordinator(config_path)

    async def handle_plan(request):
        """POST /plan — compute a shard plan for a model."""
        data = await request.json()
        model_size_gb = data.get("model_size_gb", 0)
        layer_count = data.get("layer_count", 1)
        plan = coordinator.compute_shard_plan(model_size_gb, layer_count)
        return web.json_response(plan)

    async def handle_info(request):
        """GET /info — cluster info for sharding."""
        return web.json_response({
            "nodes": coordinator.nodes,
            "threshold_gb": coordinator.threshold_gb,
            "total_vram_gb": sum(n["vram_gb"] for n in coordinator.nodes),
        })

    app = web.Application()
    app.router.add_post("/plan", handle_plan)
    app.router.add_get("/info", handle_info)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info(f"Shard coordinator running on port {port}")
    log.info(f"Nodes: {len(coordinator.nodes)}, "
             f"Shard threshold: {coordinator.threshold_gb}GB")

    # Keep running
    while True:
        await asyncio.sleep(3600)


def main():
    parser = argparse.ArgumentParser(description="Model Shard Coordinator")
    parser.add_argument("--config", default="cluster_config.yaml")
    parser.add_argument("--port", type=int, default=29600)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    asyncio.run(run_coordinator_server(config_path, args.port))


if __name__ == "__main__":
    main()
