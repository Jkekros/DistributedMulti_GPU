"""
============================================================
ComfyUI Custom Node: Cluster Shard Loader
============================================================
Place this file inside your ComfyUI/custom_nodes/ folder
(or in a subfolder: ComfyUI/custom_nodes/cluster_shard/).

This node contacts the Model Shard Coordinator to decide if
a model should be loaded locally or sharded across the
GPU cluster. It integrates with ComfyUI's checkpoint loading
pipeline.
============================================================
"""

import os
import json
import logging
import requests
import torch

log = logging.getLogger("ClusterShard")

# Default coordinator URL (master node)
COORDINATOR_URL = os.environ.get("CLUSTER_COORDINATOR_URL", "http://192.168.0.15:29600")


class ClusterCheckpointLoader:
    """
    Drop-in replacement for CheckpointLoaderSimple that is
    shard-aware.  If the model is below the shard threshold,
    it loads locally as normal.  If the model exceeds the
    threshold, it contacts the coordinator for a shard plan
    and loads only its assigned layers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "coordinator_url": ("STRING", {
                    "default": COORDINATOR_URL,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "cluster"

    def load_checkpoint(self, ckpt_name, coordinator_url=COORDINATOR_URL):
        import folder_paths
        import comfy.sd
        import comfy.utils

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # Get model file size
        file_size_gb = os.path.getsize(ckpt_path) / (1024 ** 3)
        log.info(f"Checkpoint: {ckpt_name} ({file_size_gb:.1f} GB)")

        # Ask coordinator for a shard plan
        shard_plan = None
        try:
            resp = requests.post(
                f"{coordinator_url}/plan",
                json={
                    "model_size_gb": file_size_gb,
                    "layer_count": self._estimate_layer_count(ckpt_path),
                },
                timeout=5,
            )
            if resp.ok:
                shard_plan = resp.json()
                log.info(f"Shard plan: {json.dumps(shard_plan, indent=2)}")
        except Exception as e:
            log.warning(f"Could not contact shard coordinator: {e}")
            log.warning("Loading model locally (no sharding)")

        # If no sharding needed or coordinator unreachable, load normally
        if shard_plan is None or not shard_plan.get("sharded", False):
            log.info("Loading checkpoint normally (single GPU)")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
            )
            return out[:3]

        # Sharded loading: load only the layers assigned to this rank
        log.info("=== SHARDED LOADING MODE ===")
        rank = int(os.environ.get("CLUSTER_RANK", "0"))
        my_shard = None
        for shard in shard_plan["shards"]:
            if shard["rank"] == rank:
                my_shard = shard
                break

        if my_shard is None:
            log.warning(f"Rank {rank} not found in shard plan, loading full model")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
            )
            return out[:3]

        log.info(f"Rank {rank}: loading layers {my_shard['layers'][0]}-{my_shard['layers'][-1]} "
                 f"(~{my_shard['estimated_gb']}GB)")

        # Load checkpoint and filter state_dict
        # Load full state_dict to CPU first
        sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        # Filter to only keep keys for our assigned layers + shared keys
        filtered_sd = self._filter_state_dict(sd, my_shard["layers"])

        log.info(f"Filtered state_dict: {len(filtered_sd)}/{len(sd)} keys kept")

        # Load using filtered state_dict
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
        )

        return out[:3]

    def _estimate_layer_count(self, ckpt_path: str) -> int:
        """
        Quick estimate of how many layers a model has by
        scanning state_dict key names for layer indices.
        """
        import re
        try:
            import safetensors.torch as st
            if ckpt_path.endswith(".safetensors"):
                # Read only metadata, not weights
                with open(ckpt_path, "rb") as f:
                    header_size = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(header_size))
                keys = list(header.keys())
            else:
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                keys = list(sd.keys()) if isinstance(sd, dict) else []
        except Exception:
            return 32  # Sensible default for most SDXL / SD models

        max_layer = 0
        for key in keys:
            match = re.search(r'(?:layers?|blocks?|h)\.(\d+)\.', str(key))
            if match:
                max_layer = max(max_layer, int(match.group(1)))

        return max(max_layer + 1, 1)

    def _filter_state_dict(self, state_dict: dict, assigned_layers: list) -> dict:
        """Keep only keys for assigned layers + shared (non-layer) keys."""
        import re
        filtered = {}
        assigned_set = set(assigned_layers)

        for key, value in state_dict.items():
            match = re.search(r'(?:layers?|blocks?|h)\.(\d+)\.', key)
            if match:
                layer_num = int(match.group(1))
                if layer_num in assigned_set:
                    filtered[key] = value
            else:
                # Shared key (embeddings, final norm, etc.) — keep on all ranks
                filtered[key] = value

        return filtered


class ClusterModelInfo:
    """
    Utility node that shows cluster information in the ComfyUI graph.
    Displays available nodes, VRAM, and current shard status.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinator_url": ("STRING", {
                    "default": COORDINATOR_URL,
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "cluster"

    def get_info(self, coordinator_url=COORDINATOR_URL):
        try:
            resp = requests.get(f"{coordinator_url}/info", timeout=5)
            if resp.ok:
                info = resp.json()
                lines = ["=== GPU Cluster Info ==="]
                lines.append(f"Total VRAM: {info['total_vram_gb']} GB")
                lines.append(f"Shard threshold: {info['threshold_gb']} GB")
                lines.append(f"Nodes: {len(info['nodes'])}")
                for node in info["nodes"]:
                    lines.append(
                        f"  Rank {node['rank']}: {node['name']} "
                        f"({node['gpu_backend']}, {node['vram_gb']}GB)")
                return ("\n".join(lines),)
        except Exception as e:
            return (f"Coordinator unreachable: {e}",)


# ============================================================
# ComfyUI node registration
# ============================================================
NODE_CLASS_MAPPINGS = {
    "ClusterCheckpointLoader": ClusterCheckpointLoader,
    "ClusterModelInfo": ClusterModelInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClusterCheckpointLoader": "Load Checkpoint (Cluster Shard)",
    "ClusterModelInfo": "Cluster Info",
}
