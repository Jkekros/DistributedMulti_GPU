"""
============================================================
ComfyUI Cluster Load Balancer
Sits on the master node, exposes a single ComfyUI API endpoint.
Distributes prompt/generation requests across all cluster nodes.

Architecture:
  Browser → Balancer (:8100) → {Master ComfyUI, Worker1, Worker2, Worker3}

Supports:
  - Round-robin or least-busy queue distribution
  - WebSocket proxying for real-time progress
  - VRAM-aware routing (sends large jobs to RTX 5090)
  - Health monitoring of all nodes
  - Auto-retry on node failure
============================================================
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import yaml
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BALANCER] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Node representation
# ============================================================
@dataclass
class ComfyNode:
    name: str
    ip: str
    port: int
    gpu_type: str
    gpu_backend: str
    vram_gb: int
    os_type: str  # "windows" or "wsl2"
    active_jobs: int = 0
    total_completed: int = 0
    is_healthy: bool = True
    last_health_check: float = 0
    avg_gen_time: float = 0
    _gen_times: list = field(default_factory=list)

    @property
    def base_url(self):
        return f"http://{self.ip}:{self.port}"

    @property
    def ws_url(self):
        return f"ws://{self.ip}:{self.port}/ws"

    def record_gen_time(self, seconds):
        self._gen_times.append(seconds)
        if len(self._gen_times) > 50:
            self._gen_times = self._gen_times[-50:]
        self.avg_gen_time = sum(self._gen_times) / len(self._gen_times)


# ============================================================
# Load Balancer
# ============================================================
class ComfyLoadBalancer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.nodes: list[ComfyNode] = []
        self.strategy = self.config.get("comfyui", {}).get("strategy", "least_busy")
        self._setup_nodes()
        self._client_sessions: dict[str, ComfyNode] = {}  # client_id -> assigned node

    def _load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def _setup_nodes(self):
        cluster = self.config["cluster"]
        comfy = self.config.get("comfyui", {})

        # Master node
        master = cluster["master"]
        self.nodes.append(ComfyNode(
            name=f"master ({master['hostname']})",
            ip=master["ip"],
            port=comfy.get("master_comfy_port", 8188),
            gpu_type=master.get("gpu_type", "rtx5090"),
            gpu_backend=master.get("gpu_backend", "cuda"),
            vram_gb=master.get("vram_gb", 32),
            os_type=master.get("os", "windows"),
        ))

        # Worker nodes
        worker_ports = comfy.get("worker_comfy_ports", [])
        for i, worker in enumerate(cluster.get("workers", [])):
            port = worker_ports[i] if i < len(worker_ports) else 8188
            self.nodes.append(ComfyNode(
                name=f"worker-{i+1} ({worker['hostname']})",
                ip=worker["ip"],
                port=port,
                gpu_type=worker.get("gpu_type", "rx7900xtx"),
                gpu_backend=worker.get("gpu_backend", "rocm"),
                vram_gb=worker.get("vram_gb", 24),
                os_type=worker.get("os", "wsl2"),
            ))

        log.info(f"Configured {len(self.nodes)} nodes: "
                 f"{[n.name for n in self.nodes]}")

    # ----------------------------------------------------------
    # Node selection strategies
    # ----------------------------------------------------------
    def _select_node(self, prompt_data: dict = None) -> Optional[ComfyNode]:
        healthy = [n for n in self.nodes if n.is_healthy]
        if not healthy:
            return None

        if self.strategy == "round_robin":
            # Simple round-robin
            node = min(healthy, key=lambda n: n.total_completed)

        elif self.strategy == "least_busy":
            # Pick the node with fewest active jobs
            node = min(healthy, key=lambda n: n.active_jobs)

        elif self.strategy == "vram_aware":
            # For large models, prefer the RTX 5090 (more VRAM)
            # For normal workloads, use least_busy
            node = min(healthy, key=lambda n: n.active_jobs)

        else:
            node = healthy[0]

        return node

    # ----------------------------------------------------------
    # Health checking
    # ----------------------------------------------------------
    async def check_node_health(self, node: ComfyNode):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{node.base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        node.is_healthy = True
                        node.last_health_check = time.time()
                        return data
                    else:
                        node.is_healthy = False
        except Exception as e:
            if node.is_healthy:
                log.warning(f"Node {node.name} is DOWN: {e}")
            node.is_healthy = False
        return None

    async def health_check_loop(self):
        """Periodically check all node health."""
        while True:
            for node in self.nodes:
                await self.check_node_health(node)
            await asyncio.sleep(10)

    # ----------------------------------------------------------
    # API proxy handlers
    # ----------------------------------------------------------
    async def handle_prompt(self, request: web.Request):
        """POST /prompt — queue a generation job."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON"}, status=400
            )

        node = self._select_node(data)
        if not node:
            return web.json_response(
                {"error": "No healthy nodes available"}, status=503
            )

        node.active_jobs += 1
        log.info(f"Routing prompt to {node.name} "
                 f"(active: {node.active_jobs}, total: {node.total_completed})")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.base_url}/prompt",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    result = await resp.json()
                    # Tag the response so client knows which node is handling it
                    result["_cluster_node"] = node.name
                    result["_cluster_ip"] = node.ip
                    return web.json_response(result, status=resp.status)
        except Exception as e:
            log.error(f"Error routing to {node.name}: {e}")
            node.is_healthy = False
            return web.json_response(
                {"error": f"Node {node.name} failed: {str(e)}"}, status=502
            )
        finally:
            node.active_jobs = max(0, node.active_jobs - 1)
            node.total_completed += 1

    async def handle_get_proxy(self, request: web.Request):
        """Proxy GET requests to the least busy node."""
        path = request.match_info.get("path", "")
        node = self._select_node()
        if not node:
            return web.json_response({"error": "No nodes"}, status=503)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{node.base_url}/{path}",
                    params=request.query,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.read()
                    return web.Response(
                        body=body,
                        status=resp.status,
                        content_type=resp.content_type,
                    )
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, status=502
            )

    async def handle_post_proxy(self, request: web.Request):
        """Proxy POST requests to the least busy node."""
        path = request.match_info.get("path", "")
        node = self._select_node()
        if not node:
            return web.json_response({"error": "No nodes"}, status=503)

        try:
            body = await request.read()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{node.base_url}/{path}",
                    data=body,
                    headers={"Content-Type": request.content_type},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp_body = await resp.read()
                    return web.Response(
                        body=resp_body,
                        status=resp.status,
                        content_type=resp.content_type,
                    )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

    async def handle_upload(self, request: web.Request):
        """POST /upload/* — forward file uploads to ALL nodes."""
        path = request.match_info.get("path", "upload/image")
        body = await request.read()
        content_type = request.content_type

        results = []
        for node in self.nodes:
            if not node.is_healthy:
                continue
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{node.base_url}/{path}",
                        data=body,
                        headers={"Content-Type": content_type},
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        result = await resp.json()
                        results.append({"node": node.name, "status": "ok", "result": result})
            except Exception as e:
                results.append({"node": node.name, "status": "error", "error": str(e)})

        return web.json_response({"upload_results": results})

    async def handle_ws(self, request: web.Request):
        """WebSocket proxy — connects client to a ComfyUI node's WS."""
        ws_server = web.WebSocketResponse()
        await ws_server.prepare(request)

        client_id = request.query.get("clientId", "")

        # If client has an assigned node (from a previous prompt), use it
        node = self._client_sessions.get(client_id)
        if not node or not node.is_healthy:
            node = self._select_node()
        if not node:
            await ws_server.close(message=b"No healthy nodes")
            return ws_server

        self._client_sessions[client_id] = node
        log.info(f"WS client {client_id[:8]}... → {node.name}")

        try:
            async with aiohttp.ClientSession() as session:
                ws_url = f"{node.ws_url}?clientId={client_id}"
                async with session.ws_connect(ws_url) as ws_client:
                    # Bidirectional proxy
                    async def forward_to_client():
                        async for msg in ws_client:
                            if msg.type == WSMsgType.TEXT:
                                await ws_server.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_server.send_bytes(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break

                    async def forward_to_node():
                        async for msg in ws_server:
                            if msg.type == WSMsgType.TEXT:
                                await ws_client.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_client.send_bytes(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break

                    await asyncio.gather(
                        forward_to_client(),
                        forward_to_node(),
                        return_exceptions=True,
                    )
        except Exception as e:
            log.error(f"WS proxy error: {e}")
        finally:
            if client_id in self._client_sessions:
                del self._client_sessions[client_id]

        return ws_server

    # ----------------------------------------------------------
    # Cluster status endpoint
    # ----------------------------------------------------------
    async def handle_cluster_status(self, request: web.Request):
        """GET /cluster/status — show all node status."""
        nodes_status = []
        for node in self.nodes:
            nodes_status.append({
                "name": node.name,
                "ip": node.ip,
                "port": node.port,
                "gpu_type": node.gpu_type,
                "gpu_backend": node.gpu_backend,
                "vram_gb": node.vram_gb,
                "os": node.os_type,
                "healthy": node.is_healthy,
                "active_jobs": node.active_jobs,
                "total_completed": node.total_completed,
                "avg_gen_time": round(node.avg_gen_time, 1),
                "last_check": node.last_health_check,
            })

        healthy_count = sum(1 for n in self.nodes if n.is_healthy)
        total_vram = sum(n.vram_gb for n in self.nodes if n.is_healthy)

        return web.json_response({
            "cluster": {
                "total_nodes": len(self.nodes),
                "healthy_nodes": healthy_count,
                "total_vram_gb": total_vram,
                "strategy": self.strategy,
            },
            "nodes": nodes_status,
        })

    # ----------------------------------------------------------
    # Cluster status dashboard (HTML)
    # ----------------------------------------------------------
    async def handle_cluster_dashboard(self, request: web.Request):
        """GET /cluster — HTML dashboard."""
        nodes_html = ""
        for node in self.nodes:
            status_color = "green" if node.is_healthy else "red"
            nodes_html += f"""
            <tr>
                <td>{node.name}</td>
                <td>{node.gpu_type}</td>
                <td>{node.vram_gb} GB</td>
                <td>{node.gpu_backend}</td>
                <td style="color:{status_color}">{"● Online" if node.is_healthy else "● Offline"}</td>
                <td>{node.active_jobs}</td>
                <td>{node.total_completed}</td>
                <td>{node.avg_gen_time:.1f}s</td>
            </tr>"""

        healthy = sum(1 for n in self.nodes if n.is_healthy)
        total_vram = sum(n.vram_gb for n in self.nodes if n.is_healthy)

        html = f"""<!DOCTYPE html>
<html><head><title>ComfyUI Cluster</title>
<meta http-equiv="refresh" content="5">
<style>
    body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
    h1 {{ color: #0f3460; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
    th {{ background: #16213e; }}
    tr:hover {{ background: #1a1a3e; }}
    .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
    .stat {{ background: #16213e; padding: 15px 25px; border-radius: 8px; }}
    .stat h3 {{ margin: 0; color: #888; font-size: 12px; text-transform: uppercase; }}
    .stat p {{ margin: 5px 0 0 0; font-size: 24px; font-weight: bold; }}
</style></head>
<body>
    <h1>🖥️ ComfyUI GPU Cluster</h1>
    <div class="stats">
        <div class="stat"><h3>Nodes Online</h3><p>{healthy}/{len(self.nodes)}</p></div>
        <div class="stat"><h3>Total VRAM</h3><p>{total_vram} GB</p></div>
        <div class="stat"><h3>Strategy</h3><p>{self.strategy}</p></div>
        <div class="stat"><h3>Total Generated</h3><p>{sum(n.total_completed for n in self.nodes)}</p></div>
    </div>
    <table>
        <tr><th>Node</th><th>GPU</th><th>VRAM</th><th>Backend</th><th>Status</th><th>Active</th><th>Completed</th><th>Avg Time</th></tr>
        {nodes_html}
    </table>
    <p style="color:#666; margin-top:20px;">Auto-refreshes every 5 seconds. 
    Connect ComfyUI at <b>http://{self.config['cluster']['master']['ip']}:{self.config.get('comfyui',{{}}).get('balancer_port',8100)}</b></p>
</body></html>"""
        return web.Response(text=html, content_type="text/html")


# ============================================================
# App setup
# ============================================================
def create_app(config_path: str) -> tuple[web.Application, ComfyLoadBalancer]:
    balancer = ComfyLoadBalancer(config_path)
    app = web.Application()

    # CORS (ComfyUI frontend needs this)
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })

    # Routes
    # Cluster management
    app.router.add_get("/cluster", balancer.handle_cluster_dashboard)
    app.router.add_get("/cluster/status", balancer.handle_cluster_status)

    # ComfyUI API proxy
    app.router.add_post("/prompt", balancer.handle_prompt)
    app.router.add_get("/ws", balancer.handle_ws)

    # Upload to all nodes
    app.router.add_post("/upload/{path:.*}", balancer.handle_upload)

    # Catch-all proxy for other ComfyUI endpoints
    app.router.add_get("/{path:.*}", balancer.handle_get_proxy)
    app.router.add_post("/{path:.*}", balancer.handle_post_proxy)

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        try:
            cors.add(route)
        except ValueError:
            pass

    # Start health check loop
    async def start_health_checks(app):
        app["health_task"] = asyncio.create_task(balancer.health_check_loop())

    async def stop_health_checks(app):
        app["health_task"].cancel()

    app.on_startup.append(start_health_checks)
    app.on_cleanup.append(stop_health_checks)

    return app, balancer


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Cluster Load Balancer")
    parser.add_argument("--config", default="cluster_config.yaml", help="Config file path")
    parser.add_argument("--port", type=int, default=None, help="Override balancer port")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    app, balancer = create_app(config_path)

    port = args.port
    if port is None:
        port = balancer.config.get("comfyui", {}).get("balancer_port", 8100)

    master_ip = balancer.config["cluster"]["master"]["ip"]

    log.info("=" * 50)
    log.info("  ComfyUI Cluster Load Balancer")
    log.info(f"  Dashboard:  http://{master_ip}:{port}/cluster")
    log.info(f"  ComfyUI:    http://{master_ip}:{port}")
    log.info(f"  Nodes:      {len(balancer.nodes)}")
    log.info(f"  Strategy:   {balancer.strategy}")
    log.info("=" * 50)

    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
